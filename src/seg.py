"""Config-driven multi-class conjunctiva segmentation: {bg, palpebral, forniceal}.

Device-agnostic: CUDA (H100/HPC) > MPS (local mac) > CPU, auto-detected.
Architectures (smp): unet | unetplusplus | segformer | deeplabv3plus | manet, with any smp
encoder (efficientnet-b*, mit_b* [=SegFormer backbone], resnext, ...).

Evaluation (mandatory): pooled k-fold CV (cohort-stratified) AND leave-one-cohort-out (LOCO),
plus an optional `final` model trained on all data. Metrics: per-class IoU + Dice, reported at
the converged FINAL epoch (in CV/LOCO the val split == the test split, so best-epoch selection
would peek).

Usage:
  python seg.py --config configs/seg_smoke_local.yaml
  python seg.py --config configs/seg_heavy_segformer.yaml
  python seg.py --config <cfg> --mode cv --subset 40     # CLI overrides for quick checks
"""
import os, sys, json, time, argparse, random
import numpy as np
import cv2
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import segmentation_models_pytorch as smp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import CACHE, SEG_OUT as OUT_ROOT, MANIFEST
NUM_CLASSES = 3
CLASS_NAMES = ["background", "palpebral", "forniceal"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()

# MPS-only: backprop through some .view ops fails on non-contiguous strides (torch 2.12).
# Route .view -> .reshape (contiguous-safe). NOT applied on CUDA (native, faster, and MiT
# transformer backbones train fine there).
if DEVICE.type == "mps":
    _orig_view = torch.Tensor.view
    def _safe_view(self, *shape, **kw):
        if kw or (len(shape) == 1 and isinstance(shape[0], torch.dtype)):
            return _orig_view(self, *shape, **kw)
        return self.reshape(*shape)
    torch.Tensor.view = _safe_view


def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ----------------------------- model -----------------------------
def build_model(cfg):
    arch = cfg["arch"].lower()
    kw = dict(encoder_name=cfg["encoder"], encoder_weights=cfg.get("encoder_weights", "imagenet"),
              in_channels=3, classes=NUM_CLASSES)
    table = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus, "unetpp": smp.UnetPlusPlus,
             "segformer": smp.Segformer, "deeplabv3plus": smp.DeepLabV3Plus, "manet": smp.MAnet,
             "fpn": smp.FPN, "pspnet": smp.PSPNet}
    if arch not in table:
        raise ValueError(f"unknown arch {arch}; choose {list(table)}")
    return table[arch](**kw)


# ----------------------------- data / aug -----------------------------
def build_aug(cfg, train):
    res = cfg["res"]
    ops = [A.Resize(res, res, interpolation=cv2.INTER_AREA, mask_interpolation=cv2.INTER_NEAREST)]
    if train:
        level = cfg.get("aug", "heavy")
        ops += [A.HorizontalFlip(p=0.5),
                A.Affine(scale=(0.85, 1.15), translate_percent=(0.0, 0.08), rotate=(-20, 20),
                         shear=(-6, 6), p=0.8, interpolation=cv2.INTER_LINEAR,
                         mask_interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT,
                         fill=0, fill_mask=0)]
        if level == "heavy":
            ops += [
                A.OneOf([A.GridDistortion(num_steps=5, distort_limit=0.2),
                         A.ElasticTransform(alpha=30, sigma=6)], p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
                A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=12, val_shift_limit=8, p=0.3),
                A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.05, 0.12),
                                hole_width_range=(0.05, 0.12), fill=0, p=0.2),
            ]
        else:  # light
            ops += [A.RandomBrightnessContrast(0.1, 0.1, p=0.3)]
    ops.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return A.Compose(ops)


class SegDS(Dataset):
    def __init__(self, uids, train, cfg):
        self.uids = list(uids)
        self.tf = build_aug(cfg, train)

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        uid = self.uids[i]
        img = cv2.imread(os.path.join(CACHE, "img", uid + ".png"))[:, :, ::-1]
        lbl = cv2.imread(os.path.join(CACHE, "lbl", uid + ".png"), 0)
        a = self.tf(image=np.ascontiguousarray(img), mask=lbl)
        x = torch.from_numpy(a["image"].transpose(2, 0, 1)).float()
        y = torch.from_numpy(a["mask"]).long()
        return x, y


# ----------------------------- loss & metrics -----------------------------
class DiceCELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        probs = F.softmax(logits, 1)
        oh = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * oh).sum(dims)
        denom = probs.sum(dims) + oh.sum(dims)
        dice = (2 * inter + 1.0) / (denom + 1.0)
        return ce + (1 - dice.mean())


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    inter = np.zeros(NUM_CLASSES); union = np.zeros(NUM_CLASSES)
    psum = np.zeros(NUM_CLASSES); gsum = np.zeros(NUM_CLASSES)
    for x, y in loader:
        logits = model(x.to(DEVICE))
        pred = logits.argmax(1).cpu().numpy()
        gt = y.numpy()
        for c in range(NUM_CLASSES):
            p = pred == c; g = gt == c
            inter[c] += np.logical_and(p, g).sum()
            union[c] += np.logical_or(p, g).sum()
            psum[c] += p.sum(); gsum[c] += g.sum()
    iou = inter / np.maximum(union, 1)
    dice = 2 * inter / np.maximum(psum + gsum, 1)
    return {"iou": iou.tolist(), "dice": dice.tolist()}


def class_weights(uids):
    cnt = np.zeros(NUM_CLASSES)
    for uid in uids:
        lbl = cv2.imread(os.path.join(CACHE, "lbl", uid + ".png"), 0)
        for c in range(NUM_CLASSES):
            cnt[c] += (lbl == c).sum()
    freq = cnt / cnt.sum()
    w = 1.0 / np.sqrt(freq + 1e-6)
    return torch.tensor((w / w.mean()), dtype=torch.float32)


# ----------------------------- train one split -----------------------------
def train_split(train_uids, val_uids, cfg, tag, save_path=None):
    seed_all(cfg.get("seed", 42))
    nw = cfg.get("num_workers", 2)
    bs = cfg["bs"]
    tr = DataLoader(SegDS(train_uids, True, cfg), batch_size=bs, shuffle=True, num_workers=nw,
                    drop_last=len(train_uids) > bs, persistent_workers=nw > 0, pin_memory=DEVICE.type == "cuda")
    va = (DataLoader(SegDS(val_uids, False, cfg), batch_size=bs, shuffle=False, num_workers=nw,
                     pin_memory=DEVICE.type == "cuda") if val_uids is not None and len(val_uids) else None)
    model = build_model(cfg).to(DEVICE)
    # opt-in multi-GPU (e.g. Kaggle T4 x2): split each batch across cards. Single-process, no
    # code change for HPC/local. Guarded by device count so it's a no-op on 1 GPU / MPS / CPU.
    if cfg.get("multi_gpu") and DEVICE.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"DataParallel across {torch.cuda.device_count()} GPUs", flush=True)
    crit = DiceCELoss(class_weights(train_uids).to(DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-2))
    epochs = cfg["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = bool(cfg.get("amp", False)) and DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best = {"miou_fg": -1}; m_last = None
    eval_every = max(1, epochs // 10)
    accum = max(1, cfg.get("grad_accum", 1))   # emulate big batch on small (Kaggle 16GB) GPUs
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = 0.0; opt.zero_grad()
        for i, (x, y) in enumerate(tr):
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
                loss = crit(model(x), y) / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad()
            tot += loss.item() * accum
        sched.step()
        msg = f"[{tag}] ep{ep+1}/{epochs} loss {tot/len(tr):.4f} ({time.time()-t0:.0f}s)"
        if va is not None and ((ep + 1) % eval_every == 0 or ep == epochs - 1):
            m = evaluate(model, va)
            miou_fg = float(np.mean(m["iou"][1:]))
            msg += f" | val IoU {[round(v,3) for v in m['iou']]} mIoU_fg {miou_fg:.3f}"
            m_last = m
            if miou_fg > best["miou_fg"]:
                best = {"miou_fg": miou_fg, **m}
        print(msg, flush=True)
    reported = m_last if va is not None else evaluate(model, tr)
    if save_path:
        # unwrap DataParallel so checkpoint keys have NO 'module.' prefix -> infer loads cleanly
        core = model.module if isinstance(model, nn.DataParallel) else model
        torch.save({"state_dict": core.state_dict(), "cfg": cfg}, save_path)
    return reported, model


# ----------------------------- protocols -----------------------------
def load_manifest(cfg):
    man = pd.read_csv(MANIFEST)
    sub = cfg.get("subset")
    if sub:  # quick smoke: balanced-ish subset across cohorts
        man = pd.concat([g.head(max(1, sub // 2)) for _, g in man.groupby("cohort")]).reset_index(drop=True)
    return man


def agg(results):
    iou = np.array([r["iou"] for r in results]); dice = np.array([r["dice"] for r in results])
    return {"iou_mean": iou.mean(0).tolist(), "iou_std": iou.std(0).tolist(),
            "dice_mean": dice.mean(0).tolist(), "dice_std": dice.std(0).tolist(),
            "miou_fg": float(iou[:, 1:].mean()), "mdice_fg": float(dice[:, 1:].mean())}


def run_cv(cfg, outdir):
    man = load_manifest(cfg)
    uids = man["uid"].values; strat = man["cohort"].values
    folds = cfg.get("folds", 5)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg.get("seed", 42))
    res = []
    for k, (tri, vai) in enumerate(skf.split(uids, strat)):
        r, _ = train_split(uids[tri], uids[vai], cfg, f"cv{k}")
        print(f"== CV fold {k}: IoU {[round(v,3) for v in r['iou']]} Dice {[round(v,3) for v in r['dice']]}", flush=True)
        res.append(r)
    return {"protocol": "pooled_kfold", "folds": folds, "config": cfg,
            "classes": CLASS_NAMES, "per_fold": res, **agg(res)}


def run_loco(cfg, outdir):
    man = load_manifest(cfg)
    out = {}
    for tr_c, te_c in [("India", "Italy"), ("Italy", "India")]:
        tr = man[man.cohort == tr_c]["uid"].values
        te = man[man.cohort == te_c]["uid"].values
        r, _ = train_split(tr, te, cfg, f"loco_{tr_c[:2]}->{te_c[:2]}")
        print(f"== LOCO {tr_c}->{te_c}: IoU {[round(v,3) for v in r['iou']]} Dice {[round(v,3) for v in r['dice']]}", flush=True)
        out[f"train_{tr_c}_test_{te_c}"] = r
    return {"protocol": "leave_one_cohort_out", "config": cfg, "classes": CLASS_NAMES, "results": out}


def run_final(cfg, outdir):
    man = load_manifest(cfg)
    save = os.path.join(outdir, f"{cfg['name']}_final.pt")
    r, _ = train_split(man["uid"].values, None, cfg, "final", save_path=save)
    return {"protocol": "final_all_data", "config": cfg, "weights": save, "train_fit": r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default=None, help="override config modes (cv|loco|final)")
    ap.add_argument("--subset", type=int, default=None, help="override: use N patients (smoke)")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.subset is not None: cfg["subset"] = args.subset
    if args.epochs is not None: cfg["epochs"] = args.epochs
    modes = [args.mode] if args.mode else cfg.get("modes", ["cv"])
    outdir = os.path.join(OUT_ROOT, cfg.get("out_subdir", cfg["name"]))
    os.makedirs(outdir, exist_ok=True)
    print(f"device={DEVICE} config={cfg['name']} arch={cfg['arch']} encoder={cfg['encoder']} "
          f"res={cfg['res']} bs={cfg['bs']} epochs={cfg['epochs']} amp={cfg.get('amp')} modes={modes}", flush=True)

    fns = {"cv": run_cv, "loco": run_loco, "final": run_final}
    for mode in modes:
        summary = fns[mode](cfg, outdir)
        outp = os.path.join(outdir, f"seg_{mode}_{cfg['name']}.json")
        with open(outp, "w") as f:
            json.dump(summary, f, indent=2, default=float)
        print(f"WROTE {outp}", flush=True)
        if "miou_fg" in summary:
            print(f"  mIoU_fg={summary['miou_fg']:.4f} mDice_fg={summary['mdice_fg']:.4f}", flush=True)


if __name__ == "__main__":
    main()
