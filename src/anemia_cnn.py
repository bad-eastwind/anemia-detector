"""Deep anemia classifier on the masked conjunctiva ROI (the (b)-route: ViT/CNN on ROI).

Complements anemia_clf.py (handcrafted features + GBM). Config-driven, device-agnostic
(CUDA > MPS > CPU). Input = union-mask ROI: crop the conjunctiva bbox (with margin), zero out
non-ROI pixels, resize. Two heads:
  target: classify  -> BCE, sigmoid proba
  target: regress   -> predict Hgb, apply WHO sex threshold (prevalence-robust; best for LOCO)
Protocols: pooled k-fold CV (cohort x label stratified) + leave-one-cohort-out.

Usage:
  python anemia_cnn.py --config configs/anemia_cnn_smoke_local.yaml
  python anemia_cnn.py --config configs/anemia_cnn_hpc_vit.yaml
"""
import os, sys, json, argparse, random, time
import numpy as np
import cv2
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import pandas as pd
import timm
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from anemia_clf import metrics, report, who_label  # reuse exact metric defs
CACHE = os.path.join(HERE, "..", "outputs", "cache")
MAN = os.path.join(HERE, "..", "outputs", "manifest.csv")
OUT_ROOT = os.path.join(HERE, "..", "outputs", "clf")
MEAN = (0.485, 0.456, 0.406); STD = (0.229, 0.224, 0.225)


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
if DEVICE.type == "mps":
    _ov = torch.Tensor.view
    def _sv(self, *s, **k):
        if k or (len(s) == 1 and isinstance(s[0], torch.dtype)): return _ov(self, *s, **k)
        return self.reshape(*s)
    torch.Tensor.view = _sv


def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def crop_roi(img, roi, margin=0.12):
    ys, xs = np.where(roi)
    if len(ys) == 0:
        return img
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = roi.shape
    my, mx = int((y1 - y0) * margin), int((x1 - x0) * margin)
    y0, y1 = max(0, y0 - my), min(h, y1 + my); x0, x1 = max(0, x0 - mx), min(w, x1 + mx)
    out = img.copy()
    out[~roi] = 0
    return out[y0:y1, x0:x1]


class ROIDS(Dataset):
    def __init__(self, df, train, cfg):
        self.df = df.reset_index(drop=True)
        res = cfg["res"]
        if train:
            self.tf = A.Compose([
                A.Resize(res, res), A.HorizontalFlip(p=0.5),
                A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), translate_percent=(0, 0.06), p=0.7),
                A.RandomBrightnessContrast(0.12, 0.12, p=0.4),
                A.Normalize(MEAN, STD)])
        else:
            self.tf = A.Compose([A.Resize(res, res), A.Normalize(MEAN, STD)])
        self.target = cfg.get("target", "classify")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(os.path.join(CACHE, "img", r["uid"] + ".png"))[:, :, ::-1]
        roi = cv2.imread(os.path.join(CACHE, "roi", r["uid"] + ".png"), 0) > 127
        crop = crop_roi(np.ascontiguousarray(img), roi)
        x = torch.from_numpy(self.tf(image=crop)["image"].transpose(2, 0, 1)).float()
        y = float(r["hgb"]) if self.target == "regress" else float(r["anemia"])
        return x, torch.tensor(y, dtype=torch.float32)


def build_backbone(cfg):
    return timm.create_model(cfg["backbone"], pretrained=True, num_classes=1).to(DEVICE)


def train_one(tr_df, te_df, cfg, hgb_mean=0.0, hgb_std=1.0):
    seed_all(cfg.get("seed", 42))
    nw = cfg.get("num_workers", 2); bs = cfg["bs"]; target = cfg.get("target", "classify")
    tr = DataLoader(ROIDS(tr_df, True, cfg), batch_size=bs, shuffle=True, num_workers=nw,
                    drop_last=len(tr_df) > bs, persistent_workers=nw > 0, pin_memory=DEVICE.type == "cuda")
    model = build_backbone(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-2))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    use_amp = bool(cfg.get("amp", False)) and DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if target == "classify":
        pos = max(1, int((tr_df["anemia"] == 1).sum())); neg = max(1, int((tr_df["anemia"] == 0).sum()))
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=DEVICE))
    else:
        crit = nn.MSELoss()
    for ep in range(cfg["epochs"]):
        model.train(); t0 = time.time()
        for x, y in tr:
            x = x.to(DEVICE); y = y.to(DEVICE)
            yt = (y - hgb_mean) / hgb_std if target == "regress" else y
            opt.zero_grad()
            with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
                out = model(x).squeeze(1)
                loss = crit(out, yt)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        if (ep + 1) % max(1, cfg["epochs"] // 5) == 0 or ep == cfg["epochs"] - 1:
            print(f"    ep{ep+1}/{cfg['epochs']} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    # predict test
    model.eval(); preds = []
    va = DataLoader(ROIDS(te_df, False, cfg), batch_size=bs, shuffle=False, num_workers=nw)
    with torch.no_grad():
        for x, _ in va:
            o = model(x.to(DEVICE)).squeeze(1).float().cpu().numpy()
            preds.append(o)
    out = np.concatenate(preds)
    if target == "regress":
        return out * hgb_std + hgb_mean   # predicted Hgb
    return 1 / (1 + np.exp(-out))         # proba


def _minmax(s):
    s = np.asarray(s, float); return (s - s.min()) / (s.max() - s.min() + 1e-9)


def run(cfg):
    man = pd.read_csv(MAN)
    df = man[man["anemia"].notna()].copy()
    df["anemia"] = df["anemia"].astype(int)
    df["sex_M"] = df["sex"].astype(str).str.upper().str.startswith("M").astype(int)
    target = cfg.get("target", "classify")
    sub = cfg.get("subset")
    if sub:
        df = pd.concat([g.head(max(2, sub // 2)) for _, g in df.groupby("cohort")]).reset_index(drop=True)
    hgb_mean, hgb_std = float(df["hgb"].mean()), float(df["hgb"].std())

    results = {"config": cfg}
    # ---- pooled CV ----
    if "cv" in cfg.get("modes", ["cv", "loco"]):
        y = df["anemia"].values; sexM = df["sex_M"].values; hgb = df["hgb"].values
        strat = (df["cohort"] + "_" + df["anemia"].astype(str)).values
        oof = np.zeros(len(df))
        skf = StratifiedKFold(cfg.get("folds", 5), shuffle=True, random_state=cfg.get("seed", 42))
        for k, (tri, tei) in enumerate(skf.split(df, strat)):
            print(f"  CV fold {k}", flush=True)
            pred = train_one(df.iloc[tri], df.iloc[tei], cfg, hgb_mean, hgb_std)
            oof[tei] = pred
        if target == "regress":
            cut = np.where(sexM == 1, 13.0, 12.0)
            rep = report(y, _minmax(cut - oof), df["cohort"].values)
            rep["acc_hard"] = float((who_label(oof, sexM) == y).mean())
            rep["hgb_mae"] = float(np.abs(oof - hgb).mean())
        else:
            rep = report(y, oof, df["cohort"].values)
        results["cv"] = rep
        print("CV overall:", {k: round(v, 3) for k, v in rep["overall"].items() if k in ("auc", "sens", "spec", "acc")}, flush=True)
    # ---- LOCO ----
    if "loco" in cfg.get("modes", ["cv", "loco"]):
        loco = {}
        for tr_c, te_c in [("India", "Italy"), ("Italy", "India")]:
            print(f"  LOCO {tr_c}->{te_c}", flush=True)
            tr_df = df[df.cohort == tr_c]; te_df = df[df.cohort == te_c]
            pred = train_one(tr_df, te_df, cfg, float(tr_df.hgb.mean()), float(tr_df.hgb.std()))
            yte = te_df["anemia"].values; sexte = te_df["sex_M"].values; hgte = te_df["hgb"].values
            if target == "regress":
                cut = np.where(sexte == 1, 13.0, 12.0)
                m = metrics(yte, _minmax(cut - pred))
                m["acc_hard"] = float((who_label(pred, sexte) == yte).mean())
                m["hgb_mae"] = float(np.abs(pred - hgte).mean())
            else:
                m = metrics(yte, pred)
            loco[f"train_{tr_c}_test_{te_c}"] = m
            print("   ", {k: round(v, 3) for k, v in m.items() if k in ("auc", "sens", "spec", "acc")}, flush=True)
        results["loco"] = loco
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.subset is not None: cfg["subset"] = args.subset
    if args.epochs is not None: cfg["epochs"] = args.epochs
    print(f"device={DEVICE} cnn={cfg['backbone']} target={cfg.get('target')} res={cfg['res']} "
          f"bs={cfg['bs']} epochs={cfg['epochs']} amp={cfg.get('amp')}", flush=True)
    res = run(cfg)
    os.makedirs(OUT_ROOT, exist_ok=True)
    outp = os.path.join(OUT_ROOT, f"anemia_cnn_{cfg['name']}.json")
    with open(outp, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("WROTE", outp, flush=True)


if __name__ == "__main__":
    main()
