"""Contrastive self-supervised pretraining (SimCLR / NT-Xent) for heterogeneity robustness.

Motivation: India and Italy differ in capture/appearance. A whole-image supervised model can
latch onto "which site" instead of pallor. SimCLR pretraining on the conjunctiva ROI from BOTH
cohorts (label-free, all 218 patients) teaches the encoder representations invariant to the two
augmented views -> robust to nuisance illumination/site variation. We then initialise the anemia
classifier from this encoder (anemia_cnn.py `init_weights`).

CRITICAL: pallor IS a colour signal, so we do NOT use SimCLR's standard grayscale / strong colour
jitter (that would make the encoder colour-invariant = blind to anemia). We use geometric + MILD
photometric (brightness/contrast, blur, small hue/sat) views only.

Saves outputs/ssl/<name>.pt = {"encoder": <timm backbone state_dict>, "cfg": cfg}.
Load it in anemia_cnn via `init_weights: outputs/ssl/<name>.pt` (same `backbone`).

Usage:
  python ssl_pretrain.py --config configs/ssl_smoke_local.yaml
  python ssl_pretrain.py --config configs/ssl_kaggle.yaml
"""
import os, sys, argparse, random, time
import numpy as np
import cv2
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import pandas as pd
import timm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import CACHE, MANIFEST, SSL_OUT
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
    out = img.copy(); out[~roi] = 0
    return out[y0:y1, x0:x1]


def view_aug(res):
    # geometric + MILD photometric only (preserve pallor). NO grayscale / strong colour jitter.
    return A.Compose([
        A.RandomResizedCrop(size=(res, res), scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), rotate=(-20, 20), translate_percent=(0, 0.06), p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=8, p=0.3),
        A.Normalize(MEAN, STD)])


class TwoViewDS(Dataset):
    def __init__(self, uids, cfg):
        self.uids = list(uids)
        self.use_roi = cfg.get("use_roi", True)
        self.tf = view_aug(cfg["res"])

    def __len__(self):
        return len(self.uids)

    def _img(self, uid):
        img = cv2.imread(os.path.join(CACHE, "img", uid + ".png"))[:, :, ::-1]
        if self.use_roi:
            roi = cv2.imread(os.path.join(CACHE, "roi", uid + ".png"), 0) > 127
            img = crop_roi(np.ascontiguousarray(img), roi)
        return np.ascontiguousarray(img)

    def __getitem__(self, i):
        base = self._img(self.uids[i])
        v1 = self.tf(image=base)["image"].transpose(2, 0, 1)
        v2 = self.tf(image=base)["image"].transpose(2, 0, 1)
        return torch.from_numpy(v1).float(), torch.from_numpy(v2).float()


class ProjHead(nn.Module):
    def __init__(self, dim_in, dim_out=128, hidden=2048):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_in, hidden), nn.BatchNorm1d(hidden),
                                 nn.ReLU(inplace=True), nn.Linear(hidden, dim_out))

    def forward(self, x):
        return self.net(x)


def nt_xent(z1, z2, temp):
    N = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], 0), dim=1)        # 2N x d
    sim = (z @ z.t()) / temp                              # 2N x 2N
    sim.fill_diagonal_(torch.finfo(sim.dtype).min)
    targets = torch.arange(N, device=z.device)
    targets = torch.cat([targets + N, targets], 0)        # positive partner index
    return F.cross_entropy(sim, targets)


def run(cfg):
    seed_all(cfg.get("seed", 42))
    man = pd.read_csv(MANIFEST)
    uids = man["uid"].values
    sub = cfg.get("subset")
    if sub:
        uids = uids[:sub]
    nw = cfg.get("num_workers", 2); bs = cfg["bs"]
    dl = DataLoader(TwoViewDS(uids, cfg), batch_size=bs, shuffle=True, num_workers=nw,
                    drop_last=len(uids) > bs, persistent_workers=nw > 0, pin_memory=DEVICE.type == "cuda")
    encoder = timm.create_model(cfg["backbone"], pretrained=cfg.get("pretrained", True),
                                num_classes=0).to(DEVICE)
    feat_dim = encoder.num_features
    head = ProjHead(feat_dim, cfg.get("proj_dim", 128)).to(DEVICE)
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    use_amp = bool(cfg.get("amp", False)) and DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    temp = cfg.get("temp", 0.2); accum = max(1, cfg.get("grad_accum", 1))

    encoder.train(); head.train()
    for ep in range(cfg["epochs"]):
        t0 = time.time(); tot = 0.0; opt.zero_grad()
        for i, (v1, v2) in enumerate(dl):
            v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
            with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
                z1 = head(encoder(v1)); z2 = head(encoder(v2))
                loss = nt_xent(z1, z2, temp) / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad()
            tot += loss.item() * accum
        sched.step()
        print(f"[ssl] ep{ep+1}/{cfg['epochs']} ntxent {tot/max(1,len(dl)):.4f} ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(SSL_OUT, exist_ok=True)
    out = os.path.join(SSL_OUT, f"{cfg['name']}.pt")
    torch.save({"encoder": {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
                "backbone": cfg["backbone"], "cfg": cfg}, out)
    print(f"WROTE {out}  (backbone={cfg['backbone']}, feat_dim={feat_dim})", flush=True)
    return out


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
    print(f"device={DEVICE} ssl backbone={cfg['backbone']} res={cfg['res']} bs={cfg['bs']} "
          f"epochs={cfg['epochs']} temp={cfg.get('temp',0.2)} amp={cfg.get('amp')}", flush=True)
    run(cfg)


if __name__ == "__main__":
    main()
