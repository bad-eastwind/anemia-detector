"""End-to-end anemia inference for a single fresh image + patient sex.

Chains the two persisted, Kaggle-trained artifacts:
  1. SegFormer checkpoint (seg `final` model, trained on all 218) -> conjunctiva mask, where
     ROI = palpebral (1) U forniceal (2).
  2. GBM regressor bundle (anemia `final`) -> predicted Hgb -> WHO sex threshold -> label.

Preprocessing reuses the SAME code as training (no duplication):
  - seg.build_aug / seg.build_model      (EXIF image -> normalized tensor; arch from ckpt cfg)
  - data_io.load_image_rgb               (EXIF transpose to portrait)
  - anemia_features.features_from_arrays (identical pallor/color feature math)
Features are computed at the 512px cache resolution the GBM was trained on, independent of the
seg model's input res. Device-agnostic (CUDA > MPS > CPU) via seg.DEVICE; override with --device.

Usage:
  python src/infer_pipeline.py --image /path/to/eye.jpg --sex M
  python src/infer_pipeline.py --image eye.jpg --sex F --age 34 \
      --seg-checkpoint outputs/seg/kaggle_segformer_b3/seg_kaggle_final.pt \
      --clf outputs/clf/anemia_gbm_final.joblib --save-mask out_mask.png
"""
import os, sys, json, argparse
import numpy as np
import cv2
import torch
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import seg                                   # build_model, build_aug, DEVICE, MPS view-patch
import anemia_features as af                 # features_from_arrays (shared feature core)
from data_io import load_image_rgb          # EXIF transpose -> portrait RGB
from precompute_cache import SIZE as FEAT_SIZE   # 512: res the GBM features were trained at
from paths import SEG_OUT, CLF_OUT

DEFAULT_SEG = os.path.join(SEG_OUT, "kaggle_segformer_b3", "seg_kaggle_final.pt")
DEFAULT_CLF = os.path.join(CLF_OUT, "anemia_gbm_final.joblib")


def load_seg(ckpt_path, device):
    """Rebuild the seg model from the checkpoint's own cfg and load weights (no net needed)."""
    # weights_only=True: checkpoint holds only tensors + a primitive cfg dict, so refuse arbitrary
    # unpickling. Trusted source anyway (your own Kaggle run), but no reason to allow code exec.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = dict(ckpt["cfg"])
    cfg["encoder_weights"] = None            # weights come from state_dict; skip imagenet fetch
    model = seg.build_model(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["cfg"]


@torch.no_grad()
def predict_mask(model, cfg, rgb, device):
    """Run seg on EXIF-corrected RGB -> HxW class map {0,1,2} at the model's input res."""
    tf = seg.build_aug(cfg, train=False)     # Resize(res) + Normalize, identical to SegDS(val)
    x = tf(image=np.ascontiguousarray(rgb))["image"]
    x = torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    logits = model(x)
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


def run(image_path, sex, age=None, seg_ckpt=DEFAULT_SEG, clf_path=DEFAULT_CLF,
        device=None, save_mask=None):
    device = device or seg.DEVICE
    sex_M = int(str(sex).strip().upper().startswith("M"))

    # joblib.load unpickles the sklearn estimator -> arbitrary code on load. Required (no JSON
    # form for a fitted GBM); load ONLY your own anemia_gbm_final.joblib, not third-party files.
    bundle = joblib.load(clf_path)
    feat_cols = bundle["feat_cols"]
    cut = bundle["who_cutoffs"]["M"] if sex_M else bundle["who_cutoffs"]["F"]

    # 1) segment -> ROI (palpebral U forniceal)
    model, seg_cfg = load_seg(seg_ckpt, device)
    rgb = load_image_rgb(image_path)                         # EXIF-corrected portrait RGB
    mask = predict_mask(model, seg_cfg, rgb, device)         # at seg res
    roi = np.isin(mask, (1, 2))

    # 2) features at the cache res the GBM expects (resize image + ROI to 512, match precompute)
    img512 = cv2.resize(rgb, (FEAT_SIZE, FEAT_SIZE), interpolation=cv2.INTER_AREA)
    roi512 = cv2.resize(roi.astype(np.uint8), (FEAT_SIZE, FEAT_SIZE),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
    bgr512 = np.ascontiguousarray(img512[:, :, ::-1])        # features_from_arrays expects BGR
    feats = af.features_from_arrays(bgr512, roi512)
    if feats is None:
        raise RuntimeError(f"predicted conjunctiva ROI too small ({int(roi512.sum())} px) "
                           "-- segmentation likely failed on this image")
    feats["age"] = bundle["age_default"] if age is None else float(age)
    feats["sex_M"] = sex_M

    # 3) predicted Hgb -> WHO sex threshold
    x = np.array([[feats[c] for c in feat_cols]], dtype=float)
    hgb = float(bundle["estimator"].predict(x)[0])
    anemic = bool(hgb < cut)

    if save_mask:
        cv2.imwrite(save_mask, (mask.astype(np.uint8) * 127))   # 0/127/254 for bg/pal/forn

    return {
        "image": image_path,
        "sex": "M" if sex_M else "F",
        "age": feats["age"],
        "age_imputed": age is None,
        "pred_hgb": round(hgb, 2),
        "who_cutoff": cut,
        "anemic": anemic,
        "label": "anemic" if anemic else "non-anemic",
        "roi_frac": round(feats["roi_frac"], 4),
        "device": str(device),
    }


def main():
    ap = argparse.ArgumentParser(description="single-image anemia inference (seg -> ROI -> Hgb -> WHO)")
    ap.add_argument("--image", required=True, help="fresh conjunctiva JPG")
    ap.add_argument("--sex", required=True, help="patient sex (M/F) -- sets WHO cutoff + feature")
    ap.add_argument("--age", type=float, default=None, help="optional; imputed (train median) if omitted")
    ap.add_argument("--seg-checkpoint", default=DEFAULT_SEG, help=f"default {DEFAULT_SEG}")
    ap.add_argument("--clf", default=DEFAULT_CLF, help=f"default {DEFAULT_CLF}")
    ap.add_argument("--device", default=None, help="override auto device (cuda|mps|cpu)")
    ap.add_argument("--save-mask", default=None, help="optional path to dump predicted class mask PNG")
    args = ap.parse_args()
    dev = torch.device(args.device) if args.device else None
    out = run(args.image, args.sex, args.age, args.seg_checkpoint, args.clf, dev, args.save_mask)
    print(json.dumps(out, indent=2))
    print(f"\n{out['label'].upper()}  (pred Hgb {out['pred_hgb']} g/dL, WHO cutoff {out['who_cutoff']} for sex {out['sex']})")


if __name__ == "__main__":
    main()
