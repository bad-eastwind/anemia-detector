"""Unseen-data probe on the CP-AnemiC dataset (pediatric, Ghana; pre-segmented conjunctiva crops).

Picks one image (random or --image-id), pulls GENDER + Age(Months) + HB_LEVEL + REMARK from
Anemia_Data_Collection_Sheet.xlsx, then runs anemia inference TWO ways and contrasts them:
  ROUTE 1  seg + classifier : our SegFormer predicts the ROI (ignores the provided mask).
  ROUTE 2  classifier only  : use the PNG alpha channel as the ROI (the dataset's own seg).
Same GBM classifier on both. Writes a side-by-side PNG + a JSON row to testunseendatav1/.

CAVEATS (expected distribution shift, reported not hidden):
  - age is months -> years (infants ~0.5-4 yr); our model trained on adults.
  - WHO cutoff here is ADULT (M<13, F<12); CP-AnemiC REMARK uses pediatric (~11). Our label
    can disagree with REMARK purely from the cutoff/population mismatch.

Usage:
  python src/infer_cpanemic.py                 # random image
  python src/infer_cpanemic.py --image-id Image_026 --device cpu
"""
import os, sys, json, argparse, random
import numpy as np, pandas as pd, cv2, torch, joblib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import infer_pipeline as ip
import anemia_features as af
from data_io import load_image_rgb

REPO = os.path.dirname(HERE)
DS = os.path.join(REPO, "CP-AnemiC dataset (extract")
XLSX = os.path.join(DS, "Anemia_Data_Collection_Sheet.xlsx")
IMG_DIR = os.path.join(DS, "Images")
OUT_DIR = os.path.join(REPO, "testunseendatav1")
S = ip.FEAT_SIZE   # 512


def classify(bgr, roi, sex_M, age, bundle):
    """Classifier tail: features over `roi` -> GBM Hgb -> WHO sex cutoff. Returns dict or None."""
    img512 = cv2.resize(bgr, (S, S), interpolation=cv2.INTER_AREA)
    roi512 = cv2.resize(roi.astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST).astype(bool)
    feats = af.features_from_arrays(np.ascontiguousarray(img512), roi512)
    if feats is None:
        return None
    feats["age"] = age
    feats["sex_M"] = sex_M
    x = np.array([[feats[c] for c in bundle["feat_cols"]]], float)
    hgb = float(bundle["estimator"].predict(x)[0])
    cut = bundle["who_cutoffs"]["M"] if sex_M else bundle["who_cutoffs"]["F"]
    return {"pred_hgb": round(hgb, 2), "who_cutoff": cut, "anemic": bool(hgb < cut),
            "label": "anemic" if hgb < cut else "non-anemic", "roi_frac": round(float(roi512.mean()), 4)}


def fit_box(img, box=260, bg=255):
    """Aspect-preserving resize into a square canvas (letterbox) for the figure."""
    h, w = img.shape[:2]
    s = box / max(h, w)
    r = cv2.resize(img, (max(1, int(w*s)), max(1, int(h*s))), interpolation=cv2.INTER_AREA)
    canvas = np.full((box, box, 3), bg, np.uint8)
    y, x = (box - r.shape[0]) // 2, (box - r.shape[1]) // 2
    canvas[y:y+r.shape[0], x:x+r.shape[1]] = r
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-id", default=None, help="e.g. Image_026; default = random")
    ap.add_argument("--device", default="cpu", help="cuda|mps|cpu (cpu safest for MiT on mac)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    dev = torch.device(args.device)
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_excel(XLSX).set_index("IMAGE_ID")
    iid = args.image_id or random.choice([f[:-4] for f in os.listdir(IMG_DIR) if f.endswith(".png")])
    row = df.loc[iid]
    path = os.path.join(IMG_DIR, iid + ".png")
    sex = str(row["GENDER"]); sex_M = int(sex.strip().upper().startswith("M"))
    age_y = round(float(row["Age(Months)"]) / 12.0, 2)
    gt_hgb = float(row["HB_LEVEL"]); gt_remark = str(row["REMARK"])
    print(f"== {iid}  {sex} age {row['Age(Months)']}mo -> {age_y}yr  |  GT Hgb {gt_hgb}  REMARK {gt_remark}")

    bundle = joblib.load(ip.DEFAULT_CLF)   # trusted local artifact (your Kaggle run)

    # ---- ROUTE 1: seg + classifier (our SegFormer finds the ROI) ----
    model, cfg = ip.load_seg(ip.DEFAULT_SEG, dev)
    rgb = load_image_rgb(path)
    smask = ip.predict_mask(model, cfg, rgb, dev)                 # 512 {0,1,2}
    seg_dist = {int(v): round(c/smask.size*100, 1) for v, c in zip(*np.unique(smask, return_counts=True))}
    r1 = ip.run(path, sex, age=age_y, seg_ckpt=ip.DEFAULT_SEG, clf_path=ip.DEFAULT_CLF, device=dev)

    # ---- ROUTE 2: classifier only (use the dataset's alpha mask as ROI) ----
    bgra = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    bgr = bgra[:, :, :3]
    prov_roi = bgra[:, :, 3] >= 128 if bgra.shape[2] == 4 else np.ones(bgra.shape[:2], bool)
    r2 = classify(bgr, prov_roi, sex_M, age_y, bundle)

    print(f"   ROUTE1 seg+clf : Hgb {r1['pred_hgb']}  {r1['label'].upper():12s} ROI {r1['roi_frac']*100:.1f}%  segpx{seg_dist}")
    print(f"   ROUTE2 clf-only: Hgb {r2['pred_hgb']}  {r2['label'].upper():12s} ROI {r2['roi_frac']*100:.1f}% (provided alpha mask)")
    print(f"   vs GT Hgb {gt_hgb}:  R1 err {abs(r1['pred_hgb']-gt_hgb):.2f}   R2 err {abs(r2['pred_hgb']-gt_hgb):.2f}")

    # ---- figure: original | provided mask | our-seg mask, with both verdicts ----
    comp = np.full(bgr.shape[:2] + (3,), 255, np.uint8)             # composite crop on white
    a = (bgra[:, :, 3:4].astype(float)/255.0) if bgra.shape[2] == 4 else 1.0
    comp = (bgr * a + 255 * (1 - a)).astype(np.uint8)
    prov_vis = np.zeros_like(bgr); prov_vis[prov_roi] = (0, 200, 0)
    seg_full = cv2.resize(smask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    seg_vis = np.zeros_like(bgr); seg_vis[seg_full == 1] = (0, 255, 0); seg_vis[seg_full == 2] = (0, 0, 255)
    panels = [fit_box(comp), fit_box(prov_vis), fit_box(seg_vis)]
    for p, t in zip(panels, ["crop (alpha->white)", "ROUTE2 provided mask", "ROUTE1 our seg"]):
        cv2.rectangle(p, (0, 0), (len(t)*9+8, 22), (0, 0, 0), -1)
        cv2.putText(p, t, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    grid = np.concatenate(panels, axis=1)
    banner = np.full((96, grid.shape[1], 3), 255, np.uint8)
    L = [f"{iid}  {sex}  {age_y}yr  |  GT Hgb {gt_hgb} ({gt_remark})  [adult WHO cutoff {r1['who_cutoff']}]",
         f"R1 seg+clf : Hgb {r1['pred_hgb']} -> {r1['label'].upper()}",
         f"R2 clf-only: Hgb {r2['pred_hgb']} -> {r2['label'].upper()}"]
    cv2.putText(banner, L[0], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(banner, L[1], (8, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,0,200) if r1["anemic"] else (0,140,0), 1, cv2.LINE_AA)
    cv2.putText(banner, L[2], (8, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,0,200) if r2["anemic"] else (0,140,0), 1, cv2.LINE_AA)
    fig = np.concatenate([banner, grid], axis=0)
    png = os.path.join(OUT_DIR, f"{iid}_compare.png")
    cv2.imwrite(png, fig)

    rec = {"image_id": iid, "sex": sex, "age_months": int(row["Age(Months)"]), "age_years": age_y,
           "gt_hgb": gt_hgb, "gt_remark": gt_remark, "severity": str(row.get("Severity")),
           "route1_seg_clf": {**r1, "seg_class_pct": seg_dist},
           "route2_clf_only_providedmask": r2,
           "hgb_delta_r1_minus_r2": round(r1["pred_hgb"] - r2["pred_hgb"], 2),
           "abs_err_vs_gt": {"route1": round(abs(r1["pred_hgb"]-gt_hgb), 2),
                             "route2": round(abs(r2["pred_hgb"]-gt_hgb), 2)},
           "device": args.device}
    with open(os.path.join(OUT_DIR, f"{iid}_result.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print(f"   SAVED  {png}\n          {os.path.join(OUT_DIR, iid+'_result.json')}")


if __name__ == "__main__":
    main()
