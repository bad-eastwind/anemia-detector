"""Batch the CP-AnemiC unseen-data probe over ALL images, both routes, with metrics + plots.

ROUTE 1 (seg+clf): our SegFormer predicts the ROI.   ROUTE 2 (clf-only): use the PNG alpha mask.
Same GBM classifier on both. For every image we record predicted Hgb + our (adult-WHO) label,
then aggregate:
  - per-route MAE / RMSE / bias / Pearson r vs ground-truth HB_LEVEL
  - label agreement vs REMARK (adult-WHO label, and a pediatric Hgb<11 label as a fairer control)
  - scatter: actual vs predicted Hgb (combined overlay + per-route panels)

Outputs -> testunseendatav1/:  batch_710_results.csv, batch_710_metrics.json,
                               actual_vs_pred_combined.png, actual_vs_pred_perroute.png

Usage:  python src/batch_cpanemic.py            (MPS if available, else CPU)
        python src/batch_cpanemic.py --device cpu --limit 50
"""
import os, sys, json, argparse, time
import numpy as np, pandas as pd, cv2, torch, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import infer_pipeline as ip
from infer_cpanemic import classify, DS, XLSX, IMG_DIR, OUT_DIR
from data_io import load_image_rgb

PED_CUT = 11.0   # rough pediatric Hgb cutoff (their REMARK basis), for a fairer agreement control


def remark_is_anemic(r):
    return 0 if "non" in str(r).strip().lower() else 1


def agree(pred_anemic, gt_anemic):
    p, g = np.asarray(pred_anemic), np.asarray(gt_anemic)
    tp = int(((p == 1) & (g == 1)).sum()); tn = int(((p == 0) & (g == 0)).sum())
    fp = int(((p == 1) & (g == 0)).sum()); fn = int(((p == 0) & (g == 1)).sum())
    n = len(g)
    return {"acc": round((tp+tn)/n, 3), "sens": round(tp/(tp+fn), 3) if tp+fn else None,
            "spec": round(tn/(tn+fp), 3) if tn+fp else None, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def err_stats(pred, gt):
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    d = pred - gt
    r = float(np.corrcoef(pred, gt)[0, 1]) if len(pred) > 2 and pred.std() > 0 else float("nan")
    return {"n": len(pred), "mae": round(float(np.abs(d).mean()), 3),
            "rmse": round(float(np.sqrt((d**2).mean())), 3),
            "bias_pred_minus_gt": round(float(d.mean()), 3), "pearson_r": round(r, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None, help="debug: first N images")
    args = ap.parse_args()
    dev = torch.device(args.device)
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_excel(XLSX).set_index("IMAGE_ID")
    ids = sorted(f[:-4] for f in os.listdir(IMG_DIR) if f.endswith(".png"))
    ids = [i for i in ids if i in df.index]
    if args.limit:
        ids = ids[:args.limit]
    print(f"device={dev}  images={len(ids)}  (load seg+clf once, loop)")

    model, cfg = ip.load_seg(ip.DEFAULT_SEG, dev)
    bundle = joblib.load(ip.DEFAULT_CLF)   # trusted local artifact (your Kaggle run)

    rows, fails = [], []
    t0 = time.time()
    for k, iid in enumerate(ids):
        try:
            meta = df.loc[iid]
            sex = str(meta["GENDER"]); sex_M = int(sex.strip().upper().startswith("M"))
            age_y = round(float(meta["Age(Months)"]) / 12.0, 3)
            gt_hgb = float(meta["HB_LEVEL"]); gt_an = remark_is_anemic(meta["REMARK"])
            path = os.path.join(IMG_DIR, iid + ".png")

            # route 1: our seg -> ROI
            rgb = load_image_rgb(path)
            mask = ip.predict_mask(model, cfg, rgb, dev)
            roi1 = np.isin(mask, (1, 2))
            r1 = classify(np.ascontiguousarray(rgb[:, :, ::-1]), roi1, sex_M, age_y, bundle)

            # route 2: provided alpha mask -> ROI
            bgra = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            bgr = bgra[:, :, :3]
            roi2 = bgra[:, :, 3] >= 128 if (bgra.ndim == 3 and bgra.shape[2] == 4) else np.ones(bgra.shape[:2], bool)
            r2 = classify(bgr, roi2, sex_M, age_y, bundle)
            if r1 is None or r2 is None:
                fails.append((iid, "empty ROI")); continue

            rows.append({"image_id": iid, "sex": sex, "age_years": age_y, "gt_hgb": gt_hgb,
                         "gt_remark": str(meta["REMARK"]), "gt_anemic": gt_an,
                         "r1_hgb": r1["pred_hgb"], "r1_anemic": int(r1["anemic"]), "r1_roi": r1["roi_frac"],
                         "r2_hgb": r2["pred_hgb"], "r2_anemic": int(r2["anemic"]), "r2_roi": r2["roi_frac"]})
        except Exception as e:
            fails.append((iid, f"{type(e).__name__}: {str(e)[:80]}"))
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(ids)}  ({time.time()-t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    csv = os.path.join(OUT_DIR, "batch_710_results.csv")
    res.to_csv(csv, index=False)
    print(f"\n{len(res)} ok, {len(fails)} failed -> {csv}")

    gt = res["gt_hgb"].values
    metrics = {"n": len(res), "n_failed": len(fails),
               "route1_seg_clf": err_stats(res["r1_hgb"], gt),
               "route2_clf_only": err_stats(res["r2_hgb"], gt),
               "label_agreement_vs_REMARK": {
                   "route1_adultWHO": agree(res["r1_anemic"], res["gt_anemic"]),
                   "route2_adultWHO": agree(res["r2_anemic"], res["gt_anemic"]),
                   "route1_pediatric_lt11": agree((res["r1_hgb"] < PED_CUT).astype(int), res["gt_anemic"]),
                   "route2_pediatric_lt11": agree((res["r2_hgb"] < PED_CUT).astype(int), res["gt_anemic"])},
               "gt_anemic_prevalence": round(float(res["gt_anemic"].mean()), 3)}
    with open(os.path.join(OUT_DIR, "batch_710_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Hgb error vs GT ===")
    for r in ("route1_seg_clf", "route2_clf_only"):
        m = metrics[r]; print(f"  {r:16s} MAE {m['mae']}  RMSE {m['rmse']}  bias {m['bias_pred_minus_gt']:+}  r {m['pearson_r']}")
    print("=== label agreement vs REMARK ===")
    for k_ in ("route1_adultWHO", "route2_adultWHO", "route1_pediatric_lt11", "route2_pediatric_lt11"):
        a = metrics["label_agreement_vs_REMARK"][k_]; print(f"  {k_:24s} acc {a['acc']}  sens {a['sens']}  spec {a['spec']}")
    print(f"  (GT anemic prevalence {metrics['gt_anemic_prevalence']})")

    # ---- PLOTS ----
    lo = float(min(gt.min(), res["r1_hgb"].min(), res["r2_hgb"].min())) - 0.5
    hi = float(max(gt.max(), res["r1_hgb"].max(), res["r2_hgb"].max())) + 0.5

    # combined overlay
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x (perfect)")
    ax.scatter(gt, res["r1_hgb"], s=14, alpha=0.5, c="#d62728",
               label=f"R1 seg+clf  (MAE {metrics['route1_seg_clf']['mae']}, r {metrics['route1_seg_clf']['pearson_r']})")
    ax.scatter(gt, res["r2_hgb"], s=14, alpha=0.5, c="#1f77b4",
               label=f"R2 clf-only (MAE {metrics['route2_clf_only']['mae']}, r {metrics['route2_clf_only']['pearson_r']})")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("Actual Hgb (g/dL)"); ax.set_ylabel("Predicted Hgb (g/dL)")
    ax.set_title(f"CP-AnemiC unseen data (n={len(res)}): actual vs predicted Hgb")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    p1 = os.path.join(OUT_DIR, "actual_vs_pred_combined.png"); fig.tight_layout(); fig.savefig(p1, dpi=130); plt.close(fig)

    # per-route panels, colored by GT remark
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, (col, name) in zip(axes, [("r1_hgb", "R1 seg+clf"), ("r2_hgb", "R2 clf-only")]):
        for an, c, lab in [(1, "#d62728", "GT anemic"), (0, "#2ca02c", "GT non-anemic")]:
            s = res[res["gt_anemic"] == an]
            ax.scatter(s["gt_hgb"], s[col], s=16, alpha=0.55, c=c, label=lab)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        key = "route1_seg_clf" if col == "r1_hgb" else "route2_clf_only"
        m = metrics[key]
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel("Actual Hgb (g/dL)"); ax.set_ylabel("Predicted Hgb (g/dL)")
        ax.set_title(f"{name}\nMAE {m['mae']}  RMSE {m['rmse']}  bias {m['bias_pred_minus_gt']:+}  r {m['pearson_r']}")
        ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    p2 = os.path.join(OUT_DIR, "actual_vs_pred_perroute.png"); fig.tight_layout(); fig.savefig(p2, dpi=130); plt.close(fig)
    print(f"\nplots -> {p1}\n        {p2}")
    if fails[:5]:
        print("sample failures:", fails[:5])


if __name__ == "__main__":
    main()
