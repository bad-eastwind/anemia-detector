"""Extract pallor / color / texture features from the conjunctiva ROI for anemia prediction.

ROI = union mask (palpebral ∪ forniceal) applied to the EXIF-corrected image. Anemic
conjunctiva is pale -> low saturation, lower redness (a*), higher lightness. We compute
robust per-channel statistics across several color spaces plus pallor-specific indices.

Reads the 512px cache (already EXIF-corrected, area-resized) for speed and color fidelity.
Output: outputs/anemia_features.csv  (one row per patient with a valid Hgb label).
"""
import os
import numpy as np
import cv2
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "outputs", "cache")
MAN = os.path.join(HERE, "..", "outputs", "manifest.csv")
OUT = os.path.join(HERE, "..", "outputs", "anemia_features.csv")

QUANTILES = [5, 10, 25, 50, 75, 90, 95]


def chan_stats(prefix, vals, out):
    out[f"{prefix}_mean"] = float(vals.mean())
    out[f"{prefix}_std"] = float(vals.std())
    for q in QUANTILES:
        out[f"{prefix}_p{q}"] = float(np.percentile(vals, q))


def extract_one(uid):
    img = cv2.imread(os.path.join(CACHE, "img", uid + ".png"))  # BGR
    roi = cv2.imread(os.path.join(CACHE, "roi", uid + ".png"), 0) > 127
    if roi.sum() < 50:
        return None
    rgb = img[:, :, ::-1].astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    m = roi

    R, G, B = rgb[..., 0][m], rgb[..., 1][m], rgb[..., 2][m]
    H, S, V = hsv[..., 0][m], hsv[..., 1][m], hsv[..., 2][m]
    L, A, Bb = lab[..., 0][m], lab[..., 1][m], lab[..., 2][m]
    Y, Cr, Cb = ycc[..., 0][m], ycc[..., 1][m], ycc[..., 2][m]

    f = {"uid": uid, "roi_frac": float(m.mean())}
    for name, v in [("R", R), ("G", G), ("B", B), ("H", H), ("S", S), ("V", V),
                    ("L", L), ("a", A), ("bb", Bb), ("Y", Y), ("Cr", Cr), ("Cb", Cb)]:
        chan_stats(name, v, f)

    eps = 1e-6
    # pallor / erythema indices (per-pixel then averaged) -- anemia lowers redness
    rg = (R - G) / (R + G + eps)
    rb = (R - B) / (R + B + eps)
    gr = G / (R + eps)
    ei = 100.0 * (np.log10(1.0 / (G + eps)) - np.log10(1.0 / (R + eps)))   # erythema index
    redness = R / (R + G + B + eps)
    for name, v in [("ratio_rg", rg), ("ratio_rb", rb), ("ratio_gr", gr),
                    ("erythema", ei), ("redness", redness)]:
        chan_stats(name, v, f)
    # hue dispersion (anemic conjunctiva less saturated/colorful)
    f["S_lowfrac"] = float((S < 60).mean())
    f["a_minus128_mean"] = float((A - 128).mean())   # redness around neutral 128
    return f


def main():
    man = pd.read_csv(MAN)
    rows = []
    for _, r in man.iterrows():
        if pd.isna(r["anemia"]):   # skip patients without Hgb (Italy/93)
            continue
        feats = extract_one(r["uid"])
        if feats is None:
            continue
        feats.update({"cohort": r["cohort"], "anemia": int(r["anemia"]),
                      "hgb": float(r["hgb"]), "sex": r["sex"], "age": float(r["age"]),
                      "sex_M": int(str(r["sex"]).upper().startswith("M"))})
        rows.append(feats)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"features -> {OUT}  shape={df.shape}")
    print("n features:", len([c for c in df.columns if c not in
          ("uid", "cohort", "anemia", "hgb", "sex")]))
    print(df.groupby(["cohort", "anemia"]).size().to_string())


if __name__ == "__main__":
    main()
