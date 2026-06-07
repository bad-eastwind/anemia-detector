"""Precompute training cache: EXIF-corrected resized images + 3-class label masks + union ROI.

Decouples slow/fragile IO (EXIF transpose, corrupt-iCCP PNG decode) from training.
Outputs to outputs/cache/{img,lbl,roi}/<uid>.png at SIZE x SIZE.
  img: RGB uint8
  lbl: single-channel uint8 in {0,1,2}  (bg, palpebral, forniceal)
  roi: single-channel uint8 {0,255}      (conjunctiva union mask)
Also writes a full-resolution union ROI (portrait, image-sized) is NOT cached; the
classifier reads original JPG + union mask on demand for max color fidelity.
"""
import os
import sys
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_io import load_image_rgb, load_label_mask, load_union_fg, find_patient_files

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", "dataset anemia"))
CACHE = os.path.abspath(os.path.join(HERE, "..", "outputs", "cache"))
SIZE = 512


def main():
    man = pd.read_csv(os.path.join(HERE, "..", "outputs", "manifest.csv"))
    for sub in ("img", "lbl", "roi"):
        os.makedirs(os.path.join(CACHE, sub), exist_ok=True)
    counts = {0: 0, 1: 0, 2: 0}
    for _, r in man.iterrows():
        uid = r["uid"]
        folder = os.path.join(ROOT, r["cohort"], str(r["patient"]))
        fl = find_patient_files(folder)
        img = load_image_rgb(fl["image"])             # portrait HxWx3
        H, W = img.shape[:2]
        lbl = load_label_mask(fl, target_hw=(H, W))   # full-res label
        roi = load_union_fg(fl, target_hw=(H, W)).astype(np.uint8) * 255

        img_r = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        lbl_r = cv2.resize(lbl, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        roi_r = cv2.resize(roi, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(os.path.join(CACHE, "img", uid + ".png"), img_r[:, :, ::-1])
        cv2.imwrite(os.path.join(CACHE, "lbl", uid + ".png"), lbl_r)
        cv2.imwrite(os.path.join(CACHE, "roi", uid + ".png"), roi_r)
        for k in counts:
            counts[k] += int((lbl_r == k).sum())
    tot = sum(counts.values())
    print(f"cached {len(man)} patients at {SIZE}x{SIZE} -> {CACHE}")
    print("pixel class balance:", {k: round(v / tot, 4) for k, v in counts.items()})


if __name__ == "__main__":
    main()
