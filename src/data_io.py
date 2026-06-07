"""Robust IO for the Eyes-defy-anemia dataset.

Critical facts (see instruction.md):
- Original JPGs are stored landscape (3984x2988) with EXIF orientation 6 -> must be
  exif_transpose'd to portrait (2988x3984) to match the masks' aspect.
- Mask PNGs are full-frame, downscaled to 800x1067 (one native 2988x3984). They are the
  same photo with non-class pixels blanked to background. They align to the JPG by a plain
  resize -- NO registration.
- Mask PNGs have a corrupt iCCP chunk: PIL refuses them, OpenCV reads them (with a warning).
- Foreground encoding is inconsistent:
    * India masks: foreground in the ALPHA channel (binary 0/255, or 16-bit 0/65535).
    * Italy masks: alpha is fully opaque -> foreground is the non-background RGB
      (background blanked to near-white or near-black).
  -> pick alpha when it actually carries signal, else fall back to RGB-not-background.
"""
import os
import glob
import numpy as np
import cv2
from PIL import Image, ImageOps

# class ids for multi-class segmentation
BG, PALPEBRAL, FORNICEAL = 0, 1, 2
CLASS_NAMES = {BG: "background", PALPEBRAL: "palpebral", FORNICEAL: "forniceal"}

MASK_SUFFIXES = {
    "_forniceal_palpebral.png": "union",
    "_forniceal.png": "forniceal",
    "_palpebral.png": "palpebral",
}


def classify_png(fname):
    """Return 'union'|'forniceal'|'palpebral' by exact suffix, else None.

    Order matters: '_forniceal_palpebral.png' must be tested before '_palpebral.png'.
    """
    for suf in ("_forniceal_palpebral.png", "_forniceal.png", "_palpebral.png"):
        if fname.endswith(suf):
            return MASK_SUFFIXES[suf]
    return None


def find_patient_files(folder):
    """Return dict: {'image': jpg_path, 'palpebral':..., 'forniceal':..., 'union':...}.

    Missing masks are simply absent from the dict.
    """
    out = {}
    for f in os.listdir(folder):
        fp = os.path.join(folder, f)
        if f.lower().endswith(".jpg"):
            out["image"] = fp
        elif f.lower().endswith(".png"):
            c = classify_png(f)
            if c:
                out[c] = fp
    return out


def load_image_rgb(path):
    """Load JPG, apply EXIF orientation, return HxWx3 uint8 RGB (portrait)."""
    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert("RGB")
    return np.asarray(im)


def _to_uint8(arr):
    if arr.dtype == np.uint16:
        return (arr >> 8).astype(np.uint8)
    return arr.astype(np.uint8)


def load_mask_fg(path, bg_white_thr=240, bg_black_thr=15):
    """Load a class mask PNG and return a boolean foreground map (HxW).

    Strategy:
      - Read with OpenCV IMREAD_UNCHANGED (handles the corrupt iCCP chunk).
      - If an alpha channel exists AND carries real signal (a non-trivial fraction of
        pixels is transparent), foreground = alpha high.
      - Otherwise foreground = RGB pixel is neither near-white nor near-black.
    """
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise IOError(f"cannot read mask: {path}")
    if raw.ndim == 2:  # grayscale
        g = _to_uint8(raw)
        return (g > bg_black_thr) & (g < bg_white_thr)

    raw = _to_uint8(raw)
    has_alpha = raw.shape[2] == 4
    bgr = raw[:, :, :3]
    rgb = bgr[:, :, ::-1]

    if has_alpha:
        a = raw[:, :, 3]
        # alpha carries signal only if a meaningful share of pixels is transparent
        if (a < 128).mean() > 0.02:
            return a >= 128

    # RGB fallback: background blanked to near-white or near-black
    near_white = np.all(rgb >= bg_white_thr, axis=2)
    near_black = np.all(rgb <= bg_black_thr, axis=2)
    return ~(near_white | near_black)


def load_label_mask(files, target_hw=None):
    """Build a multi-class label map {0:bg, 1:palpebral, 2:forniceal} for one patient.

    `files` is the dict from find_patient_files. Uses palpebral + forniceal class masks.
    If only the union exists, the whole union is labelled palpebral (best available).
    Where palpebral and forniceal overlap, forniceal wins (it is the deeper region and the
    union mask shows the two are near-disjoint with a thin shared border).

    Returns HxW uint8. If target_hw given, nearest-resizes to it.
    """
    ref = files.get("palpebral") or files.get("forniceal") or files.get("union")
    if ref is None:
        raise ValueError("no mask available for patient")
    fg_ref = load_mask_fg(ref)
    H, W = fg_ref.shape
    lbl = np.zeros((H, W), np.uint8)

    def _fit(m):
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        return m

    forn = _fit(load_mask_fg(files["forniceal"])) if "forniceal" in files else None

    if "palpebral" in files:
        lbl[_fit(load_mask_fg(files["palpebral"]))] = PALPEBRAL
    elif "union" in files:
        # palpebral file missing: recover it as (union - forniceal) (e.g. India/7)
        u = _fit(load_mask_fg(files["union"]))
        pal = u & ~forn if forn is not None else u
        lbl[pal] = PALPEBRAL

    if forn is not None:
        lbl[forn] = FORNICEAL  # forniceal wins on any overlap

    if target_hw is not None and (H, W) != tuple(target_hw):
        lbl = cv2.resize(lbl, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return lbl


def load_union_fg(files, target_hw=None):
    """Boolean conjunctiva ROI (palpebral ∪ forniceal). Prefer the stored union mask."""
    if "union" in files:
        fg = load_mask_fg(files["union"])
    else:
        fg = None
        for k in ("palpebral", "forniceal"):
            if k in files:
                m = load_mask_fg(files[k])
                fg = m if fg is None else (fg | m)
    if target_hw is not None and fg.shape != tuple(target_hw):
        fg = cv2.resize(fg.astype(np.uint8), (target_hw[1], target_hw[0]),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
    return fg
