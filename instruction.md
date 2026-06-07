# Project: Conjunctiva Segmentation + Anemia Detection

## Goal
From an eye photo (everted lower eyelid), do two things with **peak accuracy**:
1. **Segment** the conjunctiva (palpebral + forniceal regions).
2. **Classify anemia** (anemic vs not) from the segmented conjunctiva.

Accuracy is the only priority. **On-device / model-size / latency do NOT matter.** Use the
strongest models available (large pretrained transformers are fine and encouraged).

## Scope — read this
- This project is a **pure segmentor + anemia classifier**. Nothing else.
- **Do NOT add color correction, white balance, flash/no-flash, RAW, or shade calibration.**
  A separate project (**NormaEngine**) handles all color rectification upstream. Assume
  inputs are (or will be) color-corrected elsewhere. Use the dataset images as given.

## Dataset — structure
Path: `dataset anemia/<cohort>/<patient>/` where cohort ∈ {India, Italy}. ~217 patients total.

Each patient folder has one original photo + up to three segmentation masks, e.g.:
```
dataset anemia/India/1/20200118_164733.jpg                      <- original eye photo
dataset anemia/India/1/20200118_164733_palpebral.png           <- PALPEBRAL conjunctiva mask
dataset anemia/India/1/20200118_164733_forniceal.png           <- FORNICEAL conjunctiva mask
dataset anemia/India/1/20200118_164733_forniceal_palpebral.png <- BOTH combined (union)
```
- `_palpebral.png`   = segmentation of the **palpebral** class.
- `_forniceal.png`   = segmentation of the **forniceal** class.
- `_forniceal_palpebral.png` = union of both (the full conjunctiva ROI; this is the pallor signal region).
- Some patients are missing some masks (e.g. only palpebral exists). Handle gracefully.
- **Filename gotcha:** `_forniceal_palpebral.png` also ends with `_palpebral.png`. Match by
  exact full suffix, not `endswith("_palpebral.png")`, or you mislabel classes.

## Dataset — THE CRITICAL FACT (do not get this wrong)
The mask PNGs are **the same image, downscaled, with non-class pixels blanked to background.**
They are **NOT tight crops** and need **NO registration / alignment / template matching.**

- Original JPG: `2988 x 3984` (aspect 0.75).
- Mask PNG: `800 x 1067` (aspect 0.75) — identical aspect ratio. (A few masks are stored at
  native `2988 x 3984`.)
- **Resize the JPG down to the mask size (or upscale the mask to the JPG size) and the mask
  overlaps the conjunctiva pixel-perfectly.** Scale factor = 3984/1067 = 2988/800 ≈ 3.73.
- Background in the mask = near-white OR near-black (varies); **foreground (the class region)
  = every pixel that is not background.** Build the binary/again multi-class mask from that.

This is the entire ground truth you need. Full-frame, pixel-aligned masks for all 217 already
exist for free. (A previous attempt wrongly assumed they were tight crops needing registration
and wasted effort on SAM/template-matching — ignore that path entirely.)

## Labels (for anemia)
Patient metadata (Hemoglobin Hgb g/dL, sex, age) lives in spreadsheet(s) inside `dataset anemia/`
(plus a `.docx` description). Parse them into a manifest. Anemia label = WHO Hgb threshold by
sex/age (Hgb below cutoff → anemic). If a ready anemia label is not in the metadata, derive it
from Hgb + WHO thresholds.

## Tasks
1. **Build a manifest** (CSV): per patient → cohort, image path, the 3 mask paths, Hgb, sex,
   age, anemia label.
2. **Segmentation model** — multi-class {background, palpebral, forniceal} (forniceal_palpebral
   = union, derivable). Maximize mean IoU / Dice.
3. **Anemia classifier** — using the segmented conjunctiva ROI (the forniceal_palpebral union):
   classify anemic vs not (or regress Hgb then apply WHO threshold). Use ONLY conjunctiva pixels
   as input, not the whole frame.

## Model recommendations (accuracy-first)
- **Segmentation:** try a transformer segmenter — **SegFormer** (HuggingFace `transformers`,
  MiT-b3/b4, pretrained) or **Mask2Former**; and a strong CNN baseline — **nnU-Net** (turnkey
  SOTA for small medical sets) or a U-Net with a heavy pretrained encoder (ConvNeXt/EfficientNet
  via `segmentation_models_pytorch`). Pick the winner by **leave-one-cohort-out** IoU, not pooled.
  Optionally fine-tune **SAM2**. Use heavy augmentation (geometric + mild photometric) — only
  217 images.
- **Anemia:** segment → mask the conjunctiva → either (a) a ViT/CNN classifier on the masked ROI,
  or (b) handcrafted pallor/color/texture features → gradient-boosting/logistic. Try both.

## Evaluation protocol (MANDATORY)
- Always report **pooled k-fold CV AND leave-one-cohort-out (LOCO)** — train India → test Italy,
  and train Italy → test India. Within-cohort numbers alone are misleading.
- Segmentation: mean IoU + Dice per class.
- Anemia: AUC, sensitivity, specificity, accuracy. Report **split by cohort and by anemic/non-anemic.**
- India and Italy differ in anemia prevalence and appearance — cross-site generalization is the
  real bar.

## Pitfalls to avoid
- **No registration/template-matching** — masks already align by resize (see CRITICAL FACT).
- **No color correction here** — NormaEngine's job.
- **Site confound:** whole-image models can learn "India vs Italy" instead of pallor. Always feed
  the segmented ROI and judge by LOCO.
- **Pallor bias:** anemic conjunctiva is pale (low saturation). Any redness/saturation-threshold
  segmentation rejects exactly the anemic cases. Use a **learned, shape/texture-based** segmenter,
  never color thresholds.

## Environment
Python 3.12 (venv). GPU or Apple MPS. Core libs: `torch`, `torchvision`, `transformers`
(SegFormer/Mask2Former) or `nnunetv2`, `albumentations`, `opencv-python`, `scikit-learn`,
`pandas`, `numpy`, `pillow`, `openpyxl`.
