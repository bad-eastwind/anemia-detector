# Anemia Detector

Conjunctiva segmentation ({background, palpebral, forniceal}) + anemia classification from the
segmented ROI, on the **Eyes-defy-anemia** dataset (India + Italy, 218 patients). Pipeline:
**SegFormer → conjunctiva ROI → pallor/color features → GBM Hgb regression → WHO sex threshold**.

Heavy training runs on Kaggle/HPC (CUDA); the trained weights run inference anywhere
(CUDA > MPS > CPU, auto). See `CLAUDE.md` + `guide.txt` for the full training protocol.

## Inference on a single image

Predict anemic / non-anemic (+ estimated Hgb) for one fresh conjunctiva photo:

```bash
python src/infer_pipeline.py --image /path/to/eye.jpg --sex M
```

Options:

| Flag | Meaning |
|------|---------|
| `--image` | path to the conjunctiva JPG (**required**) |
| `--sex`   | `M` / `F` — sets the WHO cutoff (M<13.0, F<12.0) **and** a model feature (**required**) |
| `--age`   | optional; imputed from the training median (40) if omitted |
| `--device`| force `cuda` / `mps` / `cpu` (default: auto) |
| `--seg-checkpoint` | seg `.pt` (default `outputs/seg/kaggle_segformer_b3/seg_kaggle_final.pt`) |
| `--clf`   | GBM bundle (default `outputs/clf/anemia_gbm_final.joblib`) |
| `--save-mask` | optional path to dump the predicted class mask PNG |

Example (CPU, with age):

```bash
python src/infer_pipeline.py \
  --image "dataset anemia/India/26/20200213_120556.jpg" \
  --sex F --age 39 --device cpu
```

Output (JSON):

```json
{ "sex": "F", "age": 39.0, "pred_hgb": 8.04, "who_cutoff": 12.0,
  "anemic": true, "label": "anemic", "roi_frac": 0.0996, "device": "cpu" }
```

## Trained weights

`infer_pipeline.py` needs the two Kaggle/HPC-trained artifacts at these **exact** relative paths:

```
outputs/clf/anemia_gbm_final.joblib                       # GBM Hgb regressor (+ feat order, cutoffs)
outputs/seg/kaggle_segformer_b3/seg_kaggle_final.pt       # SegFormer weights + cfg
```

Train them with:

```bash
python src/anemia_clf.py                                  # -> anemia_gbm_final.joblib (CPU)
python src/seg.py --config configs/seg_kaggle_final.yaml  # -> seg_kaggle_final.pt   (GPU)
```

> **scikit-learn version must match** between training and inference (the GBM pickle is
> version-sensitive). Artifacts here were trained on **scikit-learn 1.6.1** — pin the same
> locally (`pip install scikit-learn==1.6.1`) or loading the `.joblib` fails with
> `ModuleNotFoundError: No module named '_loss'`.
