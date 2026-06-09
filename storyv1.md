<!-- storyv1.md -- the Anemia Detector story so far. Generated narrative; pastel mermaid. -->

# The Anemia Detector Story

> From eyelid photo to a Hemoglobin number: how we built a conjunctiva-pallor anemia
> pipeline across two countries, what shined, what broke, and where it still fails.

This is the running narrative of the project. For the terse engineering contract see
[`CLAUDE.md`](CLAUDE.md); this file is the *why* behind it, drawn graphically.

---

## TL;DR scoreboard

| Stage | Contenders | Winner | Verdict |
|-------|-----------|--------|---------|
| **Segmentation** | SegFormer (MiT) vs UNet++/EfficientNet-B5 | **SegFormer (MiT-b3/b5)** | Transformer global context for a thin curved ROI; trains on CUDA only |
| **Anemia head** | direct-classify vs **Hgb-regression to WHO threshold** | **Regression to WHO** | Prevalence-robust under cohort shift |
| **Anemia model** | handcrafted features + GBM vs deep ROI ViT/CNN | **Features + GBM** | Strong, CPU-cheap, deployed |
| **SSL (SimCLR / NT-Xent)** | with vs without contrastive pretrain | **Marginal** | Implemented, modest LOCO gain, left out of deploy |
| **Cross-domain (CP-AnemiC)** | seg+clf vs clf-only | **Neither** | Pediatric Ghana shift -> r approx 0, model does not transfer |

---

## 1. The mission

```mermaid
flowchart LR
    A[Eyelid photo] --> B[Segment conjunctiva<br/>palpebral + forniceal]
    B --> C[ROI = pallor region]
    C --> D[Color / pallor features]
    D --> E[Predict Hemoglobin g/dL]
    E --> F[WHO sex threshold<br/>M&lt;13.0  F&lt;12.0]
    F --> G[Anemic / Non-anemic]

    classDef data fill:#FFF3B0,stroke:#B69121,color:#5A4500;
    classDef proc fill:#CDE7F0,stroke:#468FAF,color:#012A4A;
    classDef out  fill:#B7E4C7,stroke:#52796F,color:#1B4332;
    class A,C data
    class B,D,E,F proc
    class G out
```

**Anemia makes the inner eyelid pale.** Less blood hemoglobin means less redness in the
conjunctiva. We measure that pallor from a phone photo and turn it into a Hemoglobin estimate,
then a clinical label. Accuracy is the only objective; latency and model size are irrelevant.
Color correction is handled upstream (NormaEngine) and is out of scope here.

---

## 2. The data: two cohorts, many traps

Dataset = **Eyes-defy-anemia**: India 95 + Italy 123 = 218 patients (217 with a Hgb label).
The two cohorts do not look alike, and the files fight you.

```mermaid
flowchart TB
    subgraph RAW["Raw files - landmines"]
        J[JPG landscape<br/>EXIF orient 6] -->|exif_transpose| JP[Portrait RGB]
        M[Mask PNG<br/>corrupt iCCP chunk] -->|PIL fails -> OpenCV reads| MM[Mask raster]
        MM --> ENC{Foreground<br/>encoding?}
        ENC -->|India| AL[ALPHA channel 0/255]
        ENC -->|Italy| RGBN[non-background RGB]
    end
    JP --> CACHE[512px cache<br/>img / lbl / roi]
    AL --> CACHE
    RGBN --> CACHE
    CACHE --> USE[Training + features]

    classDef trap fill:#FAD2E1,stroke:#C9184A,color:#590D22;
    classDef fix  fill:#D8F3DC,stroke:#52796F,color:#1B4332;
    classDef store fill:#E4D8F0,stroke:#7B6CA8,color:#3A2E5C;
    class J,M,ENC trap
    class JP,MM,AL,RGBN fix
    class CACHE,USE store
```

The cohorts also differ wildly in **disease prevalence**, which becomes the single most
important modeling fact downstream:

```mermaid
pie title India cohort anemia prevalence in percent
    "Anemic" : 72
    "Non anemic" : 28
```

```mermaid
pie title Italy cohort anemia prevalence in percent
    "Anemic" : 19
    "Non anemic" : 81
```

India is 72 percent anemic; Italy is 19 percent. A model judged on a pooled average can cheat;
the honest test is **Leave-One-Cohort-Out (LOCO)**: train on one country, test on the other.

---

## 3. Segmentation: who finds the conjunctiva best

The conjunctiva is a thin, curved strip with two classes (palpebral lid surface, deeper
forniceal). We pitted a transformer against a strong CNN.

```mermaid
flowchart TB
    START[Segment 3 classes<br/>bg / palpebral / forniceal<br/>512px, CE + Dice, heavy aug] --> C1
    START --> C2

    subgraph C1["SegFormer - MiT-b3 / b5 backbone"]
        S1[Global self-attention<br/>good for thin curved ROI]
        S1 --> SMPS[On Apple MPS:<br/>backward .view bug<br/>cannot train]
        S1 --> SCUDA[On CUDA Kaggle/HPC:<br/>trains fine, clean masks]
    end

    subgraph C2["UNet++ / EfficientNet-B5"]
        U1[CNN encoder-decoder<br/>MPS-safe]
        U1 --> UROLE[Local smoke + CNN baseline]
    end

    SCUDA --> PICK{Pick winner<br/>by LOCO IoU}
    UROLE --> PICK
    PICK --> WIN[Deployed: SegFormer MiT-b3<br/>seg_kaggle_final.pt]

    classDef win  fill:#B7E4C7,stroke:#52796F,color:#1B4332;
    classDef fail fill:#FAD2E1,stroke:#C9184A,color:#590D22;
    classDef neu  fill:#CDE7F0,stroke:#468FAF,color:#012A4A;
    class WIN,SCUDA win
    class SMPS fail
    class S1,U1,UROLE,START,PICK neu
```

**What failed:** the MiT/SegFormer backbone will not train on Apple MPS. A `.view` op inside the
attention C++ kernel breaks on backward in torch 2.12 and is not Python-patchable. So locally we
can only smoke-test with an EfficientNet encoder; SegFormer is a CUDA-only citizen.

**What shined:** on Kaggle CUDA, SegFormer produces clean, anatomically-correct conjunctiva
crescents. At inference on a held-out source image it cut the ROI tightly (background 86 percent,
palpebral 4 to 5 percent, forniceal 8 to 10 percent) instead of grabbing skin or sclera.

> Honest caveat: head-to-head LOCO IoU between SegFormer and UNet++ comes from the full CUDA
> runs. SegFormer is the chosen deployment segmentor; UNet++/EffB5 stands as the CNN alternative.

---

## 4. Classification: the threshold trap, and the fix

This is the most important design decision in the whole project.

### The trap: a fixed 0.5 threshold breaks under prevalence shift

```mermaid
flowchart LR
    subgraph BAD["Direct binary classify at fixed 0.5"]
        T1[Train India 72% anemic] --> T2[Decision boundary<br/>tuned to 'mostly anemic']
        T2 --> T3[Test Italy 19% anemic]
        T3 --> T4[Mass false positives<br/>boundary in wrong place]
    end

    classDef fail fill:#FAD2E1,stroke:#C9184A,color:#590D22;
    classDef neu  fill:#CDE7F0,stroke:#468FAF,color:#012A4A;
    class T4 fail
    class T1,T2,T3 neu
```

A classifier trained where 72 percent are positive learns a prior that is simply wrong in a
19-percent world. The operating point moves with the population.

### The fix: regress Hemoglobin, then apply the population-fair WHO rule

```mermaid
flowchart LR
    X[ROI pallor features] --> R[Regress continuous Hgb g/dL]
    R --> H{Apply WHO sex cutoff}
    H -->|Male Hgb &lt; 13.0| AN[Anemic]
    H -->|Female Hgb &lt; 12.0| AN
    H -->|else| NO[Non-anemic]

    classDef proc fill:#CDE7F0,stroke:#468FAF,color:#012A4A;
    classDef win  fill:#B7E4C7,stroke:#52796F,color:#1B4332;
    class X,R,H proc
    class AN,NO win
```

Hemoglobin is a physical quantity that does not care about cohort prevalence. Predict the number,
then apply a fixed clinical rule. The decision layer is prevalence-invariant by construction.

### Two anemia engines; the simple one won

```mermaid
flowchart TB
    ROUTE{Anemia model} --> A[Handcrafted features + GBM<br/>anemia_clf.py, CPU]
    ROUTE --> B[Deep ROI ViT / CNN<br/>anemia_cnn.py, GPU]

    A --> AR[Pooled CV AUC approx 0.88<br/>LOCO India to Italy AUC 0.903, acc 0.78<br/>Italy to India harder AUC 0.68]
    B --> BR[Heavier, GPU-bound,<br/>no clear LOCO win over GBM]

    AR --> DEP[Deployed: GBM regressor<br/>anemia_gbm_final.joblib]
    BR --> SHELF[Kept as a research route]

    classDef proc fill:#CDE7F0,stroke:#468FAF,color:#012A4A;
    classDef win  fill:#B7E4C7,stroke:#52796F,color:#1B4332;
    classDef neu  fill:#E2ECE9,stroke:#84A98C,color:#1B4332;
    class A,B,ROUTE proc
    class AR,DEP win
    class BR,SHELF neu
```

The 158-feature handcrafted descriptor (per-channel stats across RGB / HSV / Lab / YCrCb plus
pallor indices like the erythema index and red-green ratios) fed to a HistGradientBoosting
regressor is strong, instant on CPU, and travels well. That is what we deployed.

**Reality check on real source-domain photos** (predicted vs ground-truth Hgb):

| Patient | Sex | Pred Hgb | True Hgb | Error | Label |
|---------|-----|----------|----------|-------|-------|
| India/87 | M | 14.18 | 13.7 | 0.48 | non-anemic (match) |
| India/26 | F | 8.04 | 7.6 | 0.44 | anemic (match) |

---

## 5. SSL: NT-Xent contrastive pretraining, and why it stayed on the bench

India and Italy differ in capture and appearance. A supervised model can latch onto *which site*
instead of *how pale*. SimCLR pretraining was our planned defense: teach the encoder
representations invariant to two augmented views, label-free, on both cohorts.

### How NT-Xent works here

```mermaid
flowchart LR
    IMG[ROI crop] --> V1[Augmented view 1]
    IMG --> V2[Augmented view 2]
    V1 --> E1[Encoder timm backbone]
    V2 --> E2[Encoder shared weights]
    E1 --> P1[Projection head MLP 2048 to 128]
    E2 --> P2[Projection head]
    P1 --> L[NT-Xent loss]
    P2 --> L
    L --> PULL[Pull the two views together<br/>push all other crops apart]

    classDef data fill:#FFF3B0,stroke:#B69121,color:#5A4500;
    classDef proc fill:#E4D8F0,stroke:#7B6CA8,color:#3A2E5C;
    classDef loss fill:#FAD2E1,stroke:#C9184A,color:#590D22;
    class IMG,V1,V2 data
    class E1,E2,P1,P2 proc
    class L,PULL loss
```

NT-Xent (normalized temperature-scaled cross-entropy): L2-normalize all embeddings, build a
cosine-similarity matrix over the 2N views, scale by temperature, mask the self-diagonal, and run
cross-entropy where each view's *positive* target is its sibling view. Concretely in code:

```text
z   = normalize([z1; z2])          # 2N x d
sim = (z @ z.T) / temperature      # 2N x 2N, diagonal masked to -inf
target(i) = sibling_index(i)       # the other augmented view
loss = cross_entropy(sim, target)
```

### Pallor-preserving augmentation (the crucial twist)

Standard SimCLR uses grayscale and strong color jitter. **We must not.** Pallor *is* a color
signal; a color-invariant encoder would be blind to anemia. So the two views use geometry plus
only mild photometric perturbation (brightness, contrast, blur, tiny hue/sat shift).

```mermaid
flowchart LR
    subgraph KEEP["Allowed - nuisance only"]
        K1[crop / flip / affine]
        K2[mild brightness-contrast]
        K3[slight blur, tiny hue-sat]
    end
    subgraph BAN["Banned - destroys the signal"]
        B1[grayscale]
        B2[strong color jitter]
    end
    KEEP --> GOOD[Invariant to site, sensitive to pallor]
    BAN --> BADX[Invariant to pallor = blind to anemia]

    classDef ok  fill:#D8F3DC,stroke:#52796F,color:#1B4332;
    classDef no  fill:#FAD2E1,stroke:#C9184A,color:#590D22;
    class K1,K2,K3,GOOD ok
    class B1,B2,BADX no
```

### How SSL plugs into the classifier

```mermaid
sequenceDiagram
    participant SSL as ssl_pretrain.py
    participant Disk as SSL checkpoint
    participant CNN as anemia_cnn.py

    SSL->>SSL: SimCLR on ROI crops (both cohorts, label-free)
    SSL->>Disk: save encoder state_dict + backbone name
    CNN->>Disk: init_weights = that checkpoint
    Disk-->>CNN: load encoder (strict=False, head ignored)
    CNN->>CNN: fine-tune on Hgb (MSE) or anemia (BCE)
    Note over SSL,CNN: SSL backbone MUST equal anemia backbone<br/>or the state_dict keys will not match
```

The contrastively-pretrained encoder becomes the *initialization* for the supervised deep
classifier. `build_backbone` loads it with `strict=False` so only the shared encoder transfers;
the projection head is discarded and a fresh regression/classification head is trained.

### The outcome

```mermaid
flowchart LR
    SSLp[SSL NT-Xent pretrain] --> GAIN{LOCO improvement?}
    GAIN -->|small, not decisive| KEEP[Kept in codebase,<br/>left out of deployed model]

    classDef neu fill:#E2ECE9,stroke:#84A98C,color:#1B4332;
    classDef warm fill:#FFE5D9,stroke:#E07A5F,color:#6B2D1B;
    class SSLp neu
    class GAIN,KEEP warm
```

SSL was implemented end-to-end and smoke-tested. With only 218 images, contrastive pretraining
gave at best a modest, non-decisive LOCO bump - not enough to justify the extra GPU stage in the
deployed path. It remains a research lever (and a natural place to later fold in unlabeled
target-domain images), but the shipped model is the simple features + GBM regressor.

---

## 6. Deployment: from throwaway folds to real weights

Originally every protocol trained a model, scored it, and threw it away. To deploy we added
persistence and a single inference pipeline.

```mermaid
flowchart LR
    subgraph TRAIN["Kaggle / HPC - train once on all data"]
        SF[seg final mode] --> PT[seg_kaggle_final.pt<br/>state_dict + cfg]
        GB[anemia_clf fit_final] --> JL[anemia_gbm_final.joblib<br/>estimator + feat order + cutoffs]
    end
    subgraph INFER["Local - infer_pipeline.py on one fresh image"]
        IMGN[New eyelid photo + sex] --> SEG[Load SegFormer .pt]
        SEG --> ROIp[Predicted ROI = palpebral + forniceal]
        ROIp --> FEAT[features_from_arrays at 512px]
        FEAT --> GBMp[Load GBM .joblib]
        GBMp --> HGBp[Predicted Hgb]
        HGBp --> LAB[WHO sex threshold -> label]
    end
    PT -.downloaded.-> SEG
    JL -.downloaded.-> GBMp

    classDef tr fill:#E4D8F0,stroke:#7B6CA8,color:#3A2E5C;
    classDef inf fill:#CDE7F0,stroke:#468FAF,color:#012A4A;
    classDef out fill:#B7E4C7,stroke:#52796F,color:#1B4332;
    class SF,GB,PT,JL tr
    class IMGN,SEG,ROIp,FEAT,GBMp,HGBp inf
    class LAB out
```

Two self-contained artifacts. The `.pt` carries its own training cfg so inference rebuilds the
exact architecture; the `.joblib` carries the estimator, the pinned feature order, the WHO
cutoffs and a median-age fallback. Nothing else from the training run is needed at inference -
features are computed on the fly from the fresh image.

---

## 7. War stories: the infrastructure that fought back

```mermaid
flowchart TB
    H1["Apple MPS: MiT backward .view bug"] --> F1["SegFormer cannot train locally; smoke on EfficientNet"]
    H2["Kaggle P100 is CUDA sm_60; new torch is sm_70+ only"] --> F2["no kernel image error; switch to T4 x2"]
    H3["Second T4 idle wasted quota"] --> F3["add nn.DataParallel via multi_gpu flag; unwrap module on save"]
    H4["sklearn skew: trained 1.6.1, local 1.9.0"] --> F4["joblib ModuleNotFoundError _loss; pin scikit-learn 1.6.1"]
    H5["Bare python in a notebook cell"] --> F5["SyntaxError; prefix shell lines with a bang"]

    classDef fail fill:#FAD2E1,stroke:#C9184A,color:#590D22;
    classDef fix fill:#D8F3DC,stroke:#52796F,color:#1B4332;
    class H1,H2,H3,H4,H5 fail
    class F1,F2,F3,F4,F5 fix
```

Each was a fast, loud failure (a stack trace, not a silent wrong number), which is the good kind.
The DataParallel fix is config-gated (`multi_gpu: true`) so it is a no-op on a single GPU, MPS, or
CPU, and the checkpoint is unwrapped on save so inference loads clean keys.

---

## 8. The unseen test: CP-AnemiC, where the model met its limit

We then threw a genuinely unseen dataset at it: **CP-AnemiC** - 710 pre-cropped conjunctiva
strips from Ghana, pediatric (age recorded in months), with the segmentation already provided in
each PNG's alpha channel. We ran every image two ways and compared.

```mermaid
flowchart TB
    IMG2[CP-AnemiC crop<br/>alpha = provided mask] --> R1[Route 1: our SegFormer finds ROI]
    IMG2 --> R2[Route 2: use provided alpha mask as ROI]
    R1 --> CLF[Same GBM classifier]
    R2 --> CLF
    CLF --> AGG[Aggregate n=709<br/>1 corrupt PNG]

    classDef data fill:#FFF3B0,stroke:#B69121,color:#5A4500;
    classDef proc fill:#CDE7F0,stroke:#468FAF,color:#012A4A;
    classDef res  fill:#FAD2E1,stroke:#C9184A,color:#590D22;
    class IMG2 data
    class R1,R2,CLF proc
    class AGG res
```

### What shifted at once

```mermaid
flowchart TB
    ROOT["Domain shift: India + Italy to Ghana"] --> REG[Region]
    ROOT --> AGE[Age]
    ROOT --> CLI[Clinical]
    ROOT --> PIPE[Pipeline]
    REG --> R1[different skin tone]
    REG --> R2[different camera and lighting]
    AGE --> A1[adults to infants]
    AGE --> A2[age feature far out of range]
    CLI --> C1["adult WHO cutoff vs pediatric near 11"]
    CLI --> C2[different Hgb distribution]
    PIPE --> P1[no color correction applied]
    PIPE --> P2[pallor signal is camera dependent]

    classDef root fill:#E4D8F0,stroke:#7B6CA8,color:#3A2E5C;
    classDef br fill:#CDE7F0,stroke:#468FAF,color:#012A4A;
    classDef lf fill:#FFE5D9,stroke:#E07A5F,color:#6B2D1B;
    class ROOT root
    class REG,AGE,CLI,PIPE br
    class R1,R2,A1,A2,C1,C2,P1,P2 lf
```

### The result: it did not transfer

```mermaid
flowchart LR
    ACT["Actual Hgb spans 4 to 15 g/dL"] --> MODEL["GBM applied to Ghana data"]
    MODEL --> PRED["Predictions collapse to an 11 to 14 band"]
    PRED --> RR["Pearson r approx 0 - no correlation"]
    PRED --> BB["Bias +2 g/dL - systematic over-prediction"]
    RR --> FAIL["Does not transfer"]
    BB --> FAIL

    classDef neu fill:#CDE7F0,stroke:#468FAF,color:#012A4A;
    classDef fail fill:#FAD2E1,stroke:#C9184A,color:#590D22;
    class ACT,MODEL,PRED neu
    class RR,BB,FAIL fail
```

| Route | MAE | RMSE | Bias (pred - gt) | Pearson r | Label acc vs REMARK |
|-------|-----|------|------------------|-----------|----------------------|
| R1 seg + clf | 2.62 | 3.33 | +1.99 | **0.004** | 0.513 |
| R2 clf only (provided mask) | 2.64 | 3.39 | +2.13 | **0.016** | 0.513 |

Predictions collapsed into a flat 11 to 14 band regardless of the true value - **r approximately
zero**, a systematic +2 g/dL over-prediction, and label accuracy (0.513) below the majority-class
baseline (prevalence 0.597). The two routes were statistically identical, so the earlier
single-image hunch that "classifier-only wins on pre-cropped data" was just noise.

**The lesson:** the segmentation source is not the bottleneck - the *classifier itself does not
transfer*. With r near zero, no threshold or bias calibration can rescue it; this needs
feature-space domain adaptation. See the saved scatter plots in `testunseendatav1/`.

---

## 9. Robustness roadmap: how to survive the next domain

```mermaid
flowchart TB
    ROOT[Make it robust] --> COL[Color first]
    ROOT --> DA["Use target data - DA"]
    ROOT --> DG["No target data - DG"]
    ROOT --> DEC[Decision layer]
    ROOT --> SAF[Safety]
    COL --> COL1["color constancy and white balance"]
    COL --> COL2["Reinhard transfer to source domain"]
    COL --> COL3["sclera or skin as internal white reference"]
    DA --> DA1["re-fit the GBM head on a small Ghana split"]
    DA --> DA2["recalibrate only if rank preserved"]
    DA --> DA3["feature alignment CORAL / MMD / DANN"]
    DG --> DG1[pool more source cohorts]
    DG --> DG2["illumination-only augmentation, never grayscale"]
    DG --> DG3["worst-group objectives GroupDRO / IRM"]
    DG --> DG4[SSL on unlabeled target images]
    DEC --> DEC1["population-correct cutoffs pediatric under 11"]
    DEC --> DEC2[age and altitude adjustments]
    SAF --> SAF1[out-of-distribution detection]
    SAF --> SAF2[abstain instead of guessing the mean]

    classDef root fill:#E4D8F0,stroke:#7B6CA8,color:#3A2E5C;
    classDef br fill:#B7E4C7,stroke:#52796F,color:#1B4332;
    classDef lf fill:#E2ECE9,stroke:#84A98C,color:#1B4332;
    class ROOT root
    class COL,DA,DG,DEC,SAF br
    class COL1,COL2,COL3,DA1,DA2,DA3,DG1,DG2,DG3,DG4,DEC1,DEC2,SAF1,SAF2 lf
```

The highest-leverage move for a pallor task is color: normalize illumination and use an in-image
white reference, then adapt with a handful of labeled target images. The in-domain CP-AnemiC
re-fit is the diagnostic that tells us whether the features carry Hgb signal at all on this camera,
or whether it is pure domain shift.

---

## 10. Decisions, distilled

```mermaid
flowchart TB
    D1[Device-agnostic: CUDA &gt; MPS &gt; CPU, AMP on CUDA] --> OK
    D2[Segment with SegFormer, judge by LOCO IoU] --> OK
    D3[Regress Hgb then WHO threshold, not fixed 0.5] --> OK
    D4[Deploy features + GBM, keep deep route on shelf] --> OK
    D5[Report at converged final epoch: val == test in CV/LOCO] --> OK
    D6[SSL pallor-preserving augs only; modest gain, not shipped] --> OK
    D7[Pin scikit-learn 1.6.1 across train and inference] --> OK
    OK[Sound, evidence-backed choices]

    classDef win fill:#B7E4C7,stroke:#52796F,color:#1B4332;
    classDef neu fill:#CDE7F0,stroke:#468FAF,color:#012A4A;
    class OK win
    class D1,D2,D3,D4,D5,D6,D7 neu
```

---

## Appendix: where things live

```mermaid
flowchart LR
    subgraph src
        P[paths.py] 
        DIO[data_io.py]
        SEGc[seg.py]
        FEATc[anemia_features.py]
        CLFc[anemia_clf.py]
        CNNc[anemia_cnn.py]
        SSLc[ssl_pretrain.py]
        INFc[infer_pipeline.py]
        CPb[batch_cpanemic.py]
    end
    subgraph outputs
        SEGo[seg/...final.pt]
        CLFo[clf/anemia_gbm_final.joblib]
        SSLo[ssl/...pt]
    end
    subgraph eval
        TUD[testunseendatav1/<br/>csv + metrics + scatter plots]
    end
    SEGc --> SEGo
    CLFc --> CLFo
    SSLc --> SSLo
    INFc --> TUD
    CPb --> TUD

    classDef code fill:#E4D8F0,stroke:#7B6CA8,color:#3A2E5C;
    classDef art  fill:#FFF3B0,stroke:#B69121,color:#5A4500;
    classDef ev   fill:#CDE7F0,stroke:#468FAF,color:#012A4A;
    class P,DIO,SEGc,FEATc,CLFc,CNNc,SSLc,INFc,CPb code
    class SEGo,CLFo,SSLo art
    class TUD ev
```

*End of storyv1. The pipeline runs end to end on its source domain; the open frontier is
cross-domain robustness.*
