# Transfer Learning — ResNet18 stem + Random Forest

**The best model in the project: 0.798 per-image accuracy.**

**Notebook:** `resnet_rf.ipynb`

```
raw images
  -> preprocessing/preprocessing.ipynb   (well detection, crop, well-wise split)
  -> THIS FOLDER                         (frozen ResNet18 stem + histogram -> RF)
```

## The design question

ImageNet pretraining builds progressively more **colour-invariant** features with
depth — its augmentation includes colour jitter, because a red bus and a blue bus
are both buses. In this task hue **is** the label, so the useful representation
sits early in the network rather than at the end.

The notebook measures this stage by stage. Global-average-pooled features from
each ResNet18 stage, frozen ImageNet weights, the same Random Forest fitted to
each, grouped 5-fold CV:

| Stage | dim | per-image accuracy |
|---|---|---|
| **stem** (`conv1`+bn+relu+pool) | 64 | **0.749 ± 0.014** |
| layer1 | 64 | **0.752 ± 0.019** |
| layer2 | 128 | 0.703 ± 0.030 |
| layer3 | 256 | 0.678 ± 0.025 |
| layer4 | 512 | 0.611 ± 0.020 |

Accuracy falls with depth. The two shallowest stages are tied within the fold
spread and both carry roughly 14 points more pH signal than the 512-d final
embedding, so this pipeline uses the stem — the cheapest of the two — and
discards everything deeper.

Two supporting measurements in the notebook:

- 12 hand-computed colour moments (0.697) also outperform the 512-d layer4
  embedding, which is only possible if that embedding has discarded colour.
- Concatenating the raw layer4 embedding onto the histogram *reduces* accuracy
  (0.755 → 0.693): a Random Forest samples `sqrt(n_features)` candidates per
  split, so 512 weak dimensions dilute the informative bins.

## How much of ResNet18 is used

| | |
|---|---|
| ResNet18 total parameters | 11,689,512 |
| Used (`conv1` + `bn1`) | **9,536 (0.08%)** |
| Trained by gradient descent | **0** |

The backbone is a frozen filter bank — one 7×7 convolution with 64 filters. No
neural network is trained anywhere in this arm; the only fitted component is the
Random Forest.

## Pooling the stem

The stem output is `(64, 56, 56)`. Global average pooling asks only *what colour
is the gel*; the per-channel **standard deviation** also asks *how uneven is it*,
and degradation makes gels patchy, so that heterogeneity is signal.

| Pooling | dim | accuracy |
|---|---|---|
| avg | 64 | 0.750 ± 0.014 |
| **avg + std** | **128** | **0.797 ± 0.017** |
| 2×2 spatial | 256 | 0.769 ± 0.029 |
| 3×3 spatial | 576 | 0.789 ± 0.034 |

Adding std is worth about +4.7 points, and beats richer *spatial* pooling at a
quarter the dimensionality — the variation matters, its location does not.

## Final architecture

```
cropped well image
  -> resize 224x224, ImageNet normalise
  -> ResNet18 STEM (frozen): maxpool(relu(bn1(conv1(x))))     -> (64, 56, 56)
  -> concat [ spatial mean (64) , spatial std (64) ]          -> 128-d
  -> concat 8x8x8 HSV histogram, L2-normalised (512)          -> 640-d
  -> RandomForest(100 trees, min_samples_leaf=2)
  -> pH in {5, 6, 7, 8}
```

**Why 100 trees.** Accuracy saturates well before this: 400 trees scores 0.808
and 800 scores 0.809, differences inside the ±0.02 fold spread, while 100 keeps
the fitted model roughly 4× smaller (4.6 MB vs 18.3 MB) and 3.5× faster to fit.
`n_estimators` averages variance *across* trees rather than constraining any one
of them, so it trades model size against a small amount of accuracy, not against
fit — training accuracy is ~1.0 at every setting from 25 trees upward.

Building it up, per image, grouped 5-fold CV:

| Features | dim | accuracy | acid/alk |
|---|---|---|---|
| ResNet18 layer4 embedding | 512 | 0.611 ± 0.020 | — |
| stem, avg pool | 64 | 0.750 ± 0.014 | — |
| HSV histogram | 512 | 0.745 ± 0.007 | — |
| stem, avg+std pool | 128 | 0.797 ± 0.017 | — |
| **stem (avg+std) + histogram** | **640** | **0.798 ± 0.016** | **0.952** |

Macro-F1 **0.796**. Feature importance splits across all three blocks — stem
mean 0.28, stem std 0.35, histogram 0.37 — none dominates.

The last two rows sit within one fold's noise of each other on a single CV seed.
Averaged over three seeds the ordering is consistent (0.796 vs 0.800), so the
histogram earns its place, but only just: the 128-d stem alone is a viable
simplification if feature count matters more than half a point.

### Train / validation / test

The CV figures pool held-out predictions across all 192 wells. Fitting once on
the fixed 116-well train split and scoring all three shows the fit:

| split | wells | images | accuracy | macro-F1 | acid/alk |
|---|---|---|---|---|---|
| train | 116 | 1,177 | 0.999 | 0.999 | 1.000 |
| validation | 40 | 410 | 0.751 | 0.751 | 0.944 |
| test | 36 | 376 | 0.785 | 0.785 | 0.968 |

Train → validation gap: **+0.248**. An unpruned Random Forest grows every tree
to purity, so it interpolates the training set by construction — training
accuracy is ~1.0 at every forest size and at every `max_depth` above 6. The gap
is a property of the estimator rather than a symptom: constraining the forest
(`min_samples_leaf` 8/16/32, `max_depth` 4/6) narrows it to +0.10 but costs up
to 11 points of held-out accuracy. The reported figure is measured on held-out
wells, so the training fit does not inflate it.

### Where the errors are

Confusion pooled over the 5 CV folds (rows = true), covering all 1,963 images as
held-out predictions:

|  | pH5 | pH6 | pH7 | pH8 |
|---|---|---|---|---|
| **pH5** | **435** | 42 | 9 | 6 |
| **pH6** | 61 | **396** | 24 | 22 |
| **pH7** | 4 | 16 | **346** | 105 |
| **pH8** | 1 | 12 | 95 | **389** |

Of 1,963 images, 397 are wrong but only **94 (4.8%) cross the acid/alkaline
boundary**; the rest are adjacent-pH slips, dominated by pH7↔pH8 (200 of 397). Since healthy
skin is pH 4–6 and chronic wounds pH 7–8, the clinically important call is far
better answered than the 4-way figure suggests.

## Ideas not yet tried

- **A small CNN trained from scratch on HSV input.** The measurements above
  suggest ImageNet initialisation is not an advantage here; a 3–4 layer CNN has
  the right bias and few enough parameters for 116 training wells.
- **Fine-tuning only the stem** plus a head, rather than a frozen backbone.
- **Ordinal regression.** pH is ordered and nearly every error is off-by-one; an
  ordinal loss encodes that, whereas cross-entropy treats 5-vs-8 and 7-vs-8 as
  equally wrong.
- **Probability calibration.** Raw Random Forest votes are poorly calibrated;
  temperature or isotonic scaling on the validation wells would make the
  confidence scores usable, which matters for a clinical readout.
- **Higher crop coverage.** 7.1% of images are not recovered by the crop stage
  (`preprocessing/README.md`), biased toward 0 hr and 216/264 hr.

## Running

```bash
# open in Jupyter and Run All, or execute headlessly:
jupyter nbconvert --to notebook --execute --inplace transfer_learning/resnet_rf.ipynb
```

Paths inside the notebook are relative to the **repository root**, so start
Jupyter there (or `os.chdir("..")` in the first cell). `Preprocessed_Data/` and
`preprocessing/splits.csv` must be present. Stage features are cached to
`transfer_learning/_*.npz` on the first run and the fitted model is written to
`transfer_learning/resnet_rf_model.pkl`; both are `.gitignore`d.
