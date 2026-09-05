# Transfer Learning — ResNet18 stem + Random Forest

**The best model in the project: 0.796 per-image accuracy.**

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
| **stem** (`conv1`+bn+relu+pool) | 64 | **0.752 ± 0.021** |
| layer1 | 64 | 0.744 ± 0.020 |
| layer2 | 128 | 0.688 ± 0.037 |
| layer3 | 256 | 0.669 ± 0.045 |
| layer4 | 512 | 0.598 ± 0.041 |

Accuracy falls monotonically with depth, and the stem carries **15.3 points**
more pH signal than the 512-d final embedding. So this pipeline uses the stem and
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
| avg | 64 | 0.752 ± 0.021 |
| **avg + std** | **128** | **0.788 ± 0.019** |
| 2×2 spatial | 256 | 0.760 ± 0.026 |
| 3×3 spatial | 576 | 0.781 ± 0.023 |

Adding std is worth about +3.6 points, and beats richer *spatial* pooling at a
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

**Why 100 trees.** Accuracy saturates well before this — 400 and 800 trees score
within the fold spread of 100 — while 100 keeps
the fitted model roughly 4× smaller (4.6 MB vs 18.3 MB) and 3.5× faster to fit.
`n_estimators` averages variance *across* trees rather than constraining any one
of them, so it trades model size against a small amount of accuracy, not against
fit — training accuracy is ~1.0 at every setting from 25 trees upward.

Building it up, per image, grouped 5-fold CV:

| Features | dim | accuracy | acid/alk |
|---|---|---|---|
| ResNet18 layer4 embedding | 512 | 0.598 ± 0.041 | — |
| stem, avg pool | 64 | 0.752 ± 0.021 | — |
| HSV histogram | 512 | 0.759 ± 0.027 | — |
| stem, avg+std pool | 128 | 0.788 ± 0.019 | — |
| **stem (avg+std) + histogram** | **640** | **0.796 ± 0.031** | **0.950** |

Macro-F1 **0.797**. Feature importance splits across all three blocks — none
dominates.

The last two rows sit close together relative to the fold spread; the 128-d stem
alone is a viable simplification if feature count matters more than half a
point.

### Train / validation / test

The CV figures pool held-out predictions across all 192 wells. Fitting once on
the fixed 116-well train split and scoring all three shows the fit:

| split | wells | images | accuracy | macro-F1 | acid/alk |
|---|---|---|---|---|---|
| train | 116 | 1,264 | 0.999 | 0.999 | 1.000 |
| validation | 40 | 435 | 0.763 | 0.764 | 0.933 |
| test | 36 | 392 | 0.781 | 0.781 | 0.957 |

Train → validation gap: **+0.236**. An unpruned Random Forest grows every tree
to purity, so it interpolates the training set by construction — training
accuracy is ~1.0 at every forest size and at every `max_depth` above 6. The gap
is a property of the estimator rather than a symptom: constraining the forest
(`min_samples_leaf` 8/16/32, `max_depth` 4/6) narrows it to +0.10 but costs up
to 11 points of held-out accuracy. The reported figure is measured on held-out
wells, so the training fit does not inflate it.

### Where the errors are

Confusion pooled over the 5 CV folds, covering all 2,091 images as held-out
predictions:

Of 2,091 images, 427 are wrong but only **104 (5.0%) cross the acid/alkaline
boundary**; the rest are adjacent-pH slips, dominated by pH7↔pH8. The notebook
plots the full 4×4 matrix and its collapse to the binary question. Since healthy
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
- **More wells.** A learning curve over subsampled wells still climbs steeply at
  the full 192 and fits `acc = 0.942 - 1.390·n^(-0.453)`; reaching 0.82 needs
  roughly 214 training wells (1.4× current). Well count, not model capacity, is
  the binding constraint — regularising the forest narrows the train/validation
  gap but costs up to 11 points of held-out accuracy.

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
