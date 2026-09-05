# Transfer Learning — ResNet18 stem + Random Forest

**The best model in the project: 0.808 per-image accuracy.**

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
| **stem** (`conv1`+bn+relu+pool) | 64 | **0.751 ± 0.017** |
| layer1 | 64 | 0.750 ± 0.028 |
| layer2 | 128 | 0.713 ± 0.028 |
| layer3 | 256 | 0.695 ± 0.018 |
| layer4 | 512 | 0.629 ± 0.021 |

Accuracy falls monotonically with depth, and the first convolution's 64 channels
carry 12.2 points more pH signal than the 512-d final embedding. So this pipeline
uses the stem and discards everything deeper.

Two supporting measurements in the notebook:

- 12 hand-computed colour moments (0.697) also outperform the 512-d layer4
  embedding, which is only possible if that embedding has discarded colour.
- Concatenating the raw layer4 embedding onto the histogram *reduces* accuracy
  (0.755 → 0.693): a Random Forest samples `sqrt(n_features)` candidates per
  split, so 512 weak dimensions dilute the informative bins. Compressing the
  embedding to 32 PCs removes the damage (0.762) but adds nothing.

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
| avg | 64 | 0.752 ± 0.017 |
| **avg + std** | **128** | **0.793 ± 0.017** |
| 2×2 spatial | 256 | 0.769 ± 0.029 |
| 3×3 spatial | 576 | 0.789 ± 0.034 |

Adding std is worth +4.1 points, and beats richer *spatial* pooling at a quarter
the dimensionality — the variation matters, its location does not.

## Final architecture

```
cropped well image
  -> resize 224x224, ImageNet normalise
  -> ResNet18 STEM (frozen): maxpool(relu(bn1(conv1(x))))     -> (64, 56, 56)
  -> concat [ spatial mean (64) , spatial std (64) ]          -> 128-d
  -> concat 8x8x8 HSV histogram, L2-normalised (512)          -> 640-d
  -> RandomForest(400 trees, min_samples_leaf=2)
  -> pH in {5, 6, 7, 8}
```

Building it up, per image, grouped 5-fold CV:

| Features | dim | accuracy | acid/alk |
|---|---|---|---|
| ResNet18 layer4 embedding | 512 | 0.629 ± 0.021 | 0.840 |
| colour moments | 12 | 0.697 ± 0.023 | 0.901 |
| stem, avg pool | 64 | 0.752 ± 0.017 | 0.931 |
| HSV histogram | 512 | 0.755 ± 0.017 | 0.931 |
| stem, avg+std pool | 128 | 0.793 ± 0.017 | — |
| **stem (avg+std) + histogram** | **640** | **0.808 ± 0.021** | **0.955** |

Macro-F1 **0.806**. Feature importance splits across all three blocks — stem
mean, stem std and histogram each contribute, none dominates.

### Where the errors are

Confusion pooled over the 5 folds (rows = true):

|  | pH5 | pH6 | pH7 | pH8 |
|---|---|---|---|---|
| **pH5** | **442** | 36 | 10 | 4 |
| **pH6** | 61 | **400** | 21 | 21 |
| **pH7** | 2 | 17 | **357** | 95 |
| **pH8** | 1 | 12 | 97 | **387** |

Of 1,963 images, 377 are wrong but only **88 (4.5%) cross the acid/alkaline
boundary**; the rest are adjacent-pH slips, dominated by pH7↔pH8. Since healthy
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
