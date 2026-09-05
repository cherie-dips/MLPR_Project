# Supervised

Classical supervised classifiers — KNN, SVM, Random Forest and an MLP — fitted on
hand-crafted colour features. This is the baseline arm: it establishes how much
of the pH signal is plain colour statistics before any pretrained network is
involved.

**Notebook:** `supervised_models.ipynb`

```
raw images
  -> preprocessing/preprocessing.ipynb   (well detection, crop, well-wise split)
  -> THIS FOLDER                         (rule-based baselines -> KNN / SVM / RF / MLP)
```

## Features

```
cropped well image
  -> resize 128x128
  -> BGR to HSV
  -> joint 8x8x8 HSV histogram, L2-normalised    512 bins
  -> VarianceThreshold(1e-8)                     169 live features
```

A *joint* 3D histogram rather than three separate 1D ones: it keeps
hue–saturation–value co-occurrence, which marginal histograms discard. 241 of the
512 bins are identically zero across the dataset, so a variance filter removes
them before any distance-based model. Labels are parsed from the filename with
`re.search(r'pH(\d+)')`.

## Evaluation

`StratifiedGroupKFold(5)` grouped by the physical `(pH, well)` pair, so each of
the 192 wells is used as test data exactly once and no well appears in both train
and test.

All results are **per image** — one photograph in, one pH out. Two figures are
reported: 4-way pH accuracy, and **acid vs alkaline** (pH 5–6 vs 7–8), which is
the clinical distinction, also scored per image.

Hyperparameters come from a grid search on the held-out validation wells:

| Model | Search space | Selected |
|---|---|---|
| KNN | `k ∈ {1,3,5,9,15,25}`, weights, metric | k=9, manhattan, uniform |
| SVM | `C ∈ {0.1,1,10,100}`, kernel, gamma | RBF, C=10, gamma=scale |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` | 400 trees, depth 20, leaf 2, sqrt |
| MLP | hidden sizes, alpha | (64,64), alpha 0.1 |

## Rule-based baselines

Two hand-written classifiers threshold green intensity directly, with fixed
constants rather than fitted parameters:

- **Green-shade pixel voting** — a pixel counts as green if `g > r and g > b`;
  green pixels are bucketed by intensity and each bucket votes for a pH.
- **Mean-green threshold** — thresholds the mean green channel at 60 / 75 / 90,
  ignoring the black circular-mask corners.

| Rule | 4-way accuracy |
|---|---|
| Green-shade pixel voting | 0.162 |
| Mean-green threshold | 0.252 |
| *(majority-class baseline)* | *0.252* |

Both sit at or below the majority baseline, and the threshold rule collapses —
it predicts pH 8 for 2,060 of 2,091 images. Fixed colour thresholds carry little
pH signal on the cropped images, so the learned models below are not competing
against a strong heuristic.

## Results

Grouped 5-fold CV over all 192 wells, per image:

| Model | accuracy | acid/alk | macro-F1 |
|---|---|---|---|
| Baseline (majority) | 0.231 ± 0.009 | 0.490 | 0.094 |
| MLP (64,64) | 0.680 ± 0.032 | 0.904 | 0.680 |
| KNN (k=9, manhattan) | 0.713 ± 0.015 | 0.920 | 0.714 |
| SVM (RBF, C=10) | 0.726 ± 0.016 | 0.914 | 0.726 |
| **Random Forest** (400, leaf 2) | **0.762 ± 0.031** | 0.933 | 0.764 |

All four learners clear the 0.231 baseline by a wide margin, so colour histograms
genuinely carry pH signal.

### Train / validation / test

The CV figures above pool held-out predictions, so they say nothing about the fit
on training data. Fitting the Random Forest once on the fixed 116-well train
split and scoring all three:

| split | wells | images | accuracy | macro-F1 | acid/alk |
|---|---|---|---|---|---|
| train | 116 | 1,264 | 1.000 | 1.000 | 1.000 |
| validation | 40 | 435 | 0.738 | 0.739 | 0.910 |
| test | 36 | 392 | 0.745 | 0.746 | 0.931 |

Train → validation gap: **+0.262**. Validation and test agree closely, so the
held-out estimate is stable.

### Fit

The Random Forest reaches 1.000 training accuracy against ~0.73 validation. The grid
covered `max_depth`, `min_samples_leaf` and `max_features`, and the most heavily
regularised settings did not improve validation accuracy — with 116 training
*wells* in 169 effective dimensions, the gap reflects the size of the dataset
rather than the choice of hyperparameters. KNN is the most smoothed model
(gap +0.11) and pays about 3 points of accuracy for it. Nothing underfits.

### Where the errors are

Random Forest, confusion pooled over the 5 folds (rows = true):

|  | pH5 | pH6 | pH7 | pH8 |
|---|---|---|---|---|
| **pH5** | **429** | 49 | 12 | 2 |
| **pH6** | 56 | **399** | 29 | 19 |
| **pH7** | 7 | 18 | **320** | 126 |
| **pH8** | 2 | 14 | 108 | **373** |

Errors are almost entirely between **adjacent** pH values, dominated by
pH7↔pH8. Confusion across the acid/alkaline boundary is rare, which is why the
binary accuracy (0.931) far exceeds the 4-way figure.

Clinically this is the forgiving failure mode: healthy skin sits at pH 4–6 and
chronic non-healing wounds at pH 7–8, so the model separates healing from
non-healing far better than 0.755 suggests, and mostly errs on the exact value
within a band.

## Final architecture of this arm

```
cropped well image
  -> resize 128x128 -> HSV
  -> joint 8x8x8 HSV histogram, L2-normalised     (512)
  -> VarianceThreshold(1e-8)                      (169)
  -> RandomForest(400 trees, min_samples_leaf=2)
  -> pH in {5, 6, 7, 8}
```

**0.762 accuracy, 0.933 acid-vs-alkaline**, per image.

This is the strongest model built from colour features alone. The best model
overall is in `transfer_learning/` — the same histogram concatenated with an
avg+std pooled ResNet18 stem, reaching **0.796**.

Note the forest sizes differ between the two arms: this one uses the 400 trees
its grid search selected, while `transfer_learning/` uses 100, chosen to keep the
deliverable model small. The difference is worth about one point of accuracy,
inside the fold-to-fold spread.

## Running

```bash
# open in Jupyter and Run All, or execute headlessly:
jupyter nbconvert --to notebook --execute --inplace supervised/supervised_models.ipynb
```

Paths inside the notebook are relative to the **repository root**, so start
Jupyter there (or `os.chdir("..")` in the first cell). `Preprocessed_Data/` and
`preprocessing/splits.csv` must be present. Features are cached to
`supervised/_features.npz` on the first run.
