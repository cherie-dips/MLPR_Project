# Supervised

Stage 2a. Classical supervised classifiers — KNN, SVM, Random Forest and
friends — fitted on **hand-crafted colour features**, with no learned
representation.

This is the counterpart to `unsupervised/`, and the baseline arm of the project:
it establishes how much of the signal is plain colour statistics before any deep
model is justified. (`transfer_learning/` and `lstm/` are supervised too; what
sets this folder apart is that the features are engineered by hand rather than
learned by a network.)

Pipeline position:

```
raw images
  -> preprocessing/data.ipynb          (well detection, circular crop)
  -> preprocessing/feature_extraction  (colour descriptors + well-wise split)
  -> THIS FOLDER                       (rule-based -> SVM / RF / MLP)
```

## Notebooks

### `classical_Programming.ipynb` — rule-based classifiers (no training)

Two hand-written classifiers, included as the floor that any learned model must
beat.

**1. Green-shade pixel voting.** Every pixel is read as RGB; a pixel is "green"
if `g > r and g > b`. Green pixels are bucketed by intensity and each bucket
votes for a pH:

| Condition | pH |
|---|---|
| `g < 80`, `r < 70`, `b < 70` | 8 |
| `80 <= g < 130`, `r,b < 100` | 7 |
| `130 <= g < 180`, `r,b < 130` | 6 |
| `g >= 180`, `r,b < 170` | 5 |

The prediction is `argmax` over the four counts. Darker green reads as more
alkaline, lighter green as more acidic.

**2. Feature-threshold classifier.** Reuses the 33-dim descriptor from
`preprocessing/feature_extraction.ipynb` and thresholds a single value, the mean
green channel: `< 60 -> pH 5`, `< 75 -> 6`, `< 90 -> 7`, else `8`.

Both are scored over all of `Preprocessed_Data` with a confusion matrix and
`accuracy_score`.

### `knn_svm_rf.ipynb` — trained classifiers

**Features.** A different, larger descriptor than the one above — a *joint* 3D
HSV histogram rather than three separate 1D ones:

```python
image = cv2.resize(image, (128, 128))
hsv   = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hist  = cv2.calcHist([hsv], [0,1,2], None, [8,8,8], [0,180, 0,256, 0,256])
features = cv2.normalize(hist, hist).flatten()      # 8*8*8 = 512 dims
```

The joint histogram keeps hue–saturation–value *co-occurrence*, which separate
1D histograms discard. Labels come from the filename via `re.search(r'pH(\d+)')`.

**Models, in the order the notebook runs them:**

| Model | Configuration |
|---|---|
| SVM | `GridSearchCV`, `C ∈ {0.1,1,10}`, kernel ∈ {linear, rbf, poly}, gamma ∈ {scale, auto}, 5-fold |
| Random Forest | 100 trees, then `GridSearchCV` over `n_estimators ∈ {100,200,300}`, `max_depth ∈ {10,20,30}`, `min_samples_split ∈ {2,4,6}`, 3-fold |
| RF + time | Same features with elapsed hours parsed from the folder name and appended as a 513th feature |
| XGBoost | 200 estimators, depth 6, lr 0.1 — falls back to RF if `xgboost` is absent |
| MLP | `GridSearchCV` over hidden sizes {(64,), (128,), (64,64)}, activation {relu, tanh}, alpha {1e-4, 1e-3}, adam, `max_iter=500` |
| ResNet18 + RF | Duplicated from `transfer_learning/` — see that folder |

**Why append time.** Gel colour drifts with degradation as well as with pH, so
the same hue means different pH at 0 hr and at 264 hr. Giving the model elapsed
hours lets it condition on where in the degradation curve the sample sits.

## Final architecture of this arm

```
cropped well image
  -> resize 128x128
  -> BGR to HSV
  -> joint 8x8x8 HSV histogram, normalised   (512-dim)
  -> [optional: append elapsed hours]        (513-dim)
  -> SVM / Random Forest / MLP
  -> pH in {5, 6, 7, 8}
```

## Audit and fixes

`knn_svm_rf.ipynb` had three defects. All are fixed in `train_supervised.py`;
the notebook is left as the original exploratory record.

| # | Defect | Fix |
|---|---|---|
| 1 | Pooled `Split_Data/train` + `test` and re-split at the **image** level, so photographs of the same physical gel landed on both sides | Uses `preprocessing/splits.csv` (well-wise) as given, never re-splits |
| 2 | Selected hyperparameters by cross-validating the leaky training pool | Selection on the held-out **val wells** via `PredefinedSplit`; test touched once |
| 3 | Imported `KNeighborsClassifier` but **never fitted one** | KNN included and tuned |
| 4 | `StandardScaler` on 512 histogram bins of which **241 are identically zero** — float32 overflow in the MLP matmul | `VarianceThreshold(1e-8)` before scaling; 169 live features remain |
| 5 | `Split_Data/val` never read | val used for model selection |

XGBoost is not installed in this environment, so that variant is not reported.

## Results

Reproduce with:

```bash
python3 preprocessing/build_split.py
python3 supervised/train_supervised.py     # ~30 s; writes supervised/results.json
```

Well-wise split, 1,177 train / 410 val / 376 test images (116/40/36 wells).
Models are refit on **train only** so the train→val gap stays interpretable.

| Model | train | val | **test** | macro-F1 | train−val |
|---|---|---|---|---|---|
| Baseline (majority class) | 0.258 | 0.254 | 0.253 | 0.101 | +0.005 |
| KNN (k=9, manhattan, uniform) | 0.800 | 0.688 | 0.684 | 0.685 | +0.113 |
| SVM (RBF, C=10) | 0.991 | 0.685 | 0.689 | 0.688 | +0.305 |
| Random Forest (400 trees, depth 20, leaf 2) | 1.000 | 0.732 | 0.729 | 0.729 | +0.268 |
| MLP (64,64; alpha 0.1) | 0.964 | 0.622 | 0.662 | 0.662 | +0.342 |
| **Random Forest + elapsed time** | 1.000 | 0.763 | **0.769** | **0.769** | +0.237 |

All four learners clear the 0.25 majority baseline by a wide margin, so colour
histograms genuinely carry pH signal.

### Fit diagnosis

**Every model overfits**, and the grids were not able to regularise it away:

- RF and SVM reach **1.000 / 0.991 training accuracy** against ~0.69–0.73
  validation — a 27–31 point gap.
- The RF grid searched `max_depth ∈ {None,10,20}`, `min_samples_leaf ∈ {1,2,4}`
  and `max_features ∈ {sqrt,log2}`. The most-regularised settings did **not**
  improve val, so this gap is not a tuning oversight — 1,177 samples in 169
  effective dimensions is simply a thin regime.
- KNN is the exception (gap +0.113) because `k=9` with uniform weights is
  strongly smoothed. It pays about 4 points of test accuracy for that.
- **Nothing underfits.** The majority baseline sits at 0.253 and every model is
  far above it.
- val ≈ test throughout (e.g. RF 0.732 / 0.729), so the held-out estimate is
  stable and the val set was not overfitted by model selection.

### Where the errors are

Random Forest + time, test confusion (rows = true, cols = predicted):

|  | pH5 | pH6 | pH7 | pH8 |
|---|---|---|---|---|
| **pH5** | **81** | 7 | 3 | 1 |
| **pH6** | 17 | **67** | 8 | 3 |
| **pH7** | 3 | 0 | **72** | 18 |
| **pH8** | 1 | 1 | 25 | **69** |

Errors are almost entirely between **adjacent pH values**: 43 of 86 errors are
pH7↔pH8 and 24 are pH5↔pH6. Confusion across the acid/alkaline boundary is
rare (5 images total from pH 5/6 predicted as 8, and 2 the other way).

Clinically this is the forgiving failure mode — healthy skin is pH 4–6 and
chronic wounds pH 7–8, so the model separates *healing from non-healing*
far better than the 0.77 figure suggests, and mostly errs on the exact value
within a band.

### Why elapsed time helps (+0.040)

Appending hours-since-application lifts test accuracy from 0.729 to 0.769. The
`unsupervised/` analysis explains why: colour structure in this dataset tracks
**degradation time roughly four times more strongly than it tracks pH**
(ARI 0.31 vs 0.08). The same hue means different pH at 0 hr and 264 hr, so
telling the model where on the degradation curve a sample sits removes a large
confound. Hours-since-application is known at inference time in the intended
point-of-care use, so this is a legitimate feature rather than leakage.

## Final architecture of this arm

```
cropped well image
  -> resize 128x128
  -> BGR to HSV
  -> joint 8x8x8 HSV histogram, L2-normalised   (512 dims)
  -> VarianceThreshold                          (169 live dims)
  -> [+ elapsed hours]                          (170 dims)
  -> RandomForest(400 trees, max_depth 20, min_samples_leaf 2)
  -> pH in {5, 6, 7, 8}
```

Test accuracy **0.769**, macro-F1 **0.769**.

## Remaining caveats

- The 7.1% of images that fail to crop (see `preprocessing/README.md`) are
  absent here too, and they are concentrated at 0 hr and 216/264 hr. Reported
  accuracy is therefore conditional on the crop succeeding.
- `classical_Programming.ipynb` (rule-based) has not been re-scored on the
  corrected split; its thresholds were hand-tuned against the full
  `Preprocessed_Data` tree, so its accuracy is a training-set number and not
  comparable to the table above.
- The filename `knn_svm_rf.ipynb` still promises a KNN the notebook does not
  contain; `train_supervised.py` is where the KNN actually lives.

## Running

Paths are relative to the **repository root**. `Preprocessed_Data/` is
`.gitignore`d and must exist locally. Features are cached to
`supervised/_features.npz` after the first run.
