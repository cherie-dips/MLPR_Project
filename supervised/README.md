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
  -> preprocessing/preprocessing.ipynb          (well detection, circular crop)
  -> preprocessing/preprocessing.ipynb     (well-wise split manifest)
  -> THIS FOLDER                       (rule-based -> SVM / RF / MLP)
```

## Notebooks

### `supervised_models.ipynb` — rule-based classifiers (no training)

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
the colour descriptors and thresholds a single value, the mean
green channel: `< 60 -> pH 5`, `< 75 -> 6`, `< 90 -> 7`, else `8`.

Both are now scored on the corrected data (`supervised/supervised_models.ipynb`),
and **both fail**:

| Rule | 4-way accuracy | acid vs alkaline |
|---|---|---|
| Green-shade pixel voting | **0.156** | 0.302 |
| Mean-green threshold | **0.253** | 0.493 |
| *(majority-class baseline)* | *0.256* | *0.507* |

Pixel voting scores **below chance**. The mean-green threshold matches the
baseline only by collapsing — it predicts pH 8 for 1,933 of 1,963 images,
because thresholds tuned on raw photographs do not transfer to the darker
circular crops.

This is a useful floor: hand-written colour rules carry essentially no pH
signal here, so the learned models' 0.755 is not a marginal gain over
common sense.

### `supervised_models.ipynb` — trained classifiers

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

## Audit and fixes

The original `knn_svm_rf.ipynb` (removed) had five defects, all fixed in
`supervised_models.ipynb`:

| # | Defect | Fix |
|---|---|---|
| 1 | Pooled `Split_Data/train` + `test` and re-split at the **image** level, putting photographs of the same gel on both sides | Uses the well-wise manifest as given, never re-splits |
| 2 | Tuned hyperparameters by cross-validating that same leaky pool | Selection on held-out val wells; test touched once |
| 3 | Imported `KNeighborsClassifier` but **never fitted one** | KNN included and tuned |
| 4 | `StandardScaler` over 512 histogram bins of which **241 are identically zero** → float32 overflow in the MLP | `VarianceThreshold(1e-8)` first; 169 live features remain |
| 5 | `Split_Data/val` never read | val used for model selection |

XGBoost is not installed here, so that variant is not reported.

## Features

```
cropped well image
  -> resize 128x128
  -> BGR to HSV
  -> joint 8x8x8 HSV histogram, L2-normalised    512 dims
  -> VarianceThreshold(1e-8)                     169 live dims
```

A *joint* 3D histogram, not three separate 1D ones — it keeps hue-saturation-value
co-occurrence, which marginal histograms discard. Labels come from the filename
(`re.search(r'pH(\d+)')`).

No elapsed-time feature. It was tested (+0.04) and deliberately dropped — see
the note under Results.

## Evaluation protocol

Two protocols appear below.

- **Single split** — 116 train / 40 val / 36 test wells from
  `preprocessing/splits.csv`. Hyperparameters chosen on val, test scored once.
  Used for the fit diagnosis, since it needs a clean train/val gap.
- **Grouped 5-fold CV** — `StratifiedGroupKFold` grouped by physical well over
  all 192 wells, so every well is tested exactly once. Far tighter than a
  36-well test set (±0.02 vs ±0.13), and the numbers to quote.

Both group by the **physical well** = the `(pH, well)` pair, so no well ever
appears in both train and test.

All results are **per image**: one photograph in, one pH out. Per-well
aggregation (averaging a well's 11 predictions) is deliberately not reported —
it answers an easier question, and with 192 wells the effective sample size
collapses from 1963 to 192.

Note the distinction: the split is *grouped* by well, but every accuracy is
scored on individual images. Grouping the split is the leakage fix; aggregating
predictions would be a different, easier task.

Two figures are given: 4-way pH accuracy, and **acid vs alkaline** (pH 5–6 vs
7–8) — the clinical question, also scored per image.

## Results

Grouped 5-fold CV, all 192 wells, no time feature:

| Model | accuracy | acid/alk | macro-F1 |
|---|---|---|---|
| Baseline (majority) | 0.214 ± 0.031 | 0.473 | 0.088 |
| MLP (64,64) | 0.675 ± 0.004 | 0.905 | 0.673 |
| SVM (RBF, C=10) | 0.707 ± 0.020 | 0.920 | 0.704 |
| KNN (k=9, manhattan) | 0.721 ± 0.022 | 0.929 | 0.719 |
| **Random Forest** (400, leaf 2) | **0.755 ± 0.017** | 0.931 | 0.753 |

Single split, for the fit diagnosis:

| Model | train | val | test |
|---|---|---|---|
| KNN | 0.800 | 0.688 | 0.684 |
| SVM | 0.991 | 0.685 | 0.689 |
| Random Forest | 1.000 | 0.732 | 0.729 |
| MLP | 0.964 | 0.622 | 0.662 |

> **Elapsed time was tested and removed.** Appending hours-since-application was
> worth +0.04. It is *not* label leakage — the design is a complete factorial
> grid (11 timepoints × 4 pH × 48 wells), so hour is independent of pH:
> chi-squared p = 1.000, mutual information 0.0014 nats, and pH predicted from
> hour alone scores **0.225**, *below* the 0.256 majority baseline. It helped
> only as an *interaction*, letting the model use a time-specific decision
> boundary. It is dropped because it assumes deployment follows the lab's
> degradation schedule, where each well sits at a **fixed** pH — an assumption a
> real healing wound violates.

### Fit diagnosis

Every model overfits and the grids could not regularise it away. RF and SVM
reach 1.000 / 0.991 training accuracy against ~0.69–0.73 validation. The RF grid
searched `max_depth ∈ {None,10,20}`, `min_samples_leaf ∈ {1,2,4}` and
`max_features ∈ {sqrt,log2}`; the most-regularised settings did **not** improve
val, so this is not a tuning oversight — 116 training *wells* in 169 effective
dimensions is a thin regime. KNN is the exception (gap +0.113) because k=9 with
uniform weights is heavily smoothed, and it pays ~3 points for that.

Nothing underfits: the majority baseline is 0.214 and every model is far above
it. val ≈ test throughout, so model selection did not overfit the val set.

### Where the errors are

Random Forest, test confusion (rows = true, cols = predicted):

|  | pH5 | pH6 | pH7 | pH8 |
|---|---|---|---|---|
| **pH5** | **80** | 7 | 4 | 1 |
| **pH6** | 18 | **64** | 7 | 6 |
| **pH7** | 5 | 1 | **66** | 21 |
| **pH8** | 0 | 3 | 29 | **64** |

Errors are almost entirely between **adjacent** pH values — 50 are pH7↔pH8 and
25 are pH5↔pH6. Confusion across the acid/alkaline boundary is rare (5 images
from pH 5/6 called pH 8, and 1 the other way), which is why the binary accuracy
(0.931) far exceeds the 4-way figure.

## Final architecture of this arm

```
cropped well image
  -> resize 128x128 -> HSV
  -> joint 8x8x8 HSV histogram, L2-normalised     (512)
  -> VarianceThreshold(1e-8)                      (169)
  -> RandomForest(400 trees, min_samples_leaf=2)
  -> pH in {5, 6, 7, 8}
```

**0.755 accuracy, 0.931 acid-vs-alkaline** (per image, grouped 5-fold CV).

This is the strongest model using colour features alone. The best model overall
is in `transfer_learning/` — the same histogram concatenated with an avg+std
pooled ResNet18 **stem**, reaching **0.808**. Note it uses only the first
convolution of the backbone; the deeper layers actively hurt.

## Remaining caveats

- The 7.1% of images that fail to crop (`preprocessing/README.md`) are absent
  here too, concentrated at 0 hr and 216/264 hr. Accuracy is conditional on the
  crop succeeding.
- The rule-based thresholds were hand-tuned by eye on this same data, so the
  numbers in the table above are training-set numbers and are, if anything,
  optimistic — which makes their sub-baseline scores worse, not better.

## Running

```bash
# open in Jupyter and Run All, or execute headlessly:
jupyter nbconvert --to notebook --execute --inplace supervised/supervised_models.ipynb
```

Paths are relative to the **repository root**. `Preprocessed_Data/` is
`.gitignore`d and must exist locally; features cache to
`supervised/_features.npz`.
