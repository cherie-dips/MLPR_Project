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

## Caveats

- **There is no KNN, despite the filename.** `knn_svm_rf.ipynb` imports
  `KNeighborsClassifier` twice but never instantiates or fits it. The notebook
  trains SVM, Random Forest, XGBoost and MLP only. Either add the KNN run or
  rename the notebook — as it stands the name promises a model that is not
  there.
- **The well-wise split is discarded here.** These cells walk `Split_Data/train`
  and `Split_Data/test`, pool every image, then re-split with
  `train_test_split(..., random_state=42)`. That re-split is image-level, so
  photographs of the same physical well land on both sides — the leakage the
  well-wise split in `preprocessing/` exists to prevent. Accuracies from this
  notebook are optimistic and are **not** comparable with the well-wise numbers
  from `transfer_learning/`. To fix, load `train`/`val`/`test` as given instead
  of re-splitting.
- `Split_Data/val` is never read — only `train` and `test` are walked.
- The two notebooks use different descriptors (33-dim separate histograms vs
  512-dim joint histogram); their numbers are not directly comparable either.

## Running these notebooks

Paths are relative to the **repository root**. Start Jupyter there, or
`os.chdir("..")` in the first cell. `Preprocessed_Data/` and `Split_Data/` are
`.gitignore`d and must exist locally.
