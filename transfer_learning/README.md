# Transfer Learning

Stage 2c, and the arm the project's headline result comes from. A network
pretrained on ImageNet supplies the image representation; only the head is
trained on hydrogel data.

Pipeline position:

```
raw images
  -> preprocessing/data.ipynb          (well detection, circular crop)
  -> preprocessing/feature_extraction  (well-wise 60/20/20 split)
  -> THIS FOLDER                       (ResNet18 / VGG16 -> classifier head)
```

The dataset is ~2,100 images across 4 classes — far too small to train a
convolutional network from scratch. A frozen or lightly fine-tuned ImageNet
backbone gives well-conditioned generic filters (edges, texture, colour blobs)
and leaves only a small head to fit.

## Notebooks

### `model_analysis.ipynb` — ResNet18, cell by cell

The reference walkthrough of the fine-tuning setup, split into one step per cell
and ending in a `torchsummary` dump of the architecture. Read this one first.

### `transfer_learning.ipynb` — the main experiment

**Data loading.** `Split_Data/<split>/` holds one subdirectory per timepoint, and
each of those holds the `pH<n> <condition>` class folders. The notebook builds an
`ImageFolder` per timepoint and joins them with `ConcatDataset`, so a single
loader spans all 11 timepoints while the class folders still supply the labels.

**Transforms.** `Resize((224,224))` → `ToTensor()` → `Normalize([0.485,0.456,0.406],
[0.229,0.224,0.225])` (ImageNet statistics, required for a pretrained backbone).

**Model A — fine-tuned ResNet18.**

```python
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 4)   # 512 -> 4
```

`CrossEntropyLoss`, `Adam(lr=1e-3)`, batch 32, 15 epochs, all layers trainable.
Saved to `resnet18_ph_classifier.pth`. Evaluated with a classification report and
a confusion matrix over `["pH5","pH6","pH7","pH8"]`, then one-vs-rest ROC curves
with per-class, micro- and macro-averaged AUC.

**Model B — ResNet18 features + Random Forest** — *the final model*.

The fine-tuned network is reloaded, its classifier is replaced with
`nn.Identity()` so a forward pass emits the 512-dim penultimate embedding, and a
`RandomForestClassifier(n_estimators=100)` is fitted on those embeddings.

```python
feature_extractor = models.resnet18(pretrained=True)
feature_extractor.fc = nn.Linear(feature_extractor.fc.in_features, 4)
feature_extractor.load_state_dict(model.state_dict())   # trained weights
feature_extractor.fc = nn.Identity()                    # 512-dim embedding
```

Note the ordering: `fc` is first shaped like the trained model so
`load_state_dict` matches, and only then swapped for `Identity`.

Swapping softmax for a forest helps at this sample size — the forest is far less
prone to overfitting 512 features on ~2k examples than a jointly-trained linear
head, and its bagged trees handle the uneven class separability (pH 5 vs 6 is a
much subtler colour shift than 6 vs 8).

### `random_forest_cnn.ipynb` — VGG16 features + Random Forest

The same idea with a Keras/TensorFlow backbone, as a cross-check that the result
is not specific to ResNet:

```python
base = VGG16(weights='imagenet', include_top=False, input_shape=(128,128,3))
x    = GlobalAveragePooling2D()(base.output)     # 512-dim
```

All convolutional layers frozen (no fine-tuning at all), images scaled to
`[0,1]`, stratified 80/20 split, then `RandomForestClassifier(n_estimators=100)`.

## Audit and fixes

`transfer_learning.ipynb` had four defects. All are fixed in
`train_transfer.py`; the notebooks are kept as the original record.

| # | Defect | Fix |
|---|---|---|
| 1 | `ImageFolder` over `Split_Data/`, which is **leaky** — all 192 physical wells appear in train, 170/175 also in val/test | Reads `preprocessing/splits.csv` (well-wise, disjoint) |
| 2 | Trained a fixed 15 epochs and **never used val to pick a checkpoint** — the reported model was simply the last one | Val accuracy selects the checkpoint; early stopping (patience 8) |
| 3 | `lr=1e-3` on all 11M pretrained weights, no augmentation, no weight decay | Discriminative LRs (backbone 1e-4, head 1e-3), AdamW `wd=1e-4`, cosine decay, mild geometric augmentation, label smoothing 0.05 |
| 4 | No train/val curves recorded, so over/underfit was never diagnosed | Per-epoch curves saved to `results.json` |

Augmentation is deliberately **geometric only** (flips, ±20° rotation) plus a
mild brightness/contrast jitter of 0.10. Hue and saturation are left alone —
they *are* the pH signal, and the original `ColorJitter(0.2, 0.2)` in `lstm/`
was corrupting exactly the cue being measured.

`random_forest_cnn.ipynb` (VGG16) cannot be re-run: TensorFlow is not installed
in this environment. Its result is therefore not reported below.

## Results

```bash
python3 preprocessing/build_split.py
python3 transfer_learning/train_transfer.py   # ~16 min on Apple M4 (MPS)
```

Well-wise split, 1,177 train / 410 val / 376 test images. Per-**image** accuracy.

| Model | train | val | **test** | macro-F1 | AUC (macro OvR) |
|---|---|---|---|---|---|
| ResNet18 (fine-tuned) | 0.975 | 0.698 | **0.614** | 0.615 | 0.866 |
| ResNet18 + Random Forest | 1.000 | 0.690 | **0.662** | 0.660 | 0.879 |

Training stopped at epoch 18; best val was **0.698 at epoch 10**.

### The headline 83% does not reproduce

The root README reports **83% accuracy, AUC-ROC 0.96** for ResNet18 + RFC. On
the leakage-free split the same architecture reaches **0.662 accuracy, AUC
0.879**. The gap is the leakage, and it is now measured directly.

Frozen ImageNet embeddings + RF, identical features and model, only the split
protocol differing:

| Protocol | Test accuracy |
|---|---|
| Image-level random split (old) | 0.712 |
| Well-level split (corrected) | **0.593** |
| **Inflation from leakage** | **+0.119** |

Compare with the same measurement on hand-crafted colour histograms in
`supervised/`, where inflation was only **+0.022**. This is the expected
asymmetry and the key methodological lesson of the project: **a 512-dim learned
embedding can memorise an individual well; a 64×64 colour histogram cannot.**
Deep models are far more sensitive to this class of leakage, so the deep results
were inflated roughly five times more than the classical ones.

### Fit diagnosis — severe overfitting

| epoch | 1 | 5 | 10 | 14 | 18 |
|---|---|---|---|---|---|
| train | 0.482 | 0.811 | 0.938 | 0.970 | 0.984 |
| val | 0.537 | 0.646 | **0.698** | 0.673 | 0.649 |

Train accuracy rises monotonically to 0.984 while val peaks at 0.698 by epoch 10
and then drifts down — a ~29-point gap. The augmentation, weight decay and
discriminative learning rates slowed it but did not prevent it. With 116 training
**wells** (not 1,177 independent images — the 11 photographs of a well are
highly correlated), an 11M-parameter backbone has far more capacity than the
data supports.

Note also **val 0.698 > test 0.614** for the fine-tuned model. With only 36 test
wells that spread is within noise, but it is a reminder that the checkpoint was
chosen on val, so val is mildly optimistic and test is the honest number.

### Where the errors are

ResNet18 + RF, test confusion (rows = true, cols = predicted):

|  | pH5 | pH6 | pH7 | pH8 |
|---|---|---|---|---|
| **pH5** | **62** | 20 | 9 | 1 |
| **pH6** | 19 | **69** | 6 | 1 |
| **pH7** | 2 | 4 | **47** | 40 |
| **pH8** | 0 | 3 | 22 | **71** |

pH7 is the weakest class (recall 0.51) — 40 of 93 pH7 images are called pH8.
As everywhere in this project, errors are overwhelmingly between adjacent pH
values, and the acidic/alkaline boundary is rarely crossed.

## Final architecture

```
cropped well image
  -> resize 224x224, ImageNet normalise
  -> ResNet18 (ImageNet-pretrained; fine-tuned, backbone lr 1e-4 / head 1e-3,
     AdamW wd 1e-4, cosine, early-stopped on val at epoch 10)
  -> drop fc  ->  512-dim embedding
  -> RandomForestClassifier(400 trees, min_samples_leaf=2)
  -> pH in {5, 6, 7, 8}
```

Test accuracy **0.662**, macro-F1 **0.660**, macro AUC **0.879** — 95% CI on
accuracy [0.613, 0.708].

**This arm is beaten by the hand-crafted-feature baseline.** `supervised/`
Random Forest + elapsed time reaches **0.769** on the same split and test set,
about 11 points better, at a tiny fraction of the compute.

The next section diagnoses why, and the cause is *not* mainly the small dataset:
it is that `fc = nn.Identity()` extracts the most colour-invariant layer of a
network pretrained to ignore colour, on a task where colour is the label.

### Recommended pipeline instead

```
one well = up to 11 cropped images, ordered by elapsed hours
  for each image:
    -> resize 224x224, ImageNet normalise
    -> ResNet18 STEM only: maxpool(relu(bn1(conv1(x))))   # NOT layer4
    -> global average pool                                -> 64-d
    -> concat 8x8x8 HSV histogram (512-d)                 -> 576-d
    -> RandomForest(400 trees, min_samples_leaf=2)        -> class probabilities
  average the probabilities over the well -> argmax
```

Grouped 5-fold CV over all 192 wells: **0.775 per image, 0.917 per well,
0.948 acid-vs-alkaline**. (Elapsed time would add ~+0.013 but is deliberately
excluded — see `supervised/README.md` for why.)

## Why this arm underperforms — diagnosed

The answer is not overfitting, tuning, or too little data. It is that **ImageNet
features are the wrong inductive bias for this task**, and the notebook takes
them from the worst possible place in the network.

### Evidence 1 — accuracy falls monotonically with depth

`layer_probe.py` takes globally-average-pooled features from each ResNet18 stage
(frozen, ImageNet weights, no fine-tuning) and fits the same RF to each.
Grouped 5-fold CV over all 192 wells:

| Stage | dim | per-image accuracy |
|---|---|---|
| stem (`conv1`+pool) | 64 | **0.751** ± 0.017 |
| layer1 | 64 | 0.750 ± 0.028 |
| layer2 | 128 | 0.713 ± 0.028 |
| layer3 | 256 | 0.695 ± 0.018 |
| **layer4** — *what the notebook uses* | 512 | **0.629** ± 0.021 |

The **first convolution's 64 channels beat the final 512-d embedding by 12.2
points**. Depth is actively destroying the signal.

The reason is that ImageNet pretraining is explicitly built to make deep
features *colour-invariant* — its augmentation includes colour jitter, because a
red bus and a blue bus are both buses. Here hue is not a nuisance variable, it
**is the label**. So every stage of the hierarchy discards more of exactly what
we need, and `model.fc = nn.Identity()` — the standard transfer-learning
recipe — extracts the most colour-invariant representation in the network.

### Evidence 2 — twelve hand-picked numbers beat the 512-d embedding

| Feature set | dim | per-image accuracy |
|---|---|---|
| Colour moments (mean/std of RGB+HSV, mask-aware) | **12** | 0.697 |
| Frozen ImageNet layer4 embedding | 512 | 0.629 |

Twelve summary statistics of the crop's colour outperform a 512-dimensional
pretrained representation. That is only possible if the embedding has thrown
the colour information away.

### Evidence 3 — the features encode *time*, not pH

Linear probes on the same features (train → test, well-wise split):

| Features | predicts pH | predicts time-bin |
|---|---|---|
| HSV histogram | 0.612 | **0.888** |
| ResNet frozen (layer4) | 0.585 | **0.896** |
| ResNet fine-tuned (layer4) | 0.678 | **0.904** |

Every representation predicts *degradation stage* far better than pH, which is
exactly what `unsupervised/` found in the raw colour distribution (ARI 0.31 for
time vs 0.08 for pH). Fine-tuning lifts the pH probe from 0.585 to 0.678 — it
does help — but it cannot rebuild colour sensitivity from 116 wells.

### Evidence 4 — fusing raw embeddings makes things *worse*

| Feature set | per-image accuracy |
|---|---|
| HSV histogram + time | **0.785** |
| Embedding + histogram + time (raw concat) | 0.709 |
| Histogram + PCA-32(embedding) + time | 0.784 |

Concatenating 512 weak embedding dimensions onto 512 informative histogram bins
costs 7.6 points. A Random Forest samples `sqrt(n_features)` candidates per
split, so doubling the width with noise halves the chance of picking a useful
bin. Compressing the embedding to 32 PCs removes the damage — but adds nothing
either.

## What actually improves accuracy

All figures below are grouped 5-fold CV over all **192 wells** (every well tested
exactly once), which is far tighter than the single 36-well test set the earlier
table used.

| Pipeline | per-image | **per-well** | acid vs alkaline |
|---|---|---|---|
| layer4 embedding — *notebook's choice* | 0.629 | 0.786 | 0.840 |
| stem features (64) | 0.751 | 0.891 | 0.931 |
| stem + time | 0.775 | 0.896 | 0.955 |
| HSV histogram + time | 0.785 | **0.938** | 0.956 |
| **stem + histogram + time** | **0.788** | 0.922 | **0.957** |

Four changes, in descending order of value:

**1. Aggregate predictions per well — worth ~+15 points.** Averaging the
predicted probabilities across a well's 11 timepoints lifts 0.788 → 0.922. This
is the single largest gain available and it costs nothing: the deployment
scenario already has a time series per dressing, so there is no reason to force
a decision from one photograph.

**2. Take features from the stem, not layer4 — worth +12.2 points.** One line:
use `net.maxpool(net.relu(net.bn1(net.conv1(x))))` instead of
`net.fc = nn.Identity()`. If a pretrained backbone is to be kept at all, this is
where its useful features are.

**4. Reframe as acid vs alkaline — 0.957, and per-well it is perfect.** The
per-well confusion for stem+histogram+time over all 192 wells:

|  | pH5 | pH6 | pH7 | pH8 |
|---|---|---|---|---|
| **pH5** | **48** | 0 | 0 | 0 |
| **pH6** | 5 | **43** | 0 | 0 |
| **pH7** | 0 | 0 | **42** | 6 |
| **pH8** | 0 | 0 | 4 | **44** |

177/192 wells correct (0.922), and **not one well crosses the acid/alkaline
boundary** — 192/192. Since healthy skin is pH 4–6 and chronic wounds pH 7–8,
the clinical question this project exists to answer is already solved; all
residual error is the exact value within a band.

### Ideas not yet tested

- **Train a small CNN from scratch on HSV input.** ImageNet initialisation is a
  liability here, not an asset. A 3-4 layer CNN taking the HSV image has the
  right bias and few enough parameters for 116 wells.
- **Fine-tune only the stem + a head.** The layer probe implies the deep stages
  should be discarded, not adapted.
- **Ordinal regression.** pH is ordered, and every error is off-by-one; a
  regression or ordinal loss encodes that, whereas cross-entropy treats 5-vs-8
  and 7-vs-8 as equally wrong.
- **Fix preprocessing.** 7.1% of images never survive the crop, biased to 0 hr
  and 216/264 hr (`preprocessing/README.md`). Likely a larger gain than anything
  above.
- **Calibrate before averaging.** Per-well aggregation uses raw RF
  probabilities; temperature scaling on the val wells should improve the vote.

## Reproducing the diagnosis

```bash
python3 transfer_learning/diagnose.py       # feature ablation + linear probes
python3 transfer_learning/layer_probe.py    # accuracy by ResNet stage
python3 transfer_learning/improve.py        # grouped CV over feature sets
python3 transfer_learning/best_pipeline.py  # corrected pipeline + per-well
```

## Caveats

- 36 test wells / 376 test images. Because images within a well are strongly
  correlated, the effective sample size is closer to 36 than 376, so the
  interval above is optimistic.
- `random_forest_cnn.ipynb` (VGG16 + RF) is unverified — TensorFlow is absent.
- `ImageFolder` label ordering in the old notebooks depended on every timepoint
  folder holding the same class folders in the same order; the manifest removes
  that fragility.

## Running

Paths are relative to the **repository root**. `Preprocessed_Data/` is
`.gitignore`d and must exist locally. The script writes
`transfer_learning/resnet18_ph_wellwise.pth` (ignored) and `results.json`.
