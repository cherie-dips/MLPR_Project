# Transfer Learning

Stage 2c, and the arm the project's headline result comes from. A network
pretrained on ImageNet supplies the image representation; only the head is
trained on hydrogel data.

Pipeline position:

```
raw images
  -> preprocessing/preprocessing.ipynb          (well detection, circular crop)
  -> preprocessing/feature_extraction  (well-wise 60/20/20 split)
  -> THIS FOLDER                       (ResNet18 / VGG16 -> classifier head)
```

The dataset is ~2,100 images across 4 classes — far too small to train a
convolutional network from scratch. A frozen or lightly fine-tuned ImageNet
backbone gives well-conditioned generic filters (edges, texture, colour blobs)
and leaves only a small head to fit.

## Notebook

Everything in this folder lives in **`resnet_rf.ipynb`**, in order:

1. **Diagnosis** — accuracy of each ResNet18 stage (the core finding)
2. **Parameter count** — how little of the backbone is actually used
3. **Pooling study** — avg vs avg+std vs spatial pooling of the stem
4. **The final model** — stem (avg+std) + HSV histogram → Random Forest
5. **Evaluation** — grouped 5-fold CV, confusion matrices, per-class report
6. **Fit on all data** → `resnet_rf_model.pkl`

Four notebooks were removed as superseded: `transfer_learning.ipynb` (fine-tuned
ResNet18, 0.662), `model_analysis.ipynb` (a duplicate of its ResNet18 section),
`random_forest_cnn.ipynb` (VGG16 + RF, never reproducible here — TensorFlow is
not installed) and the diagnostic scripts, whose results are now cells in
`resnet_rf.ipynb`. Their defects are recorded below.

## Audit and fixes

The original `transfer_learning.ipynb` (removed) had four defects, all fixed
in `resnet_rf.ipynb`:

| # | Defect | Fix |
|---|---|---|
| 1 | `ImageFolder` over `Split_Data/`, which is **leaky** — all 192 physical wells appear in train, 170/175 also in val/test | Reads `preprocessing/splits.csv` (well-wise, disjoint) |
| 2 | Trained a fixed 15 epochs and **never used val to pick a checkpoint** — the reported model was simply the last one | Val accuracy selects the checkpoint; early stopping (patience 8) |
| 3 | `lr=1e-3` on all 11M pretrained weights, no augmentation, no weight decay | Discriminative LRs (backbone 1e-4, head 1e-3), AdamW `wd=1e-4`, cosine decay, mild geometric augmentation, label smoothing 0.05 |
| 4 | No train/val curves recorded, so over/underfit was never diagnosed | Train/val curves plotted in the notebook |

Augmentation is deliberately **geometric only** (flips, ±20° rotation) plus a
mild brightness/contrast jitter of 0.10. Hue and saturation are left alone —
they *are* the pH signal, and the original `ColorJitter(0.2, 0.2)` in `lstm/`
was corrupting exactly the cue being measured.

The VGG16 variant (`random_forest_cnn.ipynb`, removed) was never re-run —
TensorFlow is not installed here — so it is not reported below. The layer probe
makes it unlikely to have helped: it also read from the deepest, most
colour-invariant layer.

## Results

```bash
# open in Jupyter and Run All, or execute headlessly:
jupyter nbconvert --to notebook --execute --inplace transfer_learning/resnet_rf.ipynb
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

**As written, this arm loses to the hand-crafted-feature baseline.**
`supervised/` Random Forest on the HSV histogram reaches **0.755** under grouped
CV against **0.629** for the layer4 embedding — 13 points better, at a fraction
of the compute.

The cause is *not* mainly the small dataset: `fc = nn.Identity()` extracts the
most colour-invariant layer of a network pretrained to ignore colour, on a task
where colour is the label. Once that is fixed the ordering reverses — the
corrected pipeline in **Final architecture** above reaches **0.808**, the best
result in the project. The next section is the diagnosis.

### Recommended pipeline instead

```
one cropped well image
  -> resize 224x224, ImageNet normalise
  -> ResNet18 STEM only: maxpool(relu(bn1(conv1(x))))   # NOT layer4  -> (64,56,56)
  -> concat [ spatial mean (64) , spatial std (64) ]                  -> 128-d
  -> concat 8x8x8 HSV histogram, L2-normalised (512)                  -> 640-d
  -> RandomForest(400 trees, min_samples_leaf=2)
  -> pH in {5, 6, 7, 8}
```

Grouped 5-fold CV, **per image**: accuracy **0.808 ± 0.021**, macro-F1 **0.806**,
acid-vs-alkaline **0.955 ± 0.010**. This is the best model in the project.

```bash
# open in Jupyter and Run All, or execute headlessly:
jupyter nbconvert --to notebook --execute --inplace transfer_learning/resnet_rf.ipynb
```

## Why this arm underperforms — diagnosed

The answer is not overfitting, tuning, or too little data. It is that **ImageNet
features are the wrong inductive bias for this task**, and the notebook takes
them from the worst possible place in the network.

### Evidence 1 — accuracy falls monotonically with depth

`resnet_rf.ipynb` takes globally-average-pooled features from each ResNet18 stage
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

| Feature set | accuracy |
|---|---|
| HSV histogram (512) | **0.755 ± 0.017** |
| layer4 embedding + histogram (raw concat) | 0.693 ± 0.025 |
| histogram + PCA-32(embedding) | 0.762 ± 0.024 |

Concatenating 512 weak embedding dimensions onto 512 informative histogram bins
costs 6.2 points. A Random Forest samples `sqrt(n_features)` candidates per
split, so doubling the width with noise halves the chance of picking a useful
bin. Compressing the embedding to 32 PCs removes the damage — but adds nothing
either.

## What actually improves accuracy

All figures below are **per image**, under grouped 5-fold CV — every well tested
exactly once, far tighter than the single 36-well test set the earlier table
used. Per-well aggregation is not reported: it answers an easier question and
shrinks the effective sample from 1963 to 192.

| Pipeline | dim | accuracy | acid vs alkaline |
|---|---|---|---|
| layer4 embedding — *notebook's choice* | 512 | 0.629 ± 0.021 | 0.840 |
| colour moments | 12 | 0.697 ± 0.023 | 0.901 |
| stem, avg pool | 64 | 0.751 ± 0.017 | 0.931 |
| HSV histogram | 512 | 0.755 ± 0.017 | 0.931 |
| stem (avg) + histogram | 576 | 0.775 ± 0.017 | 0.948 |
| stem, avg+**std** pool | 128 | 0.793 ± 0.017 | — |
| **stem (avg+std) + histogram** | **640** | **0.808 ± 0.021** | **0.955** |

Four changes, in descending order of value:

**1. Take features from the stem, not layer4 — worth +12.2 points.** One line:
use `net.maxpool(net.relu(net.bn1(net.conv1(x))))` instead of
`net.fc = nn.Identity()`. If a pretrained backbone is to be kept at all, this is
where its useful features are.

**2. Pool the stem with std as well as mean — worth +4.1 points**
(0.752 → 0.793), the largest single gain after the layer choice. Global average
pooling asks only *what colour is the gel*; the per-channel standard deviation
over the 56×56 map also asks *how uneven is it*. Degradation makes gels patchy,
so that heterogeneity is real signal, and averaging alone discards it. Note it
beats richer spatial poolings (2×2 → 0.769, 3×3 → 0.789) at a quarter the
dimensionality — the *variation* matters, its spatial location does not.

**3. Add the colour histogram alongside the stem — worth +1.5 points**
(0.793 → 0.808). Complementary: 64 learned filters plus an explicit
high-resolution colour distribution.

**4. Reframe as acid vs alkaline — 0.955.** Confusion for the final pipeline,
per image, pooled over the 5 folds:

|  | pH5 | pH6 | pH7 | pH8 |
|---|---|---|---|---|
| **pH5** | **442** | 36 | 10 | 4 |
| **pH6** | 61 | **400** | 21 | 21 |
| **pH7** | 2 | 17 | **357** | 95 |
| **pH8** | 1 | 12 | 97 | **387** |

Of 1,963 images, 377 are wrong but only **88 (4.5%) cross the acid/alkaline
boundary** — the rest are adjacent-pH slips, dominated by pH7↔pH8 (192 of 377).
Since healthy skin is pH 4–6 and chronic wounds pH 7–8, the clinical question is
far better answered than the 4-way figure suggests.

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
- **Calibrate the probabilities.** Raw RF votes are poorly calibrated;
  temperature or isotonic scaling on the val wells would make the confidence
  scores usable, which matters more than accuracy for a clinical readout.

## Caveats

- Images within a well are strongly correlated, so even under grouped CV the
  effective sample size is nearer the 192 wells than the 1,963 images; treat the
  ±0.02 fold spread as a lower bound on the true uncertainty.
- The VGG16 variant was never verified — TensorFlow is absent here.
- `ImageFolder` label ordering in the old notebooks depended on every timepoint
  folder holding the same class folders in the same order; the manifest removes
  that fragility.

## Running

Paths inside the notebook are relative to the **repository root**, so start
Jupyter there (or `os.chdir("..")` in the first cell). `Preprocessed_Data/` and
`preprocessing/splits.csv` must be present. Stage features are cached to
`transfer_learning/_*.npz` on the first run; the fitted model is written to
`transfer_learning/resnet_rf_model.pkl`. All are `.gitignore`d.
