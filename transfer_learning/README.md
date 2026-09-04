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
about 11 points better, at a tiny fraction of the compute. On this dataset,
transfer learning does not pay off — 116 training wells is too few to adapt an
ImageNet backbone, while a 512-bin colour histogram already captures nearly all
the available pH signal.

## Recommendations

1. **Freeze more.** Fine-tuning all 11M weights on 116 wells is the direct cause
   of the overfitting. Training only `layer4` + head, or using frozen features
   throughout, should narrow the gap.
2. **Fix preprocessing first.** 7.1% of images never survive cropping, biased
   toward 0 hr and 216/264 hr (`preprocessing/README.md`). That is a larger
   available gain than further architecture work.
3. **Add elapsed time.** It is worth +0.040 to the classical model and is not
   used here at all.
4. **Predict per well, not per image.** See `lstm/` — aggregating a well's 11
   timepoints reaches 0.806.

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
