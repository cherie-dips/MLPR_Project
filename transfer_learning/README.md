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

## Final architecture

```
cropped well image
  -> resize 224x224, ImageNet normalise
  -> ResNet18 (ImageNet-pretrained, fine-tuned 15 epochs, Adam 1e-3)
  -> drop the fc layer  ->  512-dim embedding
  -> RandomForestClassifier (100 trees)
  -> pH in {5, 6, 7, 8}
```

Reported in the root README: **83% accuracy, AUC-ROC 0.96**, with strong recall
on pH 7–8 — the clinically important band, since chronic non-healing wounds sit
at pH 7–8 while healthy skin sits at 4–6.

## Caveats

- `models.resnet18(pretrained=True)` is deprecated in current torchvision; the
  modern spelling is `weights=ResNet18_Weights.IMAGENET1K_V1`.
- `ImageFolder` assigns labels from sorted class-folder names *per timepoint
  directory*. This is consistent only while every timepoint contains the same
  set of `pH<n> <condition>` folders in the same order — worth asserting if
  folders are ever added or renamed.
- No augmentation is applied in this folder (`lstm/` does add some).
- The ROC helper is named `plot_roc_curves_timeaware` and can pass a time
  feature, but the ResNet18 used here takes images only, so it runs through its
  single-argument path.

## Running these notebooks

Paths are relative to the **repository root**. Start Jupyter there, or
`os.chdir("..")` in the first cell. `Split_Data/` is `.gitignore`d and must exist
locally. `transfer_learning.ipynb` writes `resnet18_ph_classifier.pth`, which is
also ignored.
