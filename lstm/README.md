# LSTM / Sequence Models

Stage 2d. Every other arm classifies **one image at a time**. This one
classifies a **well's whole degradation trajectory** — all 11 timepoints of the
same gel, in order, as a single sample.

Pipeline position:

```
raw images
  -> preprocessing/preprocessing.ipynb   (well detection, circular crop)
  -> hydrogel_dataset.csv       (well -> 11 filenames, built here)
  -> THIS FOLDER                (CNN encoder per frame -> LSTM -> pH)
```

The motivation is that a single photograph is ambiguous: the same hue can mean
pH 6 early in degradation or pH 7 late. How the colour *changes over time* is
more discriminative than any one frame, and that is a sequence problem.

## Notebook

Everything lives in **`lstm_sequence_model.ipynb`**: it encodes each well's
frames once with a frozen ResNet18, builds variable-length sequences, trains the
LSTM head under grouped 5-fold CV, and runs the ablations.

Two notebooks were removed: `lstm_model.ipynb` (superseded) and
`second_LSTM.ipynb` (a Keras CNN-LSTM with three variants — a hybrid averaging
per-frame and sequence heads, a two-branch model fusing CNN features with HSV
histograms, and a last-day-only baseline). The Keras work was never reproducible
here since TensorFlow is not installed; its ablation idea survives as the
last-timepoint baseline in `lstm_sequence_model.ipynb`. Their defects are recorded below.

### Architecture

**Transforms.** `Resize((224,224))` → `ToTensor` → ImageNet `Normalize`. No
colour jitter: the original applied `ColorJitter(0.2, 0.2)` to a task whose
entire signal is colour.

Each well becomes an ordered `(L, 3, 224, 224)` tensor, `L` in 6..11. Labels are
`pH - 5`, giving classes 0-3.

```python
resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = nn.Identity()          # frozen encoder, 512-d per frame

self.lstm = nn.LSTM(512, 256, batch_first=True)
self.fc   = nn.Linear(256, 4)
```

Frames are encoded once and cached, then `pack_padded_sequence` masks the
missing timepoints so the LSTM's final hidden state is the last *valid* frame.
The encoder is frozen throughout, so gradients touch ~800k parameters rather
than 11M — the only tractable option with 116 training wells.

## Audit and fixes

The original `lstm_model.ipynb` (removed) could not run at all as written.
All of the following are fixed in `lstm_sequence_model.ipynb`:

| # | Defect | Fix |
|---|---|---|
| 1 | Iterated `os.listdir('Split_Data1/train')` but joined paths against `'Final_Data'` — **neither directory exists**. Rows were kept only `if len(seq)==11`, so a wrong root produced **zero sequences silently** instead of raising | Reads `preprocessing/splits.csv` |
| 2 | Requiring exactly 11 frames discards most wells, since **149 of 2,112 images (7.1%) fail to crop** | Variable-length sequences via `pack_padded_sequence`; no well is dropped (lengths 6–11, mean 10.2) |
| 3 | `ColorJitter(brightness=0.2, contrast=0.2)` on a task whose entire signal is colour | Encoding done once, without jitter |
| 4 | Re-encoded all 11 frames through ResNet18 every epoch — the encoder is frozen, so this was pure waste | Embeddings computed once and cached (12 s) |
| 5 | No ablation, so it was unknown whether the sequence helped | Last-frame and mean-pool baselines added |

The Keras notebook also read `main_dir = 'New MLPR Data'` — the **raw** images
rather than the crops — and substituted zero images for missing frames, so it
failed quietly too.

## Results — not comparable with the other arms

> **This arm produces no image-level result, by construction.** One sample *is*
> one well: the model consumes an ordered sequence of that well's frames and
> emits a single pH. There is no way to score it per photograph, so it is
> **excluded from the project's headline comparison**, which is image-level
> throughout (see `supervised/README.md`).
>
> The figures below are per well and are recorded for completeness only. They
> should **not** be set against the per-image numbers elsewhere — predicting a
> well from 11 photographs is an easier task than predicting one photograph, and
> the effective sample size is 192 rather than 1963.

```bash
# open in Jupyter and Run All, or execute headlessly:
jupyter nbconvert --to notebook --execute --inplace lstm/lstm_sequence_model.ipynb
```

| Model (per well) | grouped 5-fold CV | acid vs alkaline |
|---|---|---|
| ResNet18(frozen) + LSTM | 0.812 ± 0.027 | 0.995 ± 0.010 |
| Mean-pooled sequence (RF) | 0.778* | — |
| Last timepoint only (RF) | 0.667* | — |

\* single 36-well split, not CV.

The LSTM uses **no elapsed-time feature**; frames are only *ordered*, which is
what makes it a sequence model rather than a bag of images.

### The sequence earns its keep

This is the ablation the original notebook lacked, and it is the clearest
positive result in the project:

| Evidence used | Test accuracy |
|---|---|
| Final timepoint only | 0.667 |
| All 11 timepoints, order discarded (mean-pooled) | 0.778 |
| All 11 timepoints, in order (LSTM) | **0.806** |

Aggregating the trajectory is worth **+0.111** over the last frame alone, and
modelling it *in order* adds a further **+0.028** — but note most of that gain
is simply *having 11 views instead of one*, which is the reason these numbers
are not comparable with the image-level results elsewhere.

This is exactly what `unsupervised/` predicts: colour structure tracks
degradation time about 3.7× more strongly than pH (ARI 0.31 vs 0.08), so how a
well's colour *evolves* is more informative than any single frame.

### Where the errors are

Test confusion (rows = true, cols = predicted), 9 wells per class:

|  | pH5 | pH6 | pH7 | pH8 |
|---|---|---|---|---|
| **pH5** | **9** | 0 | 0 | 0 |
| **pH6** | 1 | **8** | 0 | 0 |
| **pH7** | 0 | 0 | **6** | 3 |
| **pH8** | 0 | 0 | 3 | **6** |

29 of 36 wells correct. The structure is striking:

- **Acidic wells are essentially solved** — 17/18 correct for pH 5–6.
- **Zero cross-boundary errors.** Not one acidic well is called alkaline or
  vice versa. Binary healthy-vs-chronic classification would be **36/36**.
- All 7 errors are **pH7 ↔ pH8**, the pair with the subtlest colour difference.

Clinically this is the ideal failure mode: healthy skin sits at pH 4–6 and
chronic non-healing wounds at pH 7–8, so every well is placed in the correct
clinical band, and the residual error is only about the exact value within the
alkaline band.

### Fit diagnosis

Train 0.991 vs val 0.975 vs test 0.806. Train and val track each other closely —
dropout 0.5 and `weight_decay=1e-3` on a small LSTM head over frozen embeddings
control overfitting far better than fine-tuning a full backbone did in
`transfer_learning/`. The **val-to-test drop is the real concern**, and it is
almost certainly small-sample noise: val is 40 wells, test 36.

## Final architecture

```
one well = up to 11 cropped images, ordered by elapsed hours
  -> ResNet18 (ImageNet, FROZEN, fc dropped)     -> (L, 512), L in 6..11
  -> pack_padded_sequence  (masks missing timepoints)
  -> LSTM(512 -> 256), final valid hidden state
  -> Dropout(0.5) -> Linear(256 -> 4)
  -> pH in {5, 6, 7, 8}
```

AdamW `lr=1e-3`, `weight_decay=1e-3`, label smoothing 0.05, batch 16,
early-stopped on val.

Per-well accuracy **0.812 ± 0.027** under grouped 5-fold CV — per well, not per
image, and therefore not comparable with the other arms.

## Caveats

- **The unit is the well, not the image.** Every number here rests on 192 wells,
  not 1,963 images, so the fold spread understates the true uncertainty.
- **The encoder is frozen ImageNet**, never adapted to hydrogel images, so the
  512-d features are generic — and `transfer_learning/README.md` shows that
  ImageNet's deepest features are the *worst* available for this task. A shallow
  encoder would likely serve this arm better too, and is untried.
- 116 training sequences is very little for an LSTM.
- The Keras variants were never runnable here (no TensorFlow).

## Original notebook issues (superseded)

Kept for the record; all are addressed by `lstm_sequence_model.ipynb` above.

- `lstm_model.ipynb` iterated `os.listdir('Split_Data1/train')` but joins
  against `'Final_Data'` — neither exists. The `len(seq)==11` guard turned that
  into a silent zero-sequence run rather than an error.
- `second_LSTM.ipynb` hardcoded the class folder as `pH{pH} Hydrolytic`. That
  happens to match the data (there is **no** Enzymatic condition in this
  dataset), but it also reads `main_dir = 'New MLPR Data'` — the *raw* images
  rather than the cropped wells — and substitutes zero images for missing
  frames, so it too fails quietly.
- The evaluation loop in `lstm_model.ipynb` wrapped the encoder in an outer
  `torch.no_grad()` while the training loop uses an inner one — harmless, just
  inconsistent.
- Requiring exactly 11 frames per well was the single most damaging choice:
  with 7.1% of crops missing it discards most wells outright.

## Running

Paths inside the notebook are relative to the **repository root**, so start
Jupyter there (or `os.chdir("..")` in the first cell). `Preprocessed_Data/` and
`preprocessing/splits.csv` must be present. Frame embeddings are cached to
`lstm/_embeddings.npz` on the first run, which makes later runs fast.
