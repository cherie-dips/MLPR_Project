# LSTM / Sequence Models

Stage 2d. Every other arm classifies **one image at a time**. This one
classifies a **well's whole degradation trajectory** — all 11 timepoints of the
same gel, in order, as a single sample.

Pipeline position:

```
raw images
  -> preprocessing/data.ipynb   (well detection, circular crop)
  -> hydrogel_dataset.csv       (well -> 11 filenames, built here)
  -> THIS FOLDER                (CNN encoder per frame -> LSTM -> pH)
```

The motivation is that a single photograph is ambiguous: the same hue can mean
pH 6 early in degradation or pH 7 late. How the colour *changes over time* is
more discriminative than any one frame, and that is a sequence problem.

## The sequence index — `hydrogel_dataset.csv`

Built by the first cells of `lstm_model.ipynb`. One row per well, 192 rows:

| Well | pH | Day 1 | Day 2 | ... | Day 11 |
|---|---|---|---|---|---|
| 1 | 5 | `cropped_0hr_pH5_W1.JPG` | `cropped_24hr_pH5_W1.JPG` | ... | `cropped_264hr_pH5_W1.JPG` |

The `Day N` columns are labels only — the real elapsed times are unevenly spaced:

```
Day 1..11  ->  0, 24, 30, 48, 72, 95, 120, 168, 192, 216, 264 hr
```

`second_LSTM.ipynb` carries this as the `day_to_hour` dict.

**Split.** Stratified per pH by well: 10 test, 33 train, 5 val per class
(`random_state=42`), written to `train.csv` / `val.csv` / `test.csv`. Because a
row *is* a well, this split is inherently well-wise — the leakage discussed in
`supervised/README.md` cannot occur here.

## Notebooks

### `lstm_model.ipynb` — ResNet18 encoder + PyTorch LSTM

**Transforms.** `ToPILImage` → `Resize((224,224))` → `RandomHorizontalFlip` →
`ColorJitter(brightness=0.2, contrast=0.2)` → `ToTensor`.

`ColorJitter` on a task whose entire signal is colour is a deliberate trade —
it guards against the model latching onto absolute illumination of a particular
photo session, at the cost of perturbing the very cue being measured. Worth an
ablation.

Each well becomes a `(11, 3, 224, 224)` tensor; rows that do not yield exactly 11
frames are dropped. Labels are `pH - 5`, giving classes 0–3.

**Architecture.**

```python
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Identity()          # frozen encoder, 512-dim per frame
resnet.eval()

cnn_lstm = nn.Sequential(
    nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True),
    nn.Linear(256, 4),
)
```

Forward pass — each of the 11 frames is encoded independently under
`torch.no_grad()`, the per-frame vectors are stacked to `(B, 11, 512)`, the LSTM
runs over the time axis, and **only the final timestep's hidden state**
`outputs[:, -1, :]` is fed to the classifier:

```
(B, 11, 3, 224, 224)
  -> ResNet18 per frame, frozen   -> (B, 11, 512)
  -> LSTM(512 -> 256)             -> (B, 11, 256)
  -> take last timestep           -> (B, 256)
  -> Linear(256 -> 4)             -> pH
```

`CrossEntropyLoss`, `Adam(lr=1e-4)` on the LSTM and head only, batch 4, 20
epochs. The encoder is frozen throughout, so gradients touch ~800k parameters
rather than 11M — the only tractable option with 132 training sequences.

### `second_LSTM.ipynb` — Keras CNN-LSTM, trained end to end

A TensorFlow counterpart with a small CNN learned from scratch instead of a
pretrained encoder. Input `(11, 128, 128, 3)`, fed by an `ImageSequenceGenerator`
(a `keras.utils.Sequence` that assembles one well's 11 frames per sample and
substitutes a zero image for any missing frame).

**Baseline.**

```
TimeDistributed(Conv2D(16,3x3,relu)) -> TimeDistributed(MaxPool2D)
TimeDistributed(Conv2D(32,3x3,relu)) -> TimeDistributed(MaxPool2D)
TimeDistributed(Flatten) -> LSTM(64) -> Dense(4, softmax)
```

`TimeDistributed` applies one shared CNN to every frame, so the convolutional
weights are learned once and reused across the 11 timesteps.

Three variants follow:

| Variant | Idea |
|---|---|
| **Hybrid** | Conv 32/64, then averages two heads — a per-frame `TimeDistributed(Dense)` softmax taken at the last timepoint, and an `LSTM(128)` sequence softmax. Hedges between single-image and trajectory evidence. |
| **Feature-enhanced** | Two branches — `LSTM(64)` over CNN features and `LSTM(32)` over the hand-crafted HSV histograms from `supervised/` — concatenated into `Dense(64)` → `Dense(4)`. Fuses learned and engineered features. |
| **Last-day only** | Uses `Lambda(lambda x: x[:, -1])` to keep only the final frame. The ablation that tells you whether the sequence is earning its keep. |

## Audit and fixes

`lstm_model.ipynb` could not run at all as written. Fixed in `train_lstm.py`;
the notebooks are kept as the original record.

| # | Defect | Fix |
|---|---|---|
| 1 | Iterated `os.listdir('Split_Data1/train')` but joined paths against `'Final_Data'` — **neither directory exists**. Rows were kept only `if len(seq)==11`, so a wrong root produced **zero sequences silently** instead of raising | Reads `preprocessing/splits.csv` |
| 2 | Requiring exactly 11 frames discards most wells, since **149 of 2,112 images (7.1%) fail to crop** | Variable-length sequences via `pack_padded_sequence`; no well is dropped (lengths 6–11, mean 10.2) |
| 3 | `ColorJitter(brightness=0.2, contrast=0.2)` on a task whose entire signal is colour | Encoding done once, without jitter |
| 4 | Re-encoded all 11 frames through ResNet18 every epoch — the encoder is frozen, so this was pure waste | Embeddings computed once and cached (12 s) |
| 5 | No ablation, so it was unknown whether the sequence helped | Last-frame and mean-pool baselines added |

`second_LSTM.ipynb` (Keras) cannot be re-run: TensorFlow is not installed here.
Its `main_dir = 'New MLPR Data'` also points at the **raw** images rather than
the crops, which should be corrected before it is trusted.

## Results

```bash
python3 preprocessing/build_split.py
python3 lstm/train_lstm.py     # ~3 min (12 s encoding + 76 epochs on cached embeddings)
```

**These numbers are per WELL, not per image** — 116 train / 40 val / **36 test**
wells. They are *not* directly comparable with `supervised/` and
`transfer_learning/`, which classify single images: predicting a well from 11
photographs is a genuinely easier task than predicting one photograph.

| Model | train | val | **test** | macro-F1 |
|---|---|---|---|---|
| Baseline (majority) | 0.250 | – | 0.250 | – |
| Last timepoint only (RF) | 1.000 | 0.725 | 0.667 | 0.667 |
| Mean-pooled sequence (RF) | 1.000 | 0.850 | 0.778 | 0.777 |
| **ResNet18(frozen) + LSTM** | 0.991 | 0.975 | **0.806** | **0.805** |

Early stopping at epoch 76; best val 0.975 at epoch 51.

### Under grouped 5-fold CV (all 192 wells)

The single 36-well test set above carries a ±13 point interval. `cv_lstm.py`
re-runs the same model with every well tested exactly once:

| | per well | acid vs alkaline |
|---|---|---|
| ResNet18(frozen) + LSTM | **0.812 ± 0.027** | **0.995 ± 0.010** |

The 4-way figure is *lower* than simply averaging Random Forest predictions over
a well (0.928, see `supervised/`), but the acid/alkaline figure is the highest
in the project — 0.995 means essentially every well is placed in the correct
clinical band.

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
modelling it *in order* adds a further **+0.028**. Most of the gain comes from
having 11 views rather than one, but temporal ordering does contribute.

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
early-stopped on val at epoch 51.

Test accuracy **0.806**, macro-F1 **0.805**.

## Caveats — read before quoting the 0.806

- **The test set is 36 wells.** 0.806 is 29/36. The 95% Wilson interval is
  **[0.650, 0.903]** — a spread of 25 points. It overlaps every other model in
  the project, so *this arm cannot be claimed to beat `supervised/` (0.769)*
  on this evidence. It is suggestive, not established.
- **Not comparable to the per-image arms.** Per-well prediction gets 11 images
  of evidence per decision. The honest cross-arm comparison would be to
  majority-vote the `supervised/` per-image predictions within each well, which
  has not been done.
- **The encoder is frozen ImageNet**, never adapted to hydrogel images, so the
  512-dim features are generic. Fine-tuning it jointly is untried and, on 116
  training wells, likely to overfit as badly as `transfer_learning/` did.
- 116 training sequences is very little for an LSTM. A repeated grouped
  cross-validation over all 192 wells would give a far more trustworthy estimate
  than this single 116/40/36 split, and is the clear next step.
- `second_LSTM.ipynb` remains unverified (no TensorFlow).

## Original notebook issues (superseded)

Kept for the record; all are addressed by `train_lstm.py` above.

- `lstm_model.ipynb` iterates `os.listdir('Split_Data1/train')` but joins
  against `'Final_Data'` — neither exists. The `len(seq)==11` guard turned that
  into a silent zero-sequence run rather than an error.
- `second_LSTM.ipynb` hardcodes the class folder as `pH{pH} Hydrolytic`. That
  happens to match the data (there is **no** Enzymatic condition in this
  dataset), but it also reads `main_dir = 'New MLPR Data'` — the *raw* images
  rather than the cropped wells — and substitutes zero images for missing
  frames, so it too fails quietly.
- The evaluation loop in `lstm_model.ipynb` wraps the encoder in an outer
  `torch.no_grad()` while the training loop uses an inner one — harmless, just
  inconsistent.
- Requiring exactly 11 frames per well was the single most damaging choice:
  with 7.1% of crops missing it discards most wells outright.

## Running

`train_lstm.py` runs from the **repository root**. The notebooks' paths are also
root-relative — start Jupyter there or `os.chdir("..")` in the first cell; note
`hydrogel_dataset.csv` now lives in this folder. Image directories are
`.gitignore`d and must exist locally. Embeddings are cached to
`lstm/_embeddings.npz` after the first run.
