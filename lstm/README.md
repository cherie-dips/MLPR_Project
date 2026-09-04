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

## Caveats

These notebooks are the least finished arm and need repair before the numbers
mean anything.

- **`lstm_model.ipynb` reads from directories that do not exist.** The loop
  iterates `os.listdir('Split_Data1/train')` but then joins against
  `'Final_Data'` — neither name appears anywhere else in the repo. Since a row
  is kept only `if len(well_sequence) == 11`, a wrong root silently yields
  **zero** sequences rather than an error. Point both at `Preprocessed_Data`.
- **`second_LSTM.ipynb` hardcodes `pH{pH} Hydrolytic`**, so every Enzymatic
  image is skipped, and it reads `main_dir = 'New MLPR Data'` — the *raw*
  images, not the cropped wells. Missing frames become zero images rather than
  raising, so this too fails quietly.
- The test-set evaluation loop in `lstm_model.ipynb` calls `resnet(...)` inside
  `torch.no_grad()` at the outer level, but the training loop's encoder call has
  its own inner `no_grad` — harmless, just inconsistent.
- 132 training sequences against an LSTM head is very little data. Treat any
  accuracy from this folder as provisional, and compare it against the
  last-day-only ablation before concluding the temporal model helps.

## Running these notebooks

Paths are relative to the **repository root**. Start Jupyter there, or
`os.chdir("..")` in the first cell — note that `hydrogel_dataset.csv` now lives
in this folder, so it is `lstm/hydrogel_dataset.csv` from the root. The image
directories are `.gitignore`d and must exist locally.
