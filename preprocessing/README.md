# Preprocessing

Stage 1. Turns raw well-plate photographs into circular single-well crops, then
defines the train/test split every model uses.

Everything in `supervised/`, `unsupervised/`, `transfer_learning/` and `lstm/`
starts from the outputs of this stage.

**Notebook:** `preprocessing.ipynb`

## Dataset

pH-sensitive fluorescent silk fibroin hydrogel dressings imaged in well plates.

| Property | Value |
|---|---|
| Classes | pH 5, 6, 7, 8 |
| Wells | 48 per pH → **192 physical wells** |
| Timepoints | 0, 24, 30, 48, 72, 95, 120, 168, 192, 216, 264 hr (11) |
| Condition | Hydrolytic |
| Raw images | 2,112 = 11 × 4 × 48 (complete factorial grid) |

```
New MLPR Data/<time> hr/pH<n> Hydrolytic/<time>hr_pH<n>_W<well>.JPG
```

**A W-number alone does not identify a well.** W-numbers restart at 1 for each
pH, so `W13` names four different physical gels. The physical well is the
`(pH, well)` pair. This is the unit the split groups by.

## Well detection and circular crop

Isolates the hydrogel well from each photograph so the model sees gel colour
rather than plate background, labels or lighting.

1. `cv2.imread` → `cvtColor(BGR2HSV)`
2. `cv2.inRange(hsv, lower_green, upper_green)` → green mask
3. `GaussianBlur(mask, (5,5), 0)`
4. Morphological `MORPH_CLOSE` then `MORPH_OPEN`, 7×7 kernel — closes pinholes
   inside the well, then removes speckle outside it
5. `findContours(..., RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`
6. Keep contours with `area ≥ 5000`, `radius > 30` and `circularity > 0.6`,
   where

   ```
   (x, y), radius = cv2.minEnclosingCircle(cnt)
   circularity    = 4 * pi * area / (arcLength^2 + 1e-5)
   ```

   and take the largest survivor
7. Fill a circle mask at `(x, y, r)`, `bitwise_and` it with the image, crop to
   the bounding box
8. Write `Preprocessed_Data/.../cropped_<original>.JPG`

Gel colour shifts with both pH and degradation time, so a single HSV range does
not fit the whole dataset. Five ranges are held in `GREEN_RANGES` and tried in
order until one yields a valid well.

### Coverage

| | |
|---|---|
| Raw images | 2,112 |
| Cropped | **1,963** |
| Not recovered | 149 (7.1%) |

Recovery is lowest where the green mask fits least well — at 0 hr, where the gel
is palest, and at 216/264 hr, where it is darkest:

| Timepoint | 0 hr | 24 | 30 | 48 | 95 | 120 | 168 | 192 | 216 | 264 |
|---|---|---|---|---|---|---|---|---|---|---|
| Not recovered | 68 | 4 | 1 | 2 | 6 | 6 | 4 | 11 | 24 | 23 |

The notebook plots this by timepoint and by pH. Downstream results are therefore
conditional on a successful crop, and the fresh and heavily-degraded states are
slightly under-represented. A per-image adaptive threshold — Otsu on the
saturation channel, or a Hough circle on the well rim, which does not depend on
gel colour — would raise coverage further.

## Colour descriptors

The features the models consume, computed in `supervised/supervised_models.ipynb`:

- **Joint HSV histogram** — `calcHist` over all three channels at once with
  8 bins each (512 bins), L2-normalised. A joint 3D histogram keeps
  hue–saturation–value co-occurrence, which three separate 1D histograms would
  discard. 241 bins are identically zero across the dataset and are dropped by a
  variance filter, leaving 169 live features.
- **Colour moments** — per-channel mean and standard deviation of RGB and HSV,
  ignoring the black circular-mask corners (12 values).

pH is encoded almost entirely in the gel's hue and brightness, so colour
descriptors carry most of the usable signal at a fraction of raw-pixel
dimensionality.

## The split

Each well is photographed at 11 timepoints, so those 11 images are
near-duplicates of one another. The split therefore groups by the physical
`(pH, well)` pair: every image of a well lands on the same side, and the
evaluation measures generalisation to unseen gels.

The 192 wells are shuffled with `random.seed(42)`, stratified by pH, and split
60/20/20. The result is written to `preprocessing/splits.csv`:

| | train | val | test |
|---|---|---|---|
| **wells** | 116 | 40 | 36 |
| images | 1,177 | 410 | 376 |
| pH 5 / 6 / 7 / 8 | 294/304/281/298 | 106/104/97/103 | 92/95/93/96 |

The notebook asserts that no well appears in more than one split.

## Evaluation protocol

Models are scored with `StratifiedGroupKFold(5)` grouped by physical well, so
each of the 192 wells is used as test data exactly once — a tighter estimate
(±0.02) than a single 36-well holdout (±0.13). The 60/20/20 columns above provide
a fixed holdout for the train/validation gap reported in `supervised/`.

All accuracies across the project are **per image**: one photograph in, one pH
out.

## Output

| Path | Consumed by |
|---|---|
| `Preprocessed_Data/` | all model folders |
| `preprocessing/splits.csv` | all model folders |

## Running

```bash
# open in Jupyter and Run All, or execute headlessly:
jupyter nbconvert --to notebook --execute --inplace preprocessing/preprocessing.ipynb
```

Paths inside the notebook are relative to the **repository root**, so start
Jupyter there (or `os.chdir("..")` in the first cell). The image directories are
`.gitignore`d and must be present locally. Set `RUN_FULL_CROP = True` to
regenerate `Preprocessed_Data/` from the raw images.
