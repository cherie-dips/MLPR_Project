# Preprocessing

Stage 1 of the pipeline. Turns raw well-plate photographs into clean, uniform,
single-well crops and the split that every downstream model consumes.

Everything in `supervised/`, `unsupervised/`, `transfer_learning/` and `lstm/`
assumes this stage has already run.

## Dataset

pH-sensitive fluorescent silk fibroin hydrogel dressings imaged in well plates.

| Property | Value |
|---|---|
| Classes | pH 5, 6, 7, 8 |
| Wells | 48 per pH (192 total) |
| Timepoints | 0, 24, 30, 48, 72, 95, 120, 168, 192, 216, 264 hr (11) |
| Degradation condition | `Hydrolytic` / `Enzymatic` |
| Images | 2,112 |

On disk the raw tree is `New MLPR Data/<time> hr/pH<n> <condition>/<time>hr_pH<n>_W<well>.JPG`.

## Notebooks

### `data.ipynb` — well detection and circular crop

Isolates the single hydrogel well from each photograph, so the model sees gel
colour rather than plate background, labels or lighting.

1. `cv2.imread` → `cv2.cvtColor(BGR2HSV)`.
2. `cv2.inRange(hsv, lower_green, upper_green)` for a green mask.
3. `cv2.GaussianBlur(mask, (5,5), 0)` to soften mask edges.
4. Morphological `MORPH_CLOSE` then `MORPH_OPEN` with a 7×7 kernel — closes
   pinholes inside the well, then removes speckle outside it.
5. `cv2.findContours(..., RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`.
6. Contour selection — discard `area < 5000`, then compute

   ```
   (x, y), radius = cv2.minEnclosingCircle(cnt)
   circularity    = 4 * pi * area / (arcLength^2 + 1e-5)
   ```

   and keep the contour with `radius > 30` and `circularity > 0.6`. Most cells
   take the **largest** such contour; the later cells take the **left-most**,
   for plates where a second well intrudes into the frame.
7. Build a filled circle mask at `(x, y, r)`, `cv2.bitwise_and` it with the
   image, and crop the bounding box `[y-r:y+r, x-r:x+r]`.
8. Write to `Preprocessed_Data/<time> hr/pH<n> <condition>/cropped_<original>.JPG`.

**On the HSV thresholds.** The notebook holds several cells with different
`lower_green`/`upper_green` bounds — `[20,20,100]–[70,255,255]`,
`[55,30,40]–[75,110,120]`, `[65,100,50]–[85,255,255]`, `[35,90,80]–[75,255,255]`
and others. These are not redundant. Gel colour shifts with both pH and
degradation time, so a single global threshold loses wells at the extremes; the
bounds were re-tuned per pH band and per timepoint and the folder re-run. The
later cells also raise the blur to `(21,21)` and the kernel to 15×15, which
suppresses reflections on the darker late-timepoint images.

### `feature_extraction.ipynb` — descriptors and the train/val/test split

**Part 1 — hand-crafted colour descriptors** (the input to `supervised/`):

- *RGB histogram* — 256 bins per channel, sum-normalised.
- *HSV histogram* — H in 180 bins, S and V in 256 bins, concatenated and
  sum-normalised.
- *Colour moments* — per-channel mean, standard deviation and skewness (9 values).
- *Compact vector* — the form the classifiers actually use: 8-bin H, S and V
  histograms (24) concatenated with the 9 colour moments, on a 64×64 resize.
  **33 dimensions.**

The rationale is that pH is encoded almost entirely in hue and brightness of the
gel, so a colour histogram carries most of the usable signal at a tiny fraction
of the dimensionality of the raw pixels.

**Part 2 — well-wise split** (`random.seed(42)`):

Well IDs are parsed from the filename, shuffled, and divided **60/20/20 by well**,
then images are copied into `Split_Data/{train,val,test}/<time> hr/pH<n> <condition>/`.

The split is on the *well*, not the image. Because each well is photographed at
all 11 timepoints, an image-level split would put 10 photographs of the same
physical gel in train and the 11th in test — near-duplicate leakage that inflates
accuracy. Splitting by well keeps every image of a gel on one side of the
boundary.

## Output contract

| Path | Produced by | Consumed by |
|---|---|---|
| `Preprocessed_Data/` | `data.ipynb` | `supervised/` |
| `Split_Data/{train,val,test}/` | `feature_extraction.ipynb` | `transfer_learning/`, `supervised/` |

## Running these notebooks

Paths inside are relative to the **repository root** (`New MLPR Data`,
`Preprocessed_Data`, `Split_Data`), not to this folder. Start Jupyter from the
repo root, or add `os.chdir("..")` in the first cell.

The image directories are `.gitignore`d — they are not in the repository and
must be present locally.
