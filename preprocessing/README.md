# Preprocessing

Stage 1. Turns raw well-plate photographs into clean single-well crops, and
defines the train/val/test split every downstream model consumes.

Everything in `supervised/`, `unsupervised/`, `transfer_learning/` and `lstm/`
depends on this stage.

## Dataset

pH-sensitive fluorescent silk fibroin hydrogel dressings imaged in well plates.

| Property | Value |
|---|---|
| Classes | pH 5, 6, 7, 8 |
| Wells | 48 per pH → **192 physical wells** |
| Timepoints | 0, 24, 30, 48, 72, 95, 120, 168, 192, 216, 264 hr (11) |
| Degradation condition | `Hydrolytic` only |
| Raw images | 2,112 = 11 × 4 × 48 (complete grid, verified) |

`New MLPR Data/<time> hr/pH<n> Hydrolytic/<time>hr_pH<n>_W<well>.JPG`

> **Correction.** Earlier drafts of these READMEs described an Enzymatic
> condition alongside Hydrolytic. The current dataset has **no Enzymatic
> images** — every class folder in `New MLPR Data/`, `Preprocessed_Data/` and
> `Split_Data/` is `pH<n> Hydrolytic`. Enzymatic folders existed in an older
> `MLPR_Dataset_Copy/` tree that was deleted from `main` before this work.

**A well is not identified by its W-number.** W-numbers restart at 1 for each
pH, so `W13` names four different physical gels. The physical well is the
`(pH, well)` pair. This matters enormously for splitting — see below.

## `crop_wells.py` — well detection and circular crop

Isolates the hydrogel well from each photograph so the model sees gel colour,
not plate background or labels.

1. `cv2.imread` → `cvtColor(BGR2HSV)`.
2. `cv2.inRange(hsv, lower_green, upper_green)` → green mask.
3. `GaussianBlur(mask, (5,5), 0)`.
4. Morphological `MORPH_CLOSE` then `MORPH_OPEN`, 7×7 kernel — closes pinholes
   inside the well, then removes speckle outside it.
5. `findContours(..., RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`.
6. Discard `area < 5000`, then keep the contour with `radius > 30` and
   `circularity > 0.6`, where

   ```
   (x, y), radius = cv2.minEnclosingCircle(cnt)
   circularity    = 4 * pi * area / (arcLength^2 + 1e-5)
   ```

   Most cells take the largest such contour; later cells take the left-most,
   for plates where a neighbouring well intrudes into the frame.
7. Filled circle mask at `(x, y, r)`, `bitwise_and`, crop to the bounding box.
8. Write `Preprocessed_Data/.../cropped_<original>.JPG`.

The original notebook held several `lower_green`/`upper_green` variants, because
gel colour shifts with both pH and degradation time and one global range loses
wells at the extremes. `crop_wells.py` keeps all of them in `GREEN_RANGES` and
tries them in order until one yields a valid well, which recovers more images
than any single range: at 0 hr — the worst timepoint — it finds **146/192**
wells against the original **124/192**.

### Measured coverage — 7.1% of images are silently dropped

| | |
|---|---|
| Raw images | 2,112 |
| Successfully cropped | **1,963** |
| **Failed to crop** | **149 (7.1%)** |

The loss is **not random** — it is concentrated where the green mask fits worst:

| Timepoint | 0 hr | 24 | 30 | 48 | 95 | 120 | 168 | 192 | 216 | 264 |
|---|---|---|---|---|---|---|---|---|---|---|
| Failures | **68** | 4 | 1 | 2 | 6 | 6 | 4 | 11 | **24** | **23** |

| pH | 5 | 6 | 7 | 8 |
|---|---|---|---|---|
| Failures | 36 | 25 | **57** | 31 |

68 of 192 images at 0 hr (35%) are lost, plus 47 across 216/264 hr. At 0 hr the
gel is palest and at late timepoints darkest, so a fixed HSV green range misses
both extremes. pH 7 loses the most (57).

This biases every downstream model: the training set under-represents exactly
the fresh and heavily-degraded states. A per-image adaptive threshold (Otsu on
the saturation channel, or a Hough circle on the well rim, which does not depend
on gel colour at all) would recover most of them. **The failures are silent** —
no exception is raised, the image simply never appears in `Preprocessed_Data/`.

## Colour descriptors

Implemented in `supervised/train_supervised.py`; the descriptors the original
notebook explored were:

- *RGB histogram* — 256 bins/channel, sum-normalised.
- *HSV histogram* — H 180 bins, S and V 256 bins, concatenated, normalised.
- *Colour moments* — per-channel mean, std, skewness (9 values).
- *Compact vector* — 8-bin H/S/V histograms (24) + 9 moments = **33 dims** at 64×64.

pH is encoded almost entirely in gel hue and brightness, so colour histograms
carry most of the usable signal at a fraction of raw-pixel dimensionality.

## The split — this notebook's version is broken

The original `feature_extraction.ipynb` (removed) parsed
`well_id = filename.split("_")[-1].split(".")[0]`
→ `"W13"`, and splits those IDs 60/20/20. Two defects:

1. **The W-number is not a well.** It ignores pH, so it treats four different
   physical gels as one unit.
2. **The `Split_Data/` tree on disk was not produced by that cell anyway.**
   Measured directly:

   | | train | val | test |
   |---|---|---|---|
   | images | 1,155 | 370 | 438 |
   | distinct `(pH, well)` | **192** | 170 | 175 |

   All 192 physical wells appear in train; 170 of them also appear in val and
   175 in test. It is effectively a **per-image random split**. Since each well
   is photographed at 11 timepoints, ten photographs of a gel sit in train while
   the eleventh is scored as "unseen" test data.

Every accuracy previously reported from `Split_Data/` is measured under this
leakage.

## `build_split.py` — the corrected split

Groups by the physical `(pH, well)` pair — 192 groups — and splits those,
stratified by pH. Emits a manifest CSV rather than copying ~2 GB of images
again, so the split is reproducible and costs no disk.

```bash
python3 preprocessing/crop_wells.py      # New MLPR Data/ -> Preprocessed_Data/
python3 preprocessing/build_split.py     # writes preprocessing/splits.csv
```

| | train | val | test |
|---|---|---|---|
| **wells** | 116 | 40 | 36 |
| images | 1,177 | 410 | 376 |
| pH 5 / 6 / 7 / 8 | 294/304/281/298 | 106/104/97/103 | 92/95/93/96 |

```
well overlap train&val: 0    train&test: 0    val&test: 0
```

The script **asserts** disjointness, so the failure that produced the old tree
cannot recur silently.

### What the leakage was worth

Identical features, identical model — only the split protocol differs:

| Protocol | RF test accuracy |
|---|---|
| Image-level random split (old) | 0.751 |
| Well-level split (corrected) | **0.729** |
| Inflation | **+0.022** |

On hand-crafted colour histograms the inflation is modest — histograms of a
64×64 colour distribution simply do not carry enough well-specific detail to
memorise. See `transfer_learning/README.md` for the same measurement on learned
embeddings, where a network *can* memorise individual wells.

## Output contract

| Path | Produced by | Consumed by |
|---|---|---|
| `Preprocessed_Data/` | `crop_wells.py` | all model folders |
| `preprocessing/splits.csv` | `build_split.py` | all model folders |
| `Split_Data/` | *(legacy, leaky)* | **do not use** |

## Running

`build_split.py` runs from the repo root. The notebooks' paths are also relative
to the repo root, so start Jupyter there or `os.chdir("..")` in the first cell.
The image directories are `.gitignore`d and must be present locally.
