# Unsupervised

Stage 2b. Where the other arms ask *"which pH is this?"*, this arm asks
*"what structure does the data actually have, before we impose labels on it?"*

It exists to test an assumption the supervised arms take for granted: that
pH 5, 6, 7 and 8 form four separable groups in colour space.

**They do not.** The dominant structure in this dataset is degradation *time*,
not pH. That single result explains several otherwise puzzling numbers
elsewhere in the project.

```
raw images
  -> preprocessing/data.ipynb      (well detection, circular crop)
  -> preprocessing/build_split.py  (well-wise manifest)
  -> THIS FOLDER                   (PCA / k-means / GMM / DBSCAN, labels held out)
```

## Method

`analyze.py` reuses the 512-dim joint HSV histograms cached by
`supervised/train_supervised.py`. Labels are **never** given to any clustering
algorithm — they are used only afterwards, to score what the clusters found.

241 of the 512 histogram bins are identically zero, so a variance filter reduces
the space to **169 live features**, which are then standardised and projected by
PCA before clustering.

Clusters are scored with adjusted Rand index (ARI) and normalised mutual
information (NMI) against three candidate groupings — pH, degradation timepoint,
and physical well. Whichever the clusters track is the answer.

```bash
python3 supervised/train_supervised.py   # first, to cache features
python3 unsupervised/analyze.py          # ~1 min; writes unsupervised/results.json
```

## Results

### PCA — colour variance is not low-dimensional

| | |
|---|---|
| PC1 | 10.6% |
| PC2 | 7.6% |
| PC1+PC2 | **18.1%** |
| PCs for 90% variance | **94** of 169 |
| PCs for 95% variance | 117 of 169 |

There is no dominant axis. A 2D scatter plot of this data shows less than a
fifth of the variance, so any visual claim of "clean separation" from such a
plot would be an artefact. Needing 94 components for 90% of the variance means
the colour distribution is genuinely high-dimensional — consistent with the
supervised models overfitting badly in 169 dimensions on ~1,200 samples.

### What do the clusters track? — the central result

k-means, k=4, on the 94-component PCA space:

| Grouping | ARI | NMI |
|---|---|---|
| **pH** (the labels we care about) | **+0.083** | 0.097 |
| **Degradation time** (4 bins) | **+0.309** | **0.393** |
| Degradation time (all 11 points) | +0.148 | 0.347 |
| Physical well | +0.001 | 0.052 |

**Clusters track degradation time roughly 3.7× more strongly than pH.** The
result is not an artefact of k-means — it reproduces across algorithms:

| Algorithm (k=4) | ARI vs pH | ARI vs time |
|---|---|---|
| k-means | +0.083 | **+0.309** |
| Gaussian Mixture | +0.081 | **+0.344** |
| Agglomerative (ward) | +0.084 | **+0.314** |

The near-zero ARI against *well* identity (+0.001) is reassuring: images do not
cluster by which individual gel they came from, so the crops are not encoding
per-well artefacts like plate position or lighting.

### Is 4 even the right number of groups?

| k | 2 | 3 | **4** | 5 | 6 | **7** | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| silhouette | 0.113 | 0.120 | 0.128 | 0.126 | 0.105 | **0.129** | 0.116 | 0.118 | 0.120 |

Silhouette peaks at k=7 (0.129), with k=4 essentially tied (0.128) — but **every
value is very low**. Anything below ~0.25 indicates no substantial cluster
structure at all: the data is closer to one continuous manifold than to four
discrete blobs. That is chemically sensible. pH is a continuum and the gel
response is gradual, so there is no reason for four crisp clusters to exist.

### Where pH structure does exist

k-means cluster membership by true pH:

| | c0 | c1 | c2 | c3 |
|---|---|---|---|---|
| **pH5** | **342** | 43 | 0 | 107 |
| **pH6** | **270** | 88 | 6 | 139 |
| **pH7** | 117 | **221** | 40 | 93 |
| **pH8** | 133 | **240** | 48 | 76 |

The weak signal that *is* present is **acidic vs alkaline**, not four-way. pH 5
and 6 concentrate in c0 (612 of 906 images); pH 7 and 8 concentrate in c1 (461
of 1,015). Cluster c2 is nearly empty (94 images, almost all pH 7/8) and c3 is
mixed.

This mirrors the supervised confusion matrices exactly — adjacent pH pairs
(5↔6, 7↔8) are where the errors concentrate, while cross-boundary confusion is
rare.

### DBSCAN — a QA pass over preprocessing

With `eps` at the 90th percentile of the 5-NN distance, DBSCAN finds **1 cluster
and 138 noise points (7.0%)**. Finding a single cluster is itself the finding:
the data has no density-separated groups, which agrees with the silhouette
scores.

The 7.0% flagged as noise is suggestively close to the 7.1% of images that fail
to crop — but they are different populations (the 149 crop failures never reach
this stage at all), so the resemblance is a coincidence, not a link.

## What this means for the rest of the project

1. **It explains the `RF + time` result.** Adding elapsed hours lifts supervised
   test accuracy 0.729 → 0.769. Time is the dominant axis of colour variation,
   so telling the model where on the degradation curve a sample sits removes the
   largest confound.
2. **It sets expectations.** With ARI 0.08 between colour clusters and pH, no
   model should be expected to reach very high per-image pH accuracy from colour
   alone. The supervised ceiling around 0.73–0.77 is consistent with the
   structure genuinely present in the features.
3. **It argues for the sequence models.** If time dominates single-image colour,
   then modelling the *trajectory* — how a specific well's colour evolves — is
   the principled response rather than an incremental extra. See `lstm/`.
4. **It reframes the target.** A binary acidic/alkaline (pH 5–6 vs 7–8)
   formulation matches both the cluster structure and the clinical question
   (healthy skin pH 4–6, chronic wound pH 7–8) far better than 4-way
   classification.

## Caveats

- All of the above uses hand-crafted HSV histograms. Clustering the learned
  ResNet18 embeddings from `transfer_learning/` may find different structure and
  is not yet done.
- ARI against "time in 4 bins" is compared with k=4 clustering, which is the
  fair matched comparison; the all-11-timepoints ARI (+0.148) is lower simply
  because 4 clusters cannot express 11 groups. NMI (0.347) is the better
  read there, and is still well above the pH figure.
- Silhouette is computed in the 94-D PCA space, not the raw 169-D space.

## Running

Paths are relative to the **repository root**. Requires
`supervised/_features.npz` (produced by `supervised/train_supervised.py`) and
`preprocessing/splits.csv`.
