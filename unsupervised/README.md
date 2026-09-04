# Unsupervised

> **Status: no notebooks yet.** This folder is a placeholder. Nothing in the
> repository currently performs clustering or dimensionality reduction — a
> search across every notebook for k-means, PCA, DBSCAN, t-SNE, UMAP,
> Gaussian mixtures, silhouette scores, hierarchical clustering and
> autoencoders returns no matches. The sections below describe what belongs
> here, not work that has been done.

Stage 2b. Where the other arms ask *"which pH is this?"*, this arm asks
*"what structure does the data have before we impose labels on it?"*

Pipeline position:

```
raw images
  -> preprocessing/data.ipynb          (well detection, circular crop)
  -> preprocessing/feature_extraction  (colour descriptors)
  -> THIS FOLDER                       (clustering / projection, labels held out)
```

## Why this arm is worth having

The supervised results carry an assumption that has not been tested: that pH 5,
6, 7 and 8 are four *separable* groups in colour space. Unsupervised analysis
tests it directly. If features cluster into four clean groups aligning with pH,
the labels are recoverable from colour alone and high accuracy is expected. If
they instead cluster by **timepoint** or by **Hydrolytic vs Enzymatic**, then
degradation state dominates pH in the representation — which would explain why
`classical/` appends elapsed hours as a feature, and would be a finding in its
own right.

Two specific questions the supervised arms cannot answer:

1. **Which classes overlap?** The confusion matrices suggest pH 5/6 are harder
   to separate than 7/8. A 2D projection would show whether those classes are
   genuinely contiguous in feature space or merely under-fitted.
2. **Are there outlier wells?** The crop in `preprocessing/data.ipynb` fails
   silently on some images — the HSV thresholds were re-tuned several times for
   exactly this reason. Clustering on the raw feature vectors would surface
   mis-cropped wells as a small off-manifold group, giving a QA pass over the
   preprocessing stage that nothing currently provides.

## Suggested contents

Inputs are already available — either descriptor from `preprocessing/` (33-dim
histograms + moments, or the 512-dim joint HSV histogram from `classical/`), or
the 512-dim ResNet18 embeddings from `transfer_learning/`. Comparing clustering
on hand-crafted vs learned features is itself informative.

| Step | Method | What it answers |
|---|---|---|
| Projection | PCA to 2–3 components | How much colour variance is linear; explained-variance ratio |
| Projection | t-SNE / UMAP | Whether pH classes form visually distinct neighbourhoods |
| Clustering | k-means, `k = 4` | Do discovered clusters align with pH, or with time/condition? |
| Model selection | Silhouette / elbow over `k = 2..10` | Is 4 the natural number of groups in the data at all? |
| Density | DBSCAN | Mis-cropped and outlier wells as noise points |

Scoring should use the adjusted Rand index or normalised mutual information
between cluster assignments and each of the three candidate groupings — pH,
timepoint, and degradation condition — since which one the clusters track is the
whole question.

Colour the projections by pH, by timepoint and by condition separately. A plot
coloured only by pH cannot reveal that the structure is really about time.

## Running notebooks placed here

Follow the convention used by the other folders: paths relative to the
**repository root**, so start Jupyter there or `os.chdir("..")` in the first
cell. `Preprocessed_Data/` and `Split_Data/` are `.gitignore`d and must exist
locally.
