"""Unsupervised structure analysis of the hydrogel colour features.

The supervised arms assume pH 5/6/7/8 are four separable groups in colour space.
This tests that assumption directly, with labels held out and only used to score
the clusters afterwards.

The central question: do the clusters track pH, or do they track degradation
TIME? If time dominates, a per-image pH classifier is solving the wrong problem
and the RF+time result in supervised/ is explained.
"""
import os, csv, json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)

FEATS="supervised/_features.npz"; SEED=42
OUT="unsupervised/results.json"

def main():
    d=np.load(FEATS, allow_pickle=True)
    X, y_ph, split, hour = d["X"], d["y"], d["split"], d["hour"]
    rows=list(csv.DictReader(open("preprocessing/splits.csv")))
    well=np.array([f'{r["pH"]}_{r["well"]}' for r in rows])
    assert len(well)==len(X)
    print(f"{X.shape[0]} images, {X.shape[1]} raw features")

    # 241/512 histogram bins are identically zero; drop them, then standardise.
    prep=make_pipeline(VarianceThreshold(1e-8), StandardScaler())
    Xs=prep.fit_transform(X)
    print(f"after variance filter: {Xs.shape[1]} features")

    res={}

    # ---------- PCA ----------
    pca=PCA(random_state=SEED).fit(Xs)
    cum=np.cumsum(pca.explained_variance_ratio_)
    n90=int(np.searchsorted(cum,0.90)+1); n95=int(np.searchsorted(cum,0.95)+1)
    res["pca"]=dict(pc1=float(pca.explained_variance_ratio_[0]),
                    pc2=float(pca.explained_variance_ratio_[1]),
                    pc1_2=float(cum[1]), n_components_90=n90, n_components_95=n95,
                    n_features=int(Xs.shape[1]))
    print(f"\nPCA: PC1={pca.explained_variance_ratio_[0]:.3f} PC2={pca.explained_variance_ratio_[1]:.3f} "
          f"PC1+2={cum[1]:.3f}; {n90} PCs for 90%, {n95} for 95%")
    Xp=PCA(n_components=n90, random_state=SEED).fit_transform(Xs)

    # ---------- what do clusters track? ----------
    # Three candidate groupings. Time is binned into 4 so k=4 is a fair comparison.
    hour_bin=np.digitize(hour, np.quantile(hour,[0.25,0.5,0.75]))
    groupings={"pH":y_ph, "timepoint(4 bins)":hour_bin, "timepoint(all 11)":hour, "well":well}

    print("\n--- KMeans k=4 on PCA space: which grouping do clusters follow? ---")
    km=KMeans(n_clusters=4, n_init=20, random_state=SEED).fit(Xp)
    kres={}
    for name,g in groupings.items():
        ari=adjusted_rand_score(g,km.labels_); nmi=normalized_mutual_info_score(g,km.labels_)
        kres[name]=dict(ARI=float(ari), NMI=float(nmi))
        print(f"  vs {name:20s} ARI={ari:+.3f}  NMI={nmi:.3f}")
    res["kmeans_k4"]=kres

    # ---------- how many groups are actually there? ----------
    print("\n--- silhouette over k (is 4 the natural number of groups?) ---")
    sil={}
    for k in range(2,11):
        lab=KMeans(n_clusters=k,n_init=10,random_state=SEED).fit_predict(Xp)
        s=float(silhouette_score(Xp,lab)); sil[k]=s
        print(f"  k={k:2d}  silhouette={s:.3f}")
    best_k=max(sil,key=sil.get)
    res["silhouette"]=sil; res["best_k"]=int(best_k)
    print(f"  -> best k = {best_k} (silhouette {sil[best_k]:.3f})")

    # ---------- other clusterers, scored against pH ----------
    print("\n--- other algorithms at k=4, ARI vs pH / vs time ---")
    others={}
    for name,lab in [
        ("GaussianMixture", GaussianMixture(n_components=4,random_state=SEED,n_init=5).fit_predict(Xp)),
        ("Agglomerative(ward)", AgglomerativeClustering(n_clusters=4).fit_predict(Xp)),
    ]:
        a_ph=adjusted_rand_score(y_ph,lab); a_t=adjusted_rand_score(hour_bin,lab)
        others[name]=dict(ARI_pH=float(a_ph), ARI_time=float(a_t))
        print(f"  {name:22s} ARI(pH)={a_ph:+.3f}  ARI(time)={a_t:+.3f}")
    res["other_clusterers"]=others

    # ---------- DBSCAN as a QA pass over preprocessing ----------
    print("\n--- DBSCAN outlier scan (mis-cropped wells) ---")
    from sklearn.neighbors import NearestNeighbors
    kdist=np.sort(NearestNeighbors(n_neighbors=5).fit(Xp).kneighbors(Xp)[0][:,-1])
    eps=float(np.quantile(kdist,0.90))
    db=DBSCAN(eps=eps,min_samples=5).fit(Xp)
    n_noise=int((db.labels_==-1).sum()); n_cl=len(set(db.labels_))-(1 if -1 in db.labels_ else 0)
    res["dbscan"]=dict(eps=eps, clusters=n_cl, noise=n_noise, noise_frac=n_noise/len(Xp))
    print(f"  eps={eps:.2f} -> {n_cl} clusters, {n_noise} noise points ({100*n_noise/len(Xp):.1f}%)")

    # ---------- per-pH separability in the unsupervised space ----------
    print("\n--- cluster/pH contingency (rows=pH, cols=kmeans cluster) ---")
    cont=np.zeros((4,4),int)
    for i,p in enumerate([5,6,7,8]):
        for c in range(4): cont[i,c]=int(((y_ph==p)&(km.labels_==c)).sum())
    res["contingency_pH_x_cluster"]=cont.tolist()
    print("        c0    c1    c2    c3")
    for i,p in enumerate([5,6,7,8]): print(f"  pH{p} " + "".join(f"{v:6d}" for v in cont[i]))

    json.dump(res, open(OUT,"w"), indent=2)
    print(f"\nwrote {OUT}")

if __name__=="__main__":
    main()
