"""Improvement experiments, evaluated by grouped 5-fold CV over all 192 wells.

A single 36-well test set gives a +/-13 point confidence interval, which is too
loose to rank models. StratifiedGroupKFold grouped by physical well uses every
well as test exactly once, so the estimate is far tighter and still leakage-free.

Only training-free feature extractors are used (colour features, frozen
ImageNet embeddings). Fine-tuned embeddings cannot appear here: the checkpoint
saw the train split, so reusing it inside CV would leak.
"""
import csv, json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from collections import defaultdict

SEED=42; CLASSES=[5,6,7,8]
rows=list(csv.DictReader(open("preprocessing/splits.csv")))
y=np.array([CLASSES.index(int(r["pH"])) for r in rows])
hr=np.array([int(r["hour"]) for r in rows],float)
well=np.array([f'{r["pH"]}_{r["well"]}' for r in rows])

H=np.load("supervised/_features.npz",allow_pickle=True)["X"]
M=np.load("transfer_learning/_moments.npz")["M"]
d=np.load("lstm/_embeddings.npz",allow_pickle=True)
E=np.stack([{p:e for p,e in zip(d["paths"],d["E"])}[r["path"]] for r in rows])

def run(build, name):
    """build(train_idx) -> (X, fitted-transform) so any PCA is fit on train only."""
    sgk=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    img_acc=[]; well_acc=[]; bin_acc=[]
    for tr,te in sgk.split(np.zeros(len(y)),y,groups=well):
        X=build(tr)
        m=RandomForestClassifier(n_estimators=400,min_samples_leaf=2,random_state=SEED,n_jobs=-1)
        m.fit(X[tr],y[tr])
        P=m.predict_proba(X[te]); pred=P.argmax(1)
        img_acc.append(accuracy_score(y[te],pred))
        # acid (pH5,6 -> class 0,1) vs alkaline (pH7,8 -> class 2,3)
        bin_acc.append(accuracy_score(y[te]>=2, pred>=2))
        agg=defaultdict(list); lab={}
        for i,k in zip(te,well[te]): pass
        agg=defaultdict(list)
        for j,i in enumerate(te):
            agg[well[i]].append(P[j]); lab[well[i]]=y[i]
        well_acc.append(np.mean([np.mean(v,0).argmax()==lab[k] for k,v in agg.items()]))
    f=lambda a: f"{np.mean(a):.3f}+/-{np.std(a):.3f}"
    print(f"  {name:44s} img {f(img_acc)}   well {f(well_acc)}   acid/alk {f(bin_acc)}")
    return dict(image=float(np.mean(img_acc)), image_std=float(np.std(img_acc)),
                well=float(np.mean(well_acc)), well_std=float(np.std(well_acc)),
                binary=float(np.mean(bin_acc)), binary_std=float(np.std(bin_acc)))

T=hr[:,None]
def pca_fuse(k):
    def b(tr):
        sc=StandardScaler().fit(E[tr]); p=PCA(n_components=k,random_state=SEED).fit(sc.transform(E[tr]))
        return np.hstack([H, p.transform(sc.transform(E)), T])
    return b

print("grouped 5-fold CV over all 192 wells (1963 images)\n")
res={}
res["colour moments (12)"]        = run(lambda tr: M, "colour moments (12)")
res["HSV histogram (512)"]        = run(lambda tr: H, "HSV histogram (512)")
res["frozen ImageNet emb (512)"]  = run(lambda tr: E, "frozen ImageNet emb (512)")
res["histogram + time"]           = run(lambda tr: np.hstack([H,T]), "histogram + time")
res["histogram + moments + time"] = run(lambda tr: np.hstack([H,M,T]), "histogram + moments + time")
res["emb + histogram + time (raw fuse)"] = run(lambda tr: np.hstack([E,H,T]), "emb + histogram + time (raw fuse)")
res["histogram + PCA32(emb) + time"]     = run(pca_fuse(32), "histogram + PCA32(emb) + time")
json.dump(res,open("transfer_learning/improve.json","w"),indent=2)
print("\nwrote transfer_learning/improve.json")
