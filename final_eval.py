"""Final evaluation of every arm, WITHOUT the elapsed-time feature.

One protocol for everything: StratifiedGroupKFold(5) grouped by physical well,
so each of the 192 wells is tested exactly once and no well is ever in both
train and test. Reports per-image, per-well (probabilities averaged over a
well's timepoints) and acid-vs-alkaline accuracy.
"""
import csv, json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict

SEED=42; CLASSES=[5,6,7,8]
rows=list(csv.DictReader(open("preprocessing/splits.csv")))
y=np.array([CLASSES.index(int(r["pH"])) for r in rows])
well=np.array([f'{r["pH"]}_{r["well"]}' for r in rows])
H=np.load("supervised/_features.npz",allow_pickle=True)["X"]          # HSV histogram 512
M=np.load("transfer_learning/_moments.npz")["M"]                      # colour moments 12
S=np.load("transfer_learning/_stem.npz")["S"]                         # ResNet stem 64
d=np.load("lstm/_embeddings.npz",allow_pickle=True)
idx={p:i for i,p in enumerate(d["paths"])}
E=d["E"][[idx[r["path"]] for r in rows]]                              # ResNet layer4 512

def cv(X, clf_fn, name):
    sgk=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    ia=[];wa=[];ba=[];f1=[]
    for tr,te in sgk.split(np.zeros(len(y)),y,groups=well):
        m=clf_fn(); m.fit(X[tr],y[tr])
        P=m.predict_proba(X[te]); pred=P.argmax(1)
        ia.append(accuracy_score(y[te],pred)); f1.append(f1_score(y[te],pred,average="macro"))
        ba.append(accuracy_score(y[te]>=2,pred>=2))
        agg=defaultdict(list); lab={}
        for j,i in enumerate(te): agg[well[i]].append(P[j]); lab[well[i]]=y[i]
        wa.append(np.mean([np.mean(v,0).argmax()==lab[k] for k,v in agg.items()]))
    print(f"  {name:38s} img {np.mean(ia):.3f}+/-{np.std(ia):.3f}  "
          f"well {np.mean(wa):.3f}+/-{np.std(wa):.3f}  acid/alk {np.mean(ba):.3f}  F1 {np.mean(f1):.3f}")
    return dict(image=float(np.mean(ia)),image_std=float(np.std(ia)),
                well=float(np.mean(wa)),well_std=float(np.std(wa)),
                binary=float(np.mean(ba)),macro_f1=float(np.mean(f1)))

RF=lambda: RandomForestClassifier(n_estimators=400,min_samples_leaf=2,random_state=SEED,n_jobs=-1)
VT=lambda est: make_pipeline(VarianceThreshold(1e-8),StandardScaler(),est)

res={}
print("A. Learners on the HSV histogram (supervised arm), no time\n")
res["Baseline (majority)"]=cv(H, lambda: DummyClassifier(strategy="most_frequent"), "Baseline (majority)")
res["KNN (k=9)"]      = cv(H, lambda: VT(KNeighborsClassifier(n_neighbors=9,metric="manhattan")), "KNN (k=9)")
res["SVM (RBF, C=10)"]= cv(H, lambda: VT(SVC(C=10,probability=True,random_state=SEED)), "SVM (RBF, C=10)")
res["MLP (64,64)"]    = cv(H, lambda: VT(MLPClassifier(hidden_layer_sizes=(64,64),alpha=0.1,max_iter=1000,
                                          early_stopping=True,random_state=SEED)), "MLP (64,64)")
res["Random Forest"]  = cv(H, RF, "Random Forest")

print("\nB. Feature sets, Random Forest fixed, no time\n")
res["colour moments (12)"]        = cv(M, RF, "colour moments (12)")
res["ResNet layer4 emb (512)"]    = cv(E, RF, "ResNet layer4 emb (512)")
res["ResNet stem (64)"]           = cv(S, RF, "ResNet stem (64)")
res["HSV histogram (512)"]        = res["Random Forest"]
res["stem + histogram (576)"]     = cv(np.hstack([S,H]), RF, "stem + histogram (576)")
json.dump(res,open("final_results_no_time.json","w"),indent=2)
print("\nwrote final_results_no_time.json")
