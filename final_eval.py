"""Final evaluation of every arm, WITHOUT the elapsed-time feature.

Every result here is PER IMAGE: one photograph in, one pH out. Per-well
aggregation (averaging a well's 11 predictions) is deliberately not reported --
it answers an easier question and makes the effective sample size 192 rather
than 1963.

Splitting is still grouped by physical well: StratifiedGroupKFold(5) over the
192 (pH, well) pairs, so no well appears in both train and test. That is the
leakage fix, and it is what makes the per-image numbers below honest -- without
it, 10 photographs of a gel sit in train while the 11th is scored as unseen.
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

SEED=42; CLASSES=[5,6,7,8]
rows=list(csv.DictReader(open("preprocessing/splits.csv")))
y=np.array([CLASSES.index(int(r["pH"])) for r in rows])
well=np.array([f'{r["pH"]}_{r["well"]}' for r in rows])
H=np.load("supervised/_features.npz",allow_pickle=True)["X"]          # HSV histogram 512
M=np.load("transfer_learning/_moments.npz")["M"]                      # colour moments 12
S=np.load("transfer_learning/_stem.npz")["S"]                         # ResNet stem, avg pool 64
B=np.load("transfer_learning/_best_features.npz")["X"]                # stem avg+std + histogram 640
d=np.load("lstm/_embeddings.npz",allow_pickle=True)
idx={p:i for i,p in enumerate(d["paths"])}
E=d["E"][[idx[r["path"]] for r in rows]]                              # ResNet layer4 512

def cv(X, clf_fn, name):
    sgk=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    ia=[];ba=[];f1=[]
    for tr,te in sgk.split(np.zeros(len(y)),y,groups=well):
        m=clf_fn(); m.fit(X[tr],y[tr])
        P=m.predict_proba(X[te]); pred=P.argmax(1)
        ia.append(accuracy_score(y[te],pred)); f1.append(f1_score(y[te],pred,average="macro"))
        ba.append(accuracy_score(y[te]>=2,pred>=2))
    print(f"  {name:38s} acc {np.mean(ia):.3f}+/-{np.std(ia):.3f}  "
          f"acid/alk {np.mean(ba):.3f}  macroF1 {np.mean(f1):.3f}")
    return dict(image=float(np.mean(ia)),image_std=float(np.std(ia)),
                binary=float(np.mean(ba)),macro_f1=float(np.mean(f1)))

RF=lambda: RandomForestClassifier(n_estimators=400,min_samples_leaf=2,random_state=SEED,n_jobs=-1)
VT=lambda est: make_pipeline(VarianceThreshold(1e-8),StandardScaler(),est)

res={}
print("All results PER IMAGE. Grouped 5-fold CV, split by physical well.\n")
print("A. Learners on the HSV histogram (supervised arm)\n")
res["Baseline (majority)"]=cv(H, lambda: DummyClassifier(strategy="most_frequent"), "Baseline (majority)")
res["KNN (k=9)"]      = cv(H, lambda: VT(KNeighborsClassifier(n_neighbors=9,metric="manhattan")), "KNN (k=9)")
res["SVM (RBF, C=10)"]= cv(H, lambda: VT(SVC(C=10,probability=True,random_state=SEED)), "SVM (RBF, C=10)")
res["MLP (64,64)"]    = cv(H, lambda: VT(MLPClassifier(hidden_layer_sizes=(64,64),alpha=0.1,max_iter=1000,
                                          early_stopping=True,random_state=SEED)), "MLP (64,64)")
res["Random Forest"]  = cv(H, RF, "Random Forest")

print("\nB. Feature sets, Random Forest fixed\n")
res["colour moments (12)"]        = cv(M, RF, "colour moments (12)")
res["ResNet layer4 emb (512)"]    = cv(E, RF, "ResNet layer4 emb (512)")
res["ResNet stem (64)"]           = cv(S, RF, "ResNet stem (64)")
res["HSV histogram (512)"]        = res["Random Forest"]
res["stem(avg) + histogram (576)"] = cv(np.hstack([S,H]), RF, "stem(avg) + histogram (576)")
res["stem(avg+std) + histogram (640)"] = cv(B, RF, "stem(avg+std) + histogram (640)  <- best")
json.dump(res,open("final_results_no_time.json","w"),indent=2)
print("\nwrote final_results_no_time.json")
