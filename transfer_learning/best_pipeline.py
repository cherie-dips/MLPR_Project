"""The corrected transfer-learning pipeline, given the layer probe.

layer_probe.py shows pH accuracy falls monotonically with ResNet depth
(stem 0.751 -> layer4 0.629), so the notebook's choice of the final 512-d
embedding is the single worst option available from this backbone. Here the
shallow stage is used instead, fused with the colour histogram and elapsed
time, and predictions are averaged over each well.
"""
import csv, json
import numpy as np, torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import defaultdict

SEED=42; CLASSES=[5,6,7,8]
dev=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
rows=list(csv.DictReader(open("preprocessing/splits.csv")))
y=np.array([CLASSES.index(int(r["pH"])) for r in rows])
hr=np.array([int(r["hour"]) for r in rows],float)[:,None]
well=np.array([f'{r["pH"]}_{r["well"]}' for r in rows])
H=np.load("supervised/_features.npz",allow_pickle=True)["X"]

CACHE="transfer_learning/_stem.npz"
try:
    S=np.load(CACHE)["S"]
except FileNotFoundError:
    net=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(dev).eval()
    tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    out=[]
    with torch.no_grad():
        for i in range(0,len(rows),64):
            b=torch.stack([tf(Image.open(r["path"]).convert("RGB")) for r in rows[i:i+64]]).to(dev)
            x=net.maxpool(net.relu(net.bn1(net.conv1(b))))
            out.append(torch.nn.functional.adaptive_avg_pool2d(x,1).flatten(1).cpu().numpy())
    S=np.vstack(out); np.savez_compressed(CACHE,S=S)

def cv(X,name,show=False):
    sgk=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    ia=[];wa=[];ba=[];CM=np.zeros((4,4),int)
    for tr,te in sgk.split(np.zeros(len(y)),y,groups=well):
        m=RandomForestClassifier(n_estimators=400,min_samples_leaf=2,random_state=SEED,n_jobs=-1).fit(X[tr],y[tr])
        P=m.predict_proba(X[te]); pred=P.argmax(1)
        ia.append(accuracy_score(y[te],pred)); ba.append(accuracy_score(y[te]>=2,pred>=2))
        agg=defaultdict(list); lab={}
        for j,i in enumerate(te): agg[well[i]].append(P[j]); lab[well[i]]=y[i]
        wp={k:np.mean(v,0).argmax() for k,v in agg.items()}
        wa.append(np.mean([wp[k]==lab[k] for k in wp]))
        for k in wp: CM[lab[k],wp[k]]+=1
    f=lambda a:f"{np.mean(a):.3f}+/-{np.std(a):.3f}"
    print(f"  {name:46s} img {f(ia)}  well {f(wa)}  acid/alk {f(ba)}")
    if show:
        print("\n  per-well confusion over all 192 wells (rows=true):")
        print("        pH5  pH6  pH7  pH8")
        for i,p in enumerate(CLASSES): print(f"   pH{p} " + "".join(f"{v:5d}" for v in CM[i]))
    return dict(image=float(np.mean(ia)),image_std=float(np.std(ia)),
                well=float(np.mean(wa)),well_std=float(np.std(wa)),
                binary=float(np.mean(ba)),binary_std=float(np.std(ba)),cm=CM.tolist())

print("grouped 5-fold CV over all 192 wells\n")
res={}
res["layer4 emb (notebook's choice)"] = cv(np.load("lstm/_embeddings.npz",allow_pickle=True)["E"][
    [list(np.load("lstm/_embeddings.npz",allow_pickle=True)["paths"]).index(r["path"]) for r in rows]],
    "layer4 emb (notebook's choice)")
res["stem features (64)"]            = cv(S, "stem features (64)")
res["stem + time"]                   = cv(np.hstack([S,hr]), "stem + time")
res["histogram + time"]              = cv(np.hstack([H,hr]), "histogram + time")
res["stem + histogram + time"]       = cv(np.hstack([S,H,hr]), "stem + histogram + time", show=True)
json.dump(res,open("transfer_learning/best_pipeline.json","w"),indent=2)
print("\nwrote transfer_learning/best_pipeline.json")
