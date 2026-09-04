"""Why does ResNet18 + RF underperform a colour histogram?

Feature ablation on the identical well-wise split, identical RF, per image.
Everything here is diagnosis, not new modelling.
"""
import os, csv, json, time
import numpy as np, torch, torch.nn as nn, cv2
from torchvision import models, transforms
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score
from collections import defaultdict

SEED=42; CLASSES=[5,6,7,8]
dev=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
rows=list(csv.DictReader(open("preprocessing/splits.csv")))
y=np.array([CLASSES.index(int(r["pH"])) for r in rows])
sp=np.array([r["split"] for r in rows])
hr=np.array([int(r["hour"]) for r in rows], float)
well=np.array([f'{r["pH"]}_{r["well"]}' for r in rows])
tr,va,te = sp=="train", sp=="val", sp=="test"

def rf_eval(X, name, extra=None):
    Xf = X if extra is None else np.hstack([X, extra])
    m=RandomForestClassifier(n_estimators=400,min_samples_leaf=2,random_state=SEED,n_jobs=-1)
    m.fit(Xf[tr],y[tr])
    a_tr=accuracy_score(y[tr],m.predict(Xf[tr])); a_te=accuracy_score(y[te],m.predict(Xf[te]))
    print(f"  {name:52s} dim={Xf.shape[1]:5d}  train {a_tr:.3f}  test {a_te:.3f}")
    return a_te, m, Xf

# ---------- feature sets ----------
print("building feature sets...")
# 1. frozen ImageNet embeddings (cached by lstm/train_lstm.py)
d=np.load("lstm/_embeddings.npz",allow_pickle=True)
emb_map={p:e for p,e in zip(d["paths"],d["E"])}
E_frozen=np.stack([emb_map[r["path"]] for r in rows])

# 2. fine-tuned embeddings from the trained checkpoint
CACHE="transfer_learning/_emb_finetuned.npz"
if os.path.exists(CACHE):
    E_ft=np.load(CACHE)["E"]
else:
    net=models.resnet18(); net.fc=nn.Linear(net.fc.in_features,4)
    net.load_state_dict(torch.load("transfer_learning/resnet18_ph_wellwise.pth",map_location="cpu"))
    net.fc=nn.Identity(); net=net.to(dev).eval()
    tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    out=[]
    with torch.no_grad():
        for i in range(0,len(rows),64):
            b=torch.stack([tf(Image.open(r["path"]).convert("RGB")) for r in rows[i:i+64]])
            out.append(net(b.to(dev)).cpu().numpy())
    E_ft=np.vstack(out); np.savez_compressed(CACHE,E=E_ft)
    print("  encoded fine-tuned embeddings")

# 3. colour histogram (cached by supervised/)
H=np.load("supervised/_features.npz",allow_pickle=True)["X"]

# 4. plain colour moments - the crudest possible colour descriptor
MOM="transfer_learning/_moments.npz"
if os.path.exists(MOM):
    M=np.load(MOM)["M"]
else:
    mm=[]
    for r in rows:
        img=cv2.resize(cv2.imread(r["path"]),(64,64))
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        v=[]
        for arr in (img,hsv):
            # ignore the black circular-mask corners, which are ~21% of the frame
            mask=img.sum(2)>15
            for c in range(3):
                ch=arr[:,:,c][mask]
                v += [ch.mean(), ch.std()]
        mm.append(v)
    M=np.array(mm,np.float32); np.savez_compressed(MOM,M=M)
    print("  computed colour moments")

print("\n--- per-image RandomForest, well-wise split ---")
res={}
res["colour moments (12-d, mask-aware)"] = rf_eval(M,"colour moments (12-d, mask-aware)")[0]
res["HSV histogram 512"]                 = rf_eval(H,"HSV histogram 512")[0]
res["ResNet18 frozen (ImageNet)"]        = rf_eval(E_frozen,"ResNet18 frozen (ImageNet)")[0]
res["ResNet18 fine-tuned"]               = rf_eval(E_ft,"ResNet18 fine-tuned")[0]
res["ResNet18 ft + histogram"]           = rf_eval(np.hstack([E_ft,H]),"ResNet18 ft + histogram")[0]
res["histogram + time"]                  = rf_eval(H,"histogram + time",extra=hr[:,None])[0]
res["ResNet18 ft + time"]                = rf_eval(E_ft,"ResNet18 ft + time",extra=hr[:,None])[0]
a,best_m,best_X = rf_eval(np.hstack([E_ft,H]),"ResNet18 ft + histogram + time",extra=hr[:,None])
res["ResNet18 ft + histogram + time"]=a

# ---------- what do the embeddings actually encode? ----------
print("\n--- what do features encode? (AMI with pH vs with timepoint) ---")
def probe(X,name):
    pipe=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,multi_class="multinomial"))
    pipe.fit(X[tr],y[tr]); a_ph=accuracy_score(y[te],pipe.predict(X[te]))
    hb=np.digitize(hr,np.quantile(hr,[.25,.5,.75]))
    pipe2=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,multi_class="multinomial"))
    pipe2.fit(X[tr],hb[tr]); a_t=accuracy_score(hb[te],pipe2.predict(X[te]))
    print(f"  {name:34s} linear probe -> pH {a_ph:.3f} | -> time-bin {a_t:.3f}")
for X,n in [(H,"HSV histogram"),(E_frozen,"ResNet frozen"),(E_ft,"ResNet fine-tuned")]:
    probe(X,n)

# ---------- per-well aggregation ----------
print("\n--- per-WELL aggregation (average predicted probability over a well) ---")
def per_well(X,name):
    m=RandomForestClassifier(n_estimators=400,min_samples_leaf=2,random_state=SEED,n_jobs=-1).fit(X[tr],y[tr])
    P=m.predict_proba(X[te]); w=well[te]; yt=y[te]
    agg=defaultdict(list); lab={}
    for i,k in enumerate(w): agg[k].append(P[i]); lab[k]=yt[i]
    correct=sum(int(np.mean(v,0).argmax()==lab[k]) for k,v in agg.items())
    print(f"  {name:52s} {correct}/{len(agg)} wells = {correct/len(agg):.3f}")
    return correct/len(agg)
res["_perwell_hist_time"]=per_well(np.hstack([H,hr[:,None]]),"histogram + time, averaged per well")
res["_perwell_fused"]=per_well(best_X,"ResNet ft + histogram + time, averaged per well")
json.dump(res,open("transfer_learning/diagnosis.json","w"),indent=2,default=str)
print("\nwrote transfer_learning/diagnosis.json")
