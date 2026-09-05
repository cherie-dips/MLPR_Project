"""How should the ResNet18 stem be pooled before the Random Forest?

layer_probe.py showed the stem carries the pH signal (0.751) while layer4 does
not (0.629). Global average pooling throws away all spatial layout, which may or
may not matter for a circular crop of uniform gel. This tries richer poolings.
"""
import csv, json
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

SEED=42; C=[5,6,7,8]
dev=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
rows=list(csv.DictReader(open("preprocessing/splits.csv")))
y=np.array([C.index(int(r["pH"])) for r in rows])
well=np.array([f'{r["pH"]}_{r["well"]}' for r in rows])
H=np.load("supervised/_features.npz",allow_pickle=True)["X"]

CACHE="transfer_learning/_stem_variants.npz"
try:
    z=np.load(CACHE); V={k:z[k] for k in z.files}
except FileNotFoundError:
    net=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(dev).eval()
    tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    acc={"gap":[], "gap_std":[], "p2":[], "p3":[]}
    with torch.no_grad():
        for i in range(0,len(rows),64):
            b=torch.stack([tf(Image.open(r["path"]).convert("RGB")) for r in rows[i:i+64]]).to(dev)
            x=net.maxpool(net.relu(net.bn1(net.conv1(b))))          # (B,64,56,56)
            x=x.cpu()   # MPS cannot adaptive-pool 56 -> 3 (non-divisible)
            gap=F.adaptive_avg_pool2d(x,1).flatten(1)
            acc["gap"].append(gap.numpy())
            acc["gap_std"].append(torch.cat([gap, x.std(dim=(2,3))],1).numpy())
            acc["p2"].append(F.adaptive_avg_pool2d(x,2).flatten(1).numpy())
            acc["p3"].append(F.adaptive_avg_pool2d(x,3).flatten(1).numpy())
    V={k:np.vstack(v) for k,v in acc.items()}
    np.savez_compressed(CACHE,**V)

def cv(X,name,n=400,leaf=2):
    a=[]
    for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(np.zeros(len(y)),y,groups=well):
        m=RandomForestClassifier(n_estimators=n,min_samples_leaf=leaf,random_state=SEED,n_jobs=-1).fit(X[tr],y[tr])
        a.append(accuracy_score(y[te],m.predict(X[te])))
    print(f"  {name:44s} dim={X.shape[1]:5d}  {np.mean(a):.3f} +/- {np.std(a):.3f}")
    return float(np.mean(a))

print("A. stem pooling variants, RF alone\n")
r={}
r["stem GAP (64)"]            = cv(V["gap"],     "stem, global avg pool")
r["stem GAP+std (128)"]       = cv(V["gap_std"], "stem, avg + std pool")
r["stem 2x2 (256)"]           = cv(V["p2"],      "stem, 2x2 spatial pool")
r["stem 3x3 (576)"]           = cv(V["p3"],      "stem, 3x3 spatial pool")

print("\nB. fused with the HSV histogram\n")
r["GAP + hist (576)"]         = cv(np.hstack([V["gap"],H]),     "stem GAP + histogram")
r["GAP+std + hist (640)"]     = cv(np.hstack([V["gap_std"],H]), "stem avg+std + histogram")
r["2x2 + hist (768)"]         = cv(np.hstack([V["p2"],H]),      "stem 2x2 + histogram")

print("\nC. RF size on the best fusion\n")
best=np.hstack([V["gap_std"],H])
for n,leaf in [(400,1),(800,1),(800,2),(1200,1)]:
    r[f"RF n={n} leaf={leaf}"]=cv(best,f"stem avg+std + histogram, RF({n}, leaf {leaf})",n,leaf)
json.dump(r,open("transfer_learning/stem_variants.json","w"),indent=2)
print("\nwrote transfer_learning/stem_variants.json")
