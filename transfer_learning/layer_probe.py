"""Which ResNet18 stage carries the pH signal?

Hypothesis: ImageNet pretraining deliberately builds colour invariance (its
augmentation includes colour jitter), because object identity should not depend
on hue. Here hue IS the label. If that is the mechanism, shallow stages -- which
still encode colour -- should outperform the final 512-d embedding that
transfer_learning.ipynb uses.
"""
import csv, json
import numpy as np, torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

SEED=42; CLASSES=[5,6,7,8]
dev=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
rows=list(csv.DictReader(open("preprocessing/splits.csv")))
y=np.array([CLASSES.index(int(r["pH"])) for r in rows])
well=np.array([f'{r["pH"]}_{r["well"]}' for r in rows])

net=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(dev).eval()
tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

stages={
 "stem (conv1+pool, 64ch)": lambda x: net.maxpool(net.relu(net.bn1(net.conv1(x)))),
}
def upto(n):
    def f(x):
        x=net.maxpool(net.relu(net.bn1(net.conv1(x))))
        for i,l in enumerate([net.layer1,net.layer2,net.layer3,net.layer4]):
            x=l(x)
            if i==n: return x
    return f
for i,name in enumerate(["layer1 (64ch)","layer2 (128ch)","layer3 (256ch)","layer4 (512ch, used by the notebook)"]):
    stages[name]=upto(i)

feats={k:[] for k in stages}
with torch.no_grad():
    for i in range(0,len(rows),64):
        b=torch.stack([tf(Image.open(r["path"]).convert("RGB")) for r in rows[i:i+64]]).to(dev)
        for k,fn in stages.items():
            feats[k].append(torch.nn.functional.adaptive_avg_pool2d(fn(b),1).flatten(1).cpu().numpy())
feats={k:np.vstack(v) for k,v in feats.items()}

print("grouped 5-fold CV, RF on globally-average-pooled features of each stage\n")
res={}
for k,X in feats.items():
    sgk=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED); acc=[]
    for tr,te in sgk.split(np.zeros(len(y)),y,groups=well):
        m=RandomForestClassifier(n_estimators=400,min_samples_leaf=2,random_state=SEED,n_jobs=-1).fit(X[tr],y[tr])
        acc.append(accuracy_score(y[te],m.predict(X[te])))
    res[k]=dict(dim=int(X.shape[1]),acc=float(np.mean(acc)),std=float(np.std(acc)))
    print(f"  {k:40s} dim={X.shape[1]:4d}  per-image acc {np.mean(acc):.3f}+/-{np.std(acc):.3f}")
json.dump(res,open("transfer_learning/layer_probe.json","w"),indent=2)
print("\nwrote transfer_learning/layer_probe.json")
