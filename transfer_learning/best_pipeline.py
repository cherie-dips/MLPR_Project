"""The project's best model: ResNet18 stem (avg+std pooled) + HSV histogram -> RF.

Arrived at from layer_probe.py, which showed pH accuracy falls monotonically with
ResNet18 depth (stem 0.751 -> layer4 0.629) because ImageNet pretraining builds
colour invariance and colour is the label here. So the stem is used and the rest
of the backbone discarded.

Two refinements over plain global average pooling:
  * avg AND std pooling over the 56x56 stem map. The mean says what colour the
    gel is; the standard deviation says how *uneven* it is, which matters because
    degradation makes gels patchy. Worth +4.1 points on its own (0.752 -> 0.793).
  * concatenating the explicit 8x8x8 HSV histogram, which carries far finer
    colour resolution than 64 learned filters. Worth a further +1.5 (-> 0.808).

Everything is scored PER IMAGE, with the split grouped by physical well so no
well appears in both train and test.
"""
import csv, json, argparse
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2, joblib
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)

SEED=42; CLASSES=[5,6,7,8]
dev=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MEAN,STD=[0.485,0.456,0.406],[0.229,0.224,0.225]

def stem_features(paths, batch=64):
    """ResNet18 stem -> avg and std over the spatial map -> 128-d."""
    net=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(dev).eval()
    tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                           transforms.Normalize(MEAN,STD)])
    out=[]
    with torch.no_grad():
        for i in range(0,len(paths),batch):
            b=torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i+batch]]).to(dev)
            x=net.maxpool(net.relu(net.bn1(net.conv1(b)))).cpu()      # (B,64,56,56)
            out.append(torch.cat([F.adaptive_avg_pool2d(x,1).flatten(1),
                                  x.std(dim=(2,3))],1).numpy())
    return np.vstack(out)

def histogram_features(paths):
    """Joint 8x8x8 HSV histogram, L2-normalised -> 512-d."""
    out=[]
    for p in paths:
        img=cv2.resize(cv2.imread(p),(128,128))
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        h=cv2.calcHist([hsv],[0,1,2],None,[8,8,8],[0,180,0,256,0,256])
        out.append(cv2.normalize(h,h).flatten())
    return np.array(out,np.float32)

def build_features(rows):
    paths=[r["path"] for r in rows]
    return np.hstack([stem_features(paths), histogram_features(paths)])   # 640-d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--save", metavar="PATH", help="fit on all data and save the model")
    a=ap.parse_args()

    rows=list(csv.DictReader(open("preprocessing/splits.csv")))
    y=np.array([CLASSES.index(int(r["pH"])) for r in rows])
    well=np.array([f'{r["pH"]}_{r["well"]}' for r in rows])
    CACHE="transfer_learning/_best_features.npz"
    try:
        X=np.load(CACHE)["X"]; print(f"loaded cached features {X.shape}")
    except FileNotFoundError:
        X=build_features(rows); np.savez_compressed(CACHE,X=X); print(f"built features {X.shape}")

    RF=lambda: RandomForestClassifier(n_estimators=400,min_samples_leaf=2,
                                      random_state=SEED,n_jobs=-1)
    acc=[];f1=[];binacc=[];CM=np.zeros((4,4),int)
    for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(np.zeros(len(y)),y,groups=well):
        m=RF().fit(X[tr],y[tr]); p=m.predict(X[te])
        acc.append(accuracy_score(y[te],p)); f1.append(f1_score(y[te],p,average="macro"))
        binacc.append(accuracy_score(y[te]>=2,p>=2))
        CM+=confusion_matrix(y[te],p,labels=range(4))
    print(f"\nPER-IMAGE, grouped 5-fold CV over 192 wells ({len(y)} images)")
    print(f"  accuracy        {np.mean(acc):.3f} +/- {np.std(acc):.3f}")
    print(f"  macro F1        {np.mean(f1):.3f}")
    print(f"  acid vs alkaline{np.mean(binacc):.3f} +/- {np.std(binacc):.3f}")
    print("\n  confusion (rows=true):\n        pH5  pH6  pH7  pH8")
    for i,c in enumerate(CLASSES): print(f"   pH{c} "+"".join(f"{v:5d}" for v in CM[i]))
    cross=CM[:2,2:].sum()+CM[2:,:2].sum()
    print(f"\n  errors {CM.sum()-np.trace(CM)}, of which {cross} cross the acid/alkaline boundary")

    json.dump(dict(accuracy=float(np.mean(acc)),accuracy_std=float(np.std(acc)),
                   macro_f1=float(np.mean(f1)),binary=float(np.mean(binacc)),
                   cm=CM.tolist(),n_features=int(X.shape[1])),
              open("transfer_learning/best_pipeline.json","w"),indent=2)
    if a.save:
        joblib.dump(RF().fit(X,y), a.save); print(f"\nfitted on all data -> {a.save}")

if __name__=="__main__":
    main()
