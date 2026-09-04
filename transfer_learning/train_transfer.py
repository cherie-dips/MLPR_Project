"""ResNet18 transfer learning on the leakage-free well-wise split.

Fixes relative to transfer_learning.ipynb:
  1. Split.       ImageFolder over Split_Data/ inherited the leaky split (all 192
                  physical wells appear in train, 170/175 also in val/test). This
                  reads the well-wise manifest instead.
  2. Model sel.   The notebook trained a fixed 15 epochs and never used val to
                  choose a checkpoint, so the reported model was simply the last
                  one. Here val accuracy selects the checkpoint, with early
                  stopping, and test is evaluated once at the end.
  3. Overfitting. No augmentation and lr=1e-3 on all 11M pretrained weights.
                  Here: mild geometric augmentation, discriminative learning
                  rates (backbone 1e-4 / head 1e-3), weight decay, cosine decay.
  4. Diagnostics. Per-epoch train and val curves are recorded so over/underfit
                  is visible rather than assumed.
"""
import os, csv, json, time, copy
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split

MANIFEST="preprocessing/splits.csv"; SEED=42; EPOCHS=40; PATIENCE=8; BATCH=32
OUT="transfer_learning"; CLASSES=[5,6,7,8]
torch.manual_seed(SEED); np.random.seed(SEED)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MEAN,STD=[0.485,0.456,0.406],[0.229,0.224,0.225]

# Colour IS the signal here, so augmentation stays geometric plus only a mild
# brightness/contrast jitter for illumination variation. No hue/saturation jitter.
train_tf = transforms.Compose([
    transforms.Resize((224,224)), transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(), transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.10, contrast=0.10),
    transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
eval_tf = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(MEAN,STD)])

class Wells(Dataset):
    def __init__(self, rows, tf):
        self.rows=rows; self.tf=tf
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        img=Image.open(r["path"]).convert("RGB")
        return self.tf(img), CLASSES.index(int(r["pH"]))

def loaders():
    rows=list(csv.DictReader(open(MANIFEST)))
    d={s:[r for r in rows if r["split"]==s] for s in ("train","val","test")}
    return (DataLoader(Wells(d["train"],train_tf),BATCH,shuffle=True,num_workers=4),
            DataLoader(Wells(d["train"],eval_tf), BATCH,num_workers=4),
            DataLoader(Wells(d["val"],  eval_tf), BATCH,num_workers=4),
            DataLoader(Wells(d["test"], eval_tf), BATCH,num_workers=4))

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); P=[];Y=[];L=[]
    for x,y in loader:
        out=model(x.to(dev)); L.append(out.cpu())
        P.append(out.argmax(1).cpu()); Y.append(y)
    return torch.cat(P).numpy(), torch.cat(Y).numpy(), torch.cat(L).numpy()

def main():
    tr_aug, tr_eval, va, te = loaders()
    print(f"device={dev}  train={len(tr_aug.dataset)} val={len(va.dataset)} test={len(te.dataset)}")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 4); model=model.to(dev)
    head=list(map(id,model.fc.parameters()))
    body=[p for p in model.parameters() if id(p) not in head]
    opt=torch.optim.AdamW([{"params":body,"lr":1e-4},{"params":model.fc.parameters(),"lr":1e-3}],
                          weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    crit=nn.CrossEntropyLoss(label_smoothing=0.05)

    hist=[]; best=(-1,None,-1); bad=0; t0=time.time()
    for ep in range(1,EPOCHS+1):
        model.train(); tot=0; corr=0; ls=0.0
        for x,y in tr_aug:
            x,y=x.to(dev),y.to(dev)
            opt.zero_grad(); out=model(x); loss=crit(out,y)
            loss.backward(); opt.step()
            ls+=loss.item()*y.size(0); tot+=y.size(0); corr+=(out.argmax(1)==y).sum().item()
        sched.step()
        tr_acc=corr/tot
        pv,yv,_=evaluate(model,va); va_acc=accuracy_score(yv,pv)
        hist.append(dict(epoch=ep,train_acc=tr_acc,val_acc=va_acc,train_loss=ls/tot))
        flag=""
        if va_acc>best[0]:
            best=(va_acc,copy.deepcopy(model.state_dict()),ep); bad=0; flag=" *best"
        else:
            bad+=1
        print(f"  ep{ep:02d} loss {ls/tot:.3f} train {tr_acc:.3f} val {va_acc:.3f}{flag}")
        if bad>=PATIENCE:
            print(f"  early stop (no val gain in {PATIENCE} epochs)"); break
    print(f"training {time.time()-t0:.0f}s; best val {best[0]:.3f} @ epoch {best[2]}")
    model.load_state_dict(best[1])
    torch.save(model.state_dict(), f"{OUT}/resnet18_ph_wellwise.pth")

    res={}
    # --- Model A: fine-tuned ResNet18 ---
    pt,yt,logits=evaluate(model,te)
    pr,yr,_=evaluate(model,tr_eval); pv,yv,_=evaluate(model,va)
    prob=torch.softmax(torch.tensor(logits),1).numpy()
    res["ResNet18 (fine-tuned)"]=dict(
        train=accuracy_score(yr,pr), val=accuracy_score(yv,pv), test=accuracy_score(yt,pt),
        test_f1=f1_score(yt,pt,average="macro"),
        auc_macro=roc_auc_score(yt,prob,multi_class="ovr",average="macro"),
        cm=confusion_matrix(yt,pt).tolist(), best_epoch=best[2],
        report=classification_report(yt,pt,target_names=[f"pH{c}" for c in CLASSES],zero_division=0))

    # --- Model B: fine-tuned embeddings -> Random Forest (the project's headline model) ---
    emb=copy.deepcopy(model); emb.fc=nn.Identity(); emb=emb.to(dev).eval()
    def feats(loader):
        F=[];Y=[]
        with torch.no_grad():
            for x,y in loader:
                F.append(emb(x.to(dev)).cpu().numpy()); Y.append(y.numpy())
        return np.vstack(F), np.concatenate(Y)
    Ftr,Ytr=feats(tr_eval); Fva,Yva=feats(va); Fte,Yte=feats(te)
    rf=RandomForestClassifier(n_estimators=400,min_samples_leaf=2,random_state=SEED,n_jobs=-1).fit(Ftr,Ytr)
    pte=rf.predict(Fte); prob_rf=rf.predict_proba(Fte)
    res["ResNet18 + RandomForest"]=dict(
        train=accuracy_score(Ytr,rf.predict(Ftr)), val=accuracy_score(Yva,rf.predict(Fva)),
        test=accuracy_score(Yte,pte), test_f1=f1_score(Yte,pte,average="macro"),
        auc_macro=roc_auc_score(Yte,prob_rf,multi_class="ovr",average="macro"),
        cm=confusion_matrix(Yte,pte).tolist(),
        report=classification_report(Yte,pte,target_names=[f"pH{c}" for c in CLASSES],zero_division=0))

    # --- Leakage check: frozen ImageNet embeddings, well-level vs image-level split ---
    frozen=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1); frozen.fc=nn.Identity()
    frozen=frozen.to(dev).eval()
    def ffeats(loader):
        F=[];Y=[]
        with torch.no_grad():
            for x,y in loader:
                F.append(frozen(x.to(dev)).cpu().numpy()); Y.append(y.numpy())
        return np.vstack(F), np.concatenate(Y)
    Gtr,Gytr=ffeats(tr_eval); Gva,Gyva=ffeats(va); Gte,Gyte=ffeats(te)
    rf_w=RandomForestClassifier(n_estimators=400,random_state=SEED,n_jobs=-1).fit(Gtr,Gytr)
    honest=accuracy_score(Gyte,rf_w.predict(Gte))
    Gall=np.vstack([Gtr,Gva,Gte]); Gyall=np.concatenate([Gytr,Gyva,Gyte])
    a,b,ya_,yb_=train_test_split(Gall,Gyall,test_size=0.2,random_state=SEED,stratify=Gyall)
    leaky=accuracy_score(yb_,RandomForestClassifier(n_estimators=400,random_state=SEED,n_jobs=-1).fit(a,ya_).predict(b))
    res["_leakage_check"]=dict(note="frozen ImageNet ResNet18 embeddings + RF; only the split protocol differs",
                               image_level_split=leaky, well_level_split=honest, inflation=leaky-honest)
    res["_history"]=hist

    print("\n"+"="*78)
    print(f"{'model':28s} {'train':>7s} {'val':>7s} {'test':>7s} {'macroF1':>8s} {'AUC':>6s}")
    print("-"*78)
    for k,v in res.items():
        if k.startswith("_"): continue
        print(f"{k:28s} {v['train']:7.3f} {v['val']:7.3f} {v['test']:7.3f} {v['test_f1']:8.3f} {v['auc_macro']:6.3f}")
    lc=res["_leakage_check"]
    print(f"\nLEAKAGE CHECK (frozen embeddings + RF)")
    print(f"  image-level split {lc['image_level_split']:.3f} | well-level {lc['well_level_split']:.3f} | inflation {lc['inflation']:+.3f}")
    json.dump(res, open(f"{OUT}/results.json","w"), indent=2, default=str)
    print(f"\nwrote {OUT}/results.json")

if __name__=="__main__":
    main()
