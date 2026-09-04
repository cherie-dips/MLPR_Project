"""ResNet18 + LSTM over each well's degradation trajectory.

Fixes relative to lstm_model.ipynb:
  1. Paths.    It listed 'Split_Data1/train' but joined against 'Final_Data' --
               neither exists. Because rows were kept only `if len(seq)==11`,
               a wrong root yielded ZERO sequences silently instead of erroring.
               This reads the well-wise manifest.
  2. Missing.  149 of 2112 images (7.1%) failed to crop, concentrated at 0 hr
               and the late timepoints, so requiring exactly 11 frames discards
               most wells. Sequences here are variable-length with
               pack_padded_sequence, and no well is dropped.
  3. Split.    A well is one sample, so the split is well-wise by construction
               (inherited from the manifest).
  4. Aug.      The notebook applied ColorJitter(0.2, 0.2) to a task whose entire
               signal is colour. Encoding is done once without jitter.
  5. Ablation. A last-timepoint-only baseline is run, so we can tell whether the
               sequence earns its keep.

The encoder is frozen, so embeddings are computed once and cached rather than
re-run every epoch.
"""
import os, csv, json, time, copy
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import models, transforms
from PIL import Image
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

MANIFEST="preprocessing/splits.csv"; EMB="lstm/_embeddings.npz"; SEED=42
CLASSES=[5,6,7,8]; EPOCHS=120; PATIENCE=25
torch.manual_seed(SEED); np.random.seed(SEED)
dev=torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def embeddings(rows):
    if os.path.exists(EMB):
        d=np.load(EMB,allow_pickle=True); print(f"loaded cached embeddings {d['E'].shape}")
        return {p:e for p,e in zip(d["paths"],d["E"])}
    tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    net=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1); net.fc=nn.Identity()
    net=net.to(dev).eval()
    paths=[r["path"] for r in rows]; E=[]
    t0=time.time()
    with torch.no_grad():
        for i in range(0,len(paths),64):
            batch=torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i+64]])
            E.append(net(batch.to(dev)).cpu().numpy())
            if (i+64)%640==0: print(f"  {i+64}/{len(paths)}")
    E=np.vstack(E); np.savez_compressed(EMB,E=E,paths=np.array(paths))
    print(f"encoded {E.shape} in {time.time()-t0:.0f}s")
    return {p:e for p,e in zip(paths,E)}

class Seqs(Dataset):
    def __init__(self, items): self.items=items
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        x,l,y=self.items[i]; return torch.tensor(x), l, y

def collate(b):
    xs,ls,ys=zip(*b)
    L=max(ls)
    X=torch.zeros(len(b),L,xs[0].shape[1])
    for i,x in enumerate(xs): X[i,:len(x)]=x
    return X, torch.tensor(ls), torch.tensor(ys)

class LSTMClf(nn.Module):
    def __init__(self,d=512,h=256,drop=0.5):
        super().__init__()
        self.lstm=nn.LSTM(d,h,batch_first=True)
        self.do=nn.Dropout(drop); self.fc=nn.Linear(h,4)
    def forward(self,x,lens):
        p=pack_padded_sequence(x,lens.cpu(),batch_first=True,enforce_sorted=False)
        _,(hn,_)=self.lstm(p)          # final hidden state = last VALID timestep
        return self.fc(self.do(hn[-1]))

def main():
    rows=list(csv.DictReader(open(MANIFEST)))
    emb=embeddings(rows)

    wells=defaultdict(list)
    for r in rows:
        wells[(int(r["pH"]),int(r["well"]))].append((int(r["hour"]),r["path"],r["split"]))
    data={"train":[],"val":[],"test":[]}
    lens=[]
    for (ph,w),items in wells.items():
        items.sort()
        split=items[0][2]
        seq=np.stack([emb[p] for _,p,_ in items]).astype(np.float32)
        lens.append(len(seq))
        data[split].append((seq,len(seq),CLASSES.index(ph)))
    print(f"\nwells: train {len(data['train'])} val {len(data['val'])} test {len(data['test'])}")
    print(f"sequence length: min {min(lens)} max {max(lens)} mean {np.mean(lens):.1f} (11 = complete)")

    dl={k:DataLoader(Seqs(v),batch_size=16,shuffle=(k=="train"),collate_fn=collate) for k,v in data.items()}
    model=LSTMClf().to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    crit=nn.CrossEntropyLoss(label_smoothing=0.05)

    def ev(loader):
        model.eval(); P=[];Y=[]
        with torch.no_grad():
            for X,l,y in loader:
                P.append(model(X.to(dev),l).argmax(1).cpu()); Y.append(y)
        return torch.cat(P).numpy(), torch.cat(Y).numpy()

    best=(-1,None,-1); bad=0; hist=[]
    for ep in range(1,EPOCHS+1):
        model.train(); c=0;n=0
        for X,l,y in dl["train"]:
            X,y=X.to(dev),y.to(dev)
            opt.zero_grad(); out=model(X,l); loss=crit(out,y); loss.backward(); opt.step()
            c+=(out.argmax(1)==y).sum().item(); n+=y.size(0)
        pv,yv=ev(dl["val"]); va=accuracy_score(yv,pv); tra=c/n
        hist.append(dict(epoch=ep,train_acc=tra,val_acc=va))
        if va>best[0]: best=(va,copy.deepcopy(model.state_dict()),ep); bad=0
        else: bad+=1
        if ep%10==0 or ep==1: print(f"  ep{ep:03d} train {tra:.3f} val {va:.3f}")
        if bad>=PATIENCE: print(f"  early stop at {ep} (best val {best[0]:.3f} @ {best[2]})"); break
    model.load_state_dict(best[1])

    res={}
    ptr,ytr=ev(dl["train"]); pv,yv=ev(dl["val"]); pt,yt=ev(dl["test"])
    res["ResNet18(frozen) + LSTM"]=dict(
        train=accuracy_score(ytr,ptr), val=accuracy_score(yv,pv), test=accuracy_score(yt,pt),
        test_f1=f1_score(yt,pt,average="macro"), best_epoch=best[2],
        cm=confusion_matrix(yt,pt).tolist(),
        report=classification_report(yt,pt,target_names=[f"pH{c}" for c in CLASSES],zero_division=0))

    # ---- Ablations: does the sequence actually help? ----
    def flat(split,mode):
        X=[];Y=[]
        for seq,l,y in data[split]:
            X.append(seq[-1] if mode=="last" else seq.mean(0)); Y.append(y)
        return np.array(X),np.array(Y)
    for mode,label in [("last","Last timepoint only (RF)"),("mean","Mean-pooled sequence (RF)")]:
        Xtr,Ytr=flat("train",mode); Xv,Yv=flat("val",mode); Xt,Yt=flat("test",mode)
        rf=RandomForestClassifier(n_estimators=400,random_state=SEED,n_jobs=-1).fit(Xtr,Ytr)
        p=rf.predict(Xt)
        res[label]=dict(train=accuracy_score(Ytr,rf.predict(Xtr)),val=accuracy_score(Yv,rf.predict(Xv)),
            test=accuracy_score(Yt,p),test_f1=f1_score(Yt,p,average="macro"),
            cm=confusion_matrix(Yt,p).tolist(),
            report=classification_report(Yt,p,target_names=[f"pH{c}" for c in CLASSES],zero_division=0))
    maj=max(np.bincount([y for _,_,y in data["train"]]))/len(data["train"])
    res["Baseline (majority)"]=dict(train=maj,val=None,
        test=float(max(np.bincount([y for _,_,y in data["test"]]))/len(data["test"])),test_f1=None)
    res["_history"]=hist
    res["_n_wells"]={k:len(v) for k,v in data.items()}

    print("\n"+"="*74)
    print(f"{'model':32s} {'train':>7s} {'val':>7s} {'test':>7s} {'macroF1':>8s}")
    print("-"*74)
    for k,v in res.items():
        if k.startswith("_"): continue
        f=lambda x: f"{x:7.3f}" if isinstance(x,float) else f"{'-':>7s}"
        print(f"{k:32s} {f(v['train'])} {f(v['val'])} {f(v['test'])} "
              f"{v['test_f1']:8.3f}" if v.get('test_f1') is not None else
              f"{k:32s} {f(v['train'])} {f(v['val'])} {f(v['test'])} {'-':>8s}")
    json.dump(res,open("lstm/results.json","w"),indent=2,default=str)
    print("\nwrote lstm/results.json")

if __name__=="__main__":
    main()
