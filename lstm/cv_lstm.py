"""LSTM under the same grouped 5-fold CV as every other arm.

The single 36-well test split gave 0.806 with a +/-13 point interval. Here each
of the 192 wells is tested exactly once. The LSTM uses no elapsed-time feature;
frames are simply ordered, which is what makes it a sequence model.
"""
import csv, copy
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
from collections import defaultdict
import json

SEED=42; CLASSES=[5,6,7,8]
torch.manual_seed(SEED); np.random.seed(SEED)
dev=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
rows=list(csv.DictReader(open("preprocessing/splits.csv")))
d=np.load("lstm/_embeddings.npz",allow_pickle=True); emb={p:e for p,e in zip(d["paths"],d["E"])}

wells=defaultdict(list)
for r in rows: wells[(int(r["pH"]),int(r["well"]))].append((int(r["hour"]),r["path"]))
keys=sorted(wells)
X=[np.stack([emb[p] for _,p in sorted(wells[k])]).astype(np.float32) for k in keys]
Y=np.array([CLASSES.index(k[0]) for k in keys])
print(f"{len(keys)} wells, sequence lengths {min(map(len,X))}-{max(map(len,X))}")

def collate(b):
    xs,ys=zip(*b); L=max(len(x) for x in xs)
    T=torch.zeros(len(b),L,512)
    for i,x in enumerate(xs): T[i,:len(x)]=torch.tensor(x)
    return T, torch.tensor([len(x) for x in xs]), torch.tensor(ys)

class DS(Dataset):
    def __init__(s,X,Y): s.X=X; s.Y=Y
    def __len__(s): return len(s.X)
    def __getitem__(s,i): return s.X[i], s.Y[i]

class M(nn.Module):
    def __init__(s):
        super().__init__(); s.l=nn.LSTM(512,256,batch_first=True); s.d=nn.Dropout(0.5); s.f=nn.Linear(256,4)
    def forward(s,x,l):
        _,(h,_)=s.l(pack_padded_sequence(x,l.cpu(),batch_first=True,enforce_sorted=False))
        return s.f(s.d(h[-1]))

sgk=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
accs=[]; bins=[]
for fold,(tr,te) in enumerate(sgk.split(np.zeros(len(Y)),Y,groups=np.arange(len(Y))),1):
    # hold out 20% of train wells for early stopping
    n_val=max(1,int(0.2*len(tr))); rng=np.random.RandomState(SEED); perm=rng.permutation(tr)
    va, tr2 = perm[:n_val], perm[n_val:]
    dl_tr=DataLoader(DS([X[i] for i in tr2],Y[tr2]),16,shuffle=True,collate_fn=collate)
    dl_va=DataLoader(DS([X[i] for i in va],Y[va]),16,collate_fn=collate)
    dl_te=DataLoader(DS([X[i] for i in te],Y[te]),16,collate_fn=collate)
    m=M().to(dev); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-3)
    crit=nn.CrossEntropyLoss(label_smoothing=0.05)
    def ev(dl):
        m.eval(); P=[];T=[]
        with torch.no_grad():
            for x,l,yy in dl: P.append(m(x.to(dev),l).argmax(1).cpu()); T.append(yy)
        return torch.cat(P).numpy(), torch.cat(T).numpy()
    best=(-1,None); bad=0
    for ep in range(200):
        m.train()
        for x,l,yy in dl_tr:
            opt.zero_grad(); loss=crit(m(x.to(dev),l),yy.to(dev)); loss.backward(); opt.step()
        p,t=ev(dl_va); a=accuracy_score(t,p)
        if a>best[0]: best=(a,copy.deepcopy(m.state_dict())); bad=0
        else: bad+=1
        if bad>=25: break
    m.load_state_dict(best[1]); p,t=ev(dl_te)
    accs.append(accuracy_score(t,p)); bins.append(accuracy_score(t>=2,p>=2))
    print(f"  fold {fold}: {len(te)} test wells, acc {accs[-1]:.3f}, acid/alk {bins[-1]:.3f}")

print(f"\nResNet18(frozen) + LSTM, per-well: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
print(f"                      acid/alkaline: {np.mean(bins):.3f} +/- {np.std(bins):.3f}")
json.dump(dict(well=float(np.mean(accs)),well_std=float(np.std(accs)),
               binary=float(np.mean(bins))),open("lstm/cv_results.json","w"),indent=2)
