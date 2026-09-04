"""Hand-written rule-based pH classifiers - the floor any learned model must beat.

Ported from classical_Programming.ipynb (now removed) and vectorised; the
original looped over every pixel in Python. Neither classifier is trained, so
there is no split to respect - but the thresholds were hand-tuned by eye on this
same data, so these are effectively training-set numbers and are optimistic.
"""
import csv, json
import numpy as np, cv2
from collections import defaultdict
from sklearn.metrics import accuracy_score, confusion_matrix

CLASSES=[5,6,7,8]

def green_shade_vote(img_bgr):
    """Bucket 'green-dominant' pixels by intensity; each bucket votes for a pH."""
    b,g,r=img_bgr[:,:,0].astype(int), img_bgr[:,:,1].astype(int), img_bgr[:,:,2].astype(int)
    green=(g>r)&(g>b)
    counts={
        8: np.sum(green & (g<80)             & (r<70)  & (b<70)),
        7: np.sum(green & (g>=80) &(g<130)   & (r<100) & (b<100)),
        6: np.sum(green & (g>=130)&(g<180)   & (r<130) & (b<130)),
        5: np.sum(green & (g>=180)           & (r<170) & (b<170)),
    }
    return max(counts, key=counts.get)

def mean_green_threshold(img_bgr):
    """Threshold the mean green channel, ignoring the black circular-mask corners."""
    mask=img_bgr.sum(2)>15
    mg=img_bgr[:,:,1][mask].mean() if mask.any() else 0
    return 5 if mg<60 else 6 if mg<75 else 7 if mg<90 else 8

def main():
    rows=list(csv.DictReader(open("preprocessing/splits.csv")))
    y=[]; p1=[]; p2=[]; per_split=defaultdict(lambda: ([],[]))
    for i,r in enumerate(rows):
        img=cv2.imread(r["path"])
        if img is None: continue
        img=cv2.resize(img,(128,128))
        t=int(r["pH"]); a=green_shade_vote(img); b=mean_green_threshold(img)
        y.append(t); p1.append(a); p2.append(b)
        per_split[r["split"]][0].append(t); per_split[r["split"]][1].append(a)
        if (i+1)%600==0: print(f"  {i+1}/{len(rows)}")
    y=np.array(y)
    res={}
    print(f"\n{len(y)} images, majority baseline {max(np.bincount(y)[5:])/len(y):.3f}\n")
    for name,pred in [("green-shade pixel voting",np.array(p1)),
                      ("mean-green threshold",np.array(p2))]:
        acc=accuracy_score(y,pred)
        binacc=accuracy_score(y>=7,pred>=7)
        res[name]=dict(accuracy=float(acc), acid_alkaline=float(binacc))
        print(f"{name}: accuracy {acc:.3f}   acid/alkaline {binacc:.3f}")
        cm=confusion_matrix(y,pred,labels=CLASSES)
        print("  cm (rows=true 5,6,7,8):")
        for row in cm: print("   ",row)
    json.dump(res,open("supervised/rule_based_results.json","w"),indent=2)
    print("\nwrote supervised/rule_based_results.json")

if __name__=="__main__":
    main()
