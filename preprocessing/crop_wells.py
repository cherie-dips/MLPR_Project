"""Stage 1: detect the hydrogel well in each raw photograph and crop it.

Ported from data.ipynb (now removed). Reads `New MLPR Data/` and writes
`Preprocessed_Data/`, preserving the <time> hr/pH<n> <condition>/ layout and
prefixing each file with `cropped_`.

The notebook contained several HSV threshold variants because gel colour shifts
with both pH and degradation time; a single global range loses wells at the
extremes. GREEN_RANGES below holds the variants, tried in order until one yields
a valid well, which recovers more images than any single range alone.

Known limitation: with the notebook's single range, 149 of 2112 images (7.1%)
produced no contour and were silently skipped, concentrated at 0 hr (68) and
216/264 hr (47). This script reports the failures instead of hiding them.
"""
import os, sys, argparse
import cv2, numpy as np

SRC_DEFAULT="New MLPR Data"; DST_DEFAULT="Preprocessed_Data"

# (lower, upper) HSV bounds, tried in order. From the notebook's tuning cells.
GREEN_RANGES=[
    ((10, 20, 100), (60, 200, 255)),   # primary: the full-dataset run
    ((20, 20, 100), (70, 255, 255)),   # wider, for pale 0 hr wells
    ((35, 90,  80), (75, 255, 255)),   # saturated mid-timepoint wells
    ((55, 30,  40), (75, 110, 120)),   # dark late-timepoint wells
    ((65,100,  50), (85, 255, 255)),
]
MIN_AREA=5000; MIN_RADIUS=30; MIN_CIRCULARITY=0.6

def find_well(image, blur=5, kernel=7):
    """Return (x, y, r) of the most circular large green blob, or None."""
    hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for lo, hi in GREEN_RANGES:
        mask=cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask=cv2.GaussianBlur(mask, (blur, blur), 0)
        k=np.ones((kernel, kernel), np.uint8)
        mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        contours,_=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best=None; best_r=0
        for cnt in contours:
            area=cv2.contourArea(cnt)
            if area < MIN_AREA: continue
            (x,y),r=cv2.minEnclosingCircle(cnt)
            circ=(4*np.pi*area)/(cv2.arcLength(cnt, True)**2 + 1e-5)
            if r > best_r and r > MIN_RADIUS and circ > MIN_CIRCULARITY:
                best_r=r; best=(int(x), int(y), int(r))
        if best: return best
    return None

def crop(image, well):
    """Mask everything outside the well circle, then crop to its bounding box."""
    x,y,r=well
    h,w=image.shape[:2]
    circle=np.zeros((h,w), np.uint8)
    cv2.circle(circle,(x,y),r,255,-1)
    masked=cv2.bitwise_and(image,image,mask=circle)
    x1,y1=max(x-r,0), max(y-r,0); x2,y2=min(x+r,w), min(y+r,h)
    out=masked[y1:y2, x1:x2]
    return out if out.size else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT); ap.add_argument("--dst", default=DST_DEFAULT)
    a=ap.parse_args()
    ok=0; failed=[]
    for dirpath,_,filenames in os.walk(a.src):
        for fn in sorted(filenames):
            if not fn.lower().endswith((".jpg",".jpeg",".png")): continue
            img=cv2.imread(os.path.join(dirpath,fn))
            if img is None:
                failed.append((os.path.join(dirpath,fn),"unreadable")); continue
            well=find_well(img)
            if well is None:
                failed.append((os.path.join(dirpath,fn),"no well found")); continue
            out=crop(img,well)
            if out is None:
                failed.append((os.path.join(dirpath,fn),"empty crop")); continue
            rel=os.path.relpath(dirpath,a.src)
            dest=os.path.join(a.dst,rel); os.makedirs(dest,exist_ok=True)
            cv2.imwrite(os.path.join(dest,f"cropped_{fn}"),out); ok+=1
    total=ok+len(failed)
    print(f"cropped {ok}/{total} images -> {a.dst}")
    if failed:
        print(f"\n{len(failed)} FAILURES ({100*len(failed)/total:.1f}%) - these are silently "
              f"absent downstream:")
        for p,why in failed[:20]: print(f"  {why:14s} {p}")
        if len(failed)>20: print(f"  ... and {len(failed)-20} more")
    return 0

if __name__=="__main__":
    sys.exit(main())
