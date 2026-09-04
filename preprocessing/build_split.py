"""Build a leakage-free, well-wise train/val/test split manifest.

The split in `feature_extraction.ipynb` groups by the W-number parsed from the
filename. That number is reused across pH classes -- `W13` names four different
physical gels -- so it does not identify a well. The `Split_Data/` tree on disk
is worse still: all 192 physical wells appear in train, and 170/175 of them also
appear in val/test, i.e. it is effectively a per-image random split. Because each
well is photographed at 11 timepoints, that puts near-duplicate images of the
same gel on both sides of the boundary.

This script groups by the physical well -- the (pH, well) pair -- and splits
those 192 groups, stratified by pH. Output is a manifest CSV rather than a copied
image tree, so the split is reproducible and costs no disk.
"""
import os, re, csv, random
from collections import defaultdict

SRC   = "Preprocessed_Data"      # circular crops from data.ipynb
OUT   = "preprocessing/splits.csv"
SEED  = 42
RATIO = (0.60, 0.20, 0.20)       # train / val / test, by well

FNAME = re.compile(r"cropped_(\d+)hr_pH(\d+)_W(\d+)\.", re.I)

def main():
    rows = []
    for dirpath, _, filenames in os.walk(SRC):
        for fn in filenames:
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            m = FNAME.match(fn)
            if not m:
                print(f"  skip (unparsed name): {fn}")
                continue
            hour, ph, well = (int(g) for g in m.groups())
            rows.append({
                "path": os.path.join(dirpath, fn),
                "pH": ph, "well": well, "hour": hour,
            })

    # Group by physical well: (pH, well). 192 groups, 48 per pH.
    wells = defaultdict(list)
    for r in rows:
        wells[(r["pH"], r["well"])].append(r)

    # Stratify by pH so every split keeps all four classes in proportion.
    by_ph = defaultdict(list)
    for key in wells:
        by_ph[key[0]].append(key)

    rng = random.Random(SEED)
    assign = {}
    for ph, keys in sorted(by_ph.items()):
        keys = sorted(keys)
        rng.shuffle(keys)
        n = len(keys)
        n_tr = int(round(RATIO[0] * n))
        n_va = int(round(RATIO[1] * n))
        for k in keys[:n_tr]:            assign[k] = "train"
        for k in keys[n_tr:n_tr + n_va]: assign[k] = "val"
        for k in keys[n_tr + n_va:]:     assign[k] = "test"

    for r in rows:
        r["split"] = assign[(r["pH"], r["well"])]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "pH", "well", "hour", "split"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["split"], r["pH"], r["well"], r["hour"])))

    # Report, and assert the property the old split failed to hold.
    print(f"\nwrote {OUT}  ({len(rows)} images, {len(wells)} physical wells)\n")
    seen = defaultdict(set)
    for r in rows:
        seen[r["split"]].add((r["pH"], r["well"]))
    for s in ("train", "val", "test"):
        imgs = [r for r in rows if r["split"] == s]
        dist = {p: sum(1 for r in imgs if r["pH"] == p) for p in (5, 6, 7, 8)}
        print(f"  {s:5s}  wells={len(seen[s]):3d}  images={len(imgs):4d}  per-pH={dist}")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = seen[a] & seen[b]
        print(f"  well overlap {a}&{b}: {len(overlap)}")
        assert not overlap, f"LEAKAGE: {len(overlap)} wells shared by {a}/{b}"
    print("\nOK - no physical well appears in more than one split.")

if __name__ == "__main__":
    main()
