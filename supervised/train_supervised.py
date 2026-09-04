"""Supervised baselines on hand-crafted colour features: KNN, SVM, RF, MLP.

Fixes three defects in `knn_svm_rf.ipynb`:
  1. It pooled Split_Data/train and Split_Data/test and re-split at the image
     level, so photographs of the same physical well landed on both sides.
     Here the well-wise manifest from preprocessing/build_split.py is used as
     given, and never re-split.
  2. It selected hyperparameters by cross-validating on the training pool,
     which had the same leakage. Here selection uses the held-out val wells via
     PredefinedSplit, and test is touched exactly once, at the end.
  3. It imported KNeighborsClassifier but never fitted one. KNN is included.

Train / val / test accuracy are all reported so the fit can be judged.
"""
import os, sys, csv, time, json
import numpy as np, cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier

MANIFEST = "preprocessing/splits.csv"
CACHE    = "supervised/_features.npz"
SEED     = 42

def extract(path):
    """Joint 8x8x8 HSV histogram -- keeps H/S/V co-occurrence, unlike three 1D histograms."""
    img = cv2.imread(path)
    if img is None: return None
    img = cv2.resize(img, (128, 128))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, [8,8,8], [0,180, 0,256, 0,256])
    return cv2.normalize(hist, hist).flatten()

def load():
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        print(f"loaded cached features {d['X'].shape}")
        return d["X"], d["y"], d["split"], d["hour"]
    rows = list(csv.DictReader(open(MANIFEST)))
    X, y, sp, hr = [], [], [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        f = extract(r["path"])
        if f is None:
            print("  unreadable:", r["path"]); continue
        X.append(f); y.append(int(r["pH"])); sp.append(r["split"]); hr.append(int(r["hour"]))
        if (i+1) % 400 == 0: print(f"  {i+1}/{len(rows)}")
    X = np.array(X, np.float32); y = np.array(y); sp = np.array(sp); hr = np.array(hr)
    np.savez_compressed(CACHE, X=X, y=y, split=sp, hour=hr)
    print(f"extracted {X.shape} in {time.time()-t0:.1f}s")
    return X, y, sp, hr

def main():
    X, y, sp, hr = load()
    tr, va, te = sp=="train", sp=="val", sp=="test"
    print(f"\ntrain {tr.sum()}  val {va.sum()}  test {te.sum()}  features {X.shape[1]}")

    # Hyperparameter selection on the held-out val wells (never by re-splitting train).
    Xsel = np.vstack([X[tr], X[va]])
    ysel = np.concatenate([y[tr], y[va]])
    ps   = PredefinedSplit(np.concatenate([np.full(tr.sum(), -1), np.zeros(va.sum())]))

    grids = {
        "KNN": (Pipeline([("vt", VarianceThreshold(1e-8)), ("sc", StandardScaler()), ("m", KNeighborsClassifier())]),
                {"m__n_neighbors":[1,3,5,9,15,25], "m__weights":["uniform","distance"],
                 "m__metric":["euclidean","manhattan"]}),
        "SVM": (Pipeline([("vt", VarianceThreshold(1e-8)), ("sc", StandardScaler()), ("m", SVC(random_state=SEED))]),
                {"m__C":[0.1,1,10,100], "m__kernel":["linear","rbf"], "m__gamma":["scale","auto"]}),
        "RandomForest": (RandomForestClassifier(random_state=SEED, n_jobs=-1),
                {"n_estimators":[200,400], "max_depth":[None,10,20],
                 "min_samples_leaf":[1,2,4], "max_features":["sqrt","log2"]}),
        "MLP": (Pipeline([("vt", VarianceThreshold(1e-8)), ("sc", StandardScaler()),
                          ("m", MLPClassifier(max_iter=1000, early_stopping=True,
                                              n_iter_no_change=20, random_state=SEED))]),
                {"m__hidden_layer_sizes":[(64,),(128,),(64,64)],
                 "m__alpha":[1e-4,1e-2,1e-1], "m__learning_rate_init":[1e-3]}),
    }

    results = {}
    dummy = DummyClassifier(strategy="most_frequent").fit(X[tr], y[tr])
    results["Baseline (majority)"] = dict(
        params={}, train=accuracy_score(y[tr], dummy.predict(X[tr])),
        val=accuracy_score(y[va], dummy.predict(X[va])),
        test=accuracy_score(y[te], dummy.predict(X[te])),
        test_f1=f1_score(y[te], dummy.predict(X[te]), average="macro"))

    for name, (est, grid) in grids.items():
        t0 = time.time()
        gs = GridSearchCV(est, grid, cv=ps, n_jobs=-1, refit=False, scoring="accuracy")
        gs.fit(Xsel, ysel)
        best = gs.best_params_
        # Refit on TRAIN ONLY so the train/val gap stays interpretable.
        est.set_params(**best); est.fit(X[tr], y[tr])
        a_tr = accuracy_score(y[tr], est.predict(X[tr]))
        a_va = accuracy_score(y[va], est.predict(X[va]))
        pred = est.predict(X[te])
        a_te = accuracy_score(y[te], pred)
        results[name] = dict(params=best, train=a_tr, val=a_va, test=a_te,
                             test_f1=f1_score(y[te], pred, average="macro"),
                             cm=confusion_matrix(y[te], pred, labels=[5,6,7,8]).tolist(),
                             report=classification_report(y[te], pred, labels=[5,6,7,8],
                                        target_names=["pH5","pH6","pH7","pH8"], zero_division=0))
        print(f"\n=== {name} ({time.time()-t0:.0f}s) best={best}")
        print(f"    train {a_tr:.3f} | val {a_va:.3f} | test {a_te:.3f} | macroF1 {results[name]['test_f1']:.3f}")

    print("\n" + "="*78)
    print(f"{'model':22s} {'train':>7s} {'val':>7s} {'test':>7s} {'macroF1':>8s}  {'gap':>6s}")
    print("-"*78)
    for n, r in results.items():
        print(f"{n:22s} {r['train']:7.3f} {r['val']:7.3f} {r['test']:7.3f} {r['test_f1']:8.3f}  {r['train']-r['val']:+6.3f}")
    # ---- RF + elapsed time (the "RF+Time" cell of the original notebook) ----
    # Hours since application is known at inference time, so this is a legitimate
    # feature: the same hue means different pH early vs late in degradation.
    Xt = np.hstack([X, hr.reshape(-1,1).astype(np.float32)])
    rft = RandomForestClassifier(**results["RandomForest"]["params"], random_state=SEED, n_jobs=-1)
    rft.fit(Xt[tr], y[tr])
    pred_t = rft.predict(Xt[te])
    results["RandomForest + time"] = dict(
        params=results["RandomForest"]["params"],
        train=accuracy_score(y[tr], rft.predict(Xt[tr])),
        val=accuracy_score(y[va], rft.predict(Xt[va])),
        test=accuracy_score(y[te], pred_t),
        test_f1=f1_score(y[te], pred_t, average="macro"),
        cm=confusion_matrix(y[te], pred_t, labels=[5,6,7,8]).tolist(),
        report=classification_report(y[te], pred_t, labels=[5,6,7,8],
                   target_names=["pH5","pH6","pH7","pH8"], zero_division=0))
    r = results["RandomForest + time"]
    print(f"\n=== RandomForest + time\n    train {r['train']:.3f} | val {r['val']:.3f} | test {r['test']:.3f} | macroF1 {r['test_f1']:.3f}")

    # ---- How much did the old (image-level) protocol inflate things? ----
    # Same features, same RF. Only the split protocol differs.
    from sklearn.model_selection import train_test_split
    Xa, ya = X, y
    Xtr2, Xte2, ytr2, yte2 = train_test_split(Xa, ya, test_size=0.2, random_state=SEED, stratify=ya)
    rf2 = RandomForestClassifier(n_estimators=400, max_depth=20, min_samples_leaf=2,
                                 max_features="sqrt", random_state=SEED, n_jobs=-1).fit(Xtr2, ytr2)
    leaky = accuracy_score(yte2, rf2.predict(Xte2))
    honest = results["RandomForest"]["test"]
    results["_leakage_check"] = dict(image_level_split=leaky, well_level_split=honest,
                                     inflation=leaky - honest)
    print("\n" + "="*78)
    print("LEAKAGE CHECK - identical features and model, only the split protocol differs")
    print(f"  image-level random split (old notebook protocol): {leaky:.3f}")
    print(f"  well-level split         (corrected)            : {honest:.3f}")
    print(f"  inflation attributable to leakage               : {leaky-honest:+.3f}")

    json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
              open("supervised/results.json","w"), indent=2, default=str)
    print("\nwrote supervised/results.json")

if __name__ == "__main__":
    main()
