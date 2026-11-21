"""Train the learned candidate scorer on TRAIN and select the model + threshold on VAL.

The test split is never touched. The chosen model is saved to models/pwave_ml.pkl.
"""
import json
import os
import pickle
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import splits                                            # noqa: E402
from bench import DATASETS, aggregate, score_item        # noqa: E402
from pwave import PParams                                # noqa: E402
from pwave_ml import FEATURES, MLDetector, decide, extract, label  # noqa: E402

HERE = splits.HERE
PRM = PParams()


def featurize(split):
    path = os.path.join(HERE, f".cache_feat_{split}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    out = []
    for it in splits.load(split):
        X, meta, ctx = extract(it["signal"].astype(float), it["fs"], PRM, it["rpeaks"])
        fs = it["fs"]
        y = label(meta, fs, it["p"])
        # rows usable for training: beats lying inside the labelled region
        lo, hi = it["range"]
        use = np.array([lo <= r / fs <= hi for _, _, r in meta], bool)
        if it["label_scope"] == "beats":
            use &= np.array([np.any(np.abs(it["qrs"] - r / fs) <= 0.08) for _, _, r in meta], bool)
        out.append(dict(item=it, X=X, y=y, use=use, meta=meta,
                        ctx=None if ctx is None else dict(xp=ctx["xp"], pol=ctx["pol"])))
    pickle.dump(out, open(path, "wb"), protocol=4)
    return out


def stack(feats):
    X = np.concatenate([f["X"][f["use"]] for f in feats if len(f["meta"])])
    y = np.concatenate([f["y"][f["use"]] for f in feats if len(f["meta"])])
    ds = np.concatenate([[f["item"]["dataset"]] * int(f["use"].sum()) for f in feats if len(f["meta"])])
    w = np.ones(len(y))
    for d in DATASETS:  # equal total weight per dataset (macro objective)
        m = ds == d
        if m.any():
            w[m] = len(y) / (len(DATASETS) * m.sum())
    return X, y, w


def evaluate(feats, model, thrs):
    probs = [model.predict_proba(f["X"])[:, 1] if len(f["meta"]) else np.array([]) for f in feats]
    res = {}
    for thr in thrs:
        per = {d: [] for d in DATASETS}
        for f, p in zip(feats, probs):
            det = decide(p, f["meta"], f["ctx"], thr) if len(f["meta"]) else np.array([])
            per[f["item"]["dataset"]].append(
                score_item(f["item"], PRM, detector=lambda *a, _d=det, **k: _d))
        r = {d: aggregate(rows) for d, rows in per.items() if rows}
        r["macro_f1"] = float(np.mean([r[d]["f1"] for d in DATASETS if d in r]))
        res[thr] = r
    return res


MODELS = {
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
    "hgb_small": lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.1,
                                                         l2_regularization=1.0, random_state=0),
    "hgb_mid": lambda: HistGradientBoostingClassifier(max_depth=5, max_iter=300, learning_rate=0.05,
                                                       min_samples_leaf=50, l2_regularization=1.0,
                                                       random_state=0),
    "hgb_large": lambda: HistGradientBoostingClassifier(max_depth=8, max_iter=400, learning_rate=0.05,
                                                         min_samples_leaf=100, l2_regularization=1.0,
                                                         random_state=0),
}
THRS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

if __name__ == "__main__":
    t0 = time.time()
    ftr, fva = featurize("train"), featurize("val")
    Xtr, ytr, wtr = stack(ftr)
    print(f"features: {len(FEATURES)}, train candidates {len(ytr)} (pos {ytr.sum()}), "
          f"{time.time()-t0:.0f}s", flush=True)
    results = {}
    for name, mk in MODELS.items():
        m = mk()
        if name == "logreg":
            m.fit(Xtr, ytr, logisticregression__sample_weight=wtr)
        else:
            m.fit(Xtr, ytr, sample_weight=wtr)
        rtr = evaluate(ftr, m, THRS)
        rva = evaluate(fva, m, THRS)
        for thr in THRS:
            print(f"{name:10s} thr={thr:.1f}  train={rtr[thr]['macro_f1']*100:.2f}  val={rva[thr]['macro_f1']*100:.2f}  "
                  + " ".join(f"{d}={rva[thr][d]['f1']*100:.1f}" for d in DATASETS), flush=True)
        results[name] = (m, rtr, rva)
    best = max(((n, t) for n in results for t in THRS), key=lambda nt: results[nt[0]][2][nt[1]]["macro_f1"])
    name, thr = best
    m, rtr, rva = results[name]
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    MLDetector(m, thr).save(os.path.join(HERE, "models", "pwave_ml.pkl"))
    json.dump(dict(model=name, thr=thr, train_macro=rtr[thr]["macro_f1"], val_macro=rva[thr]["macro_f1"],
                   val={d: rva[thr][d]["f1"] for d in DATASETS}),
              open(os.path.join(HERE, "models", "pwave_ml.json"), "w"), indent=1)
    print(f"chosen {name} thr={thr}: train={rtr[thr]['macro_f1']*100:.2f} val={rva[thr]['macro_f1']*100:.2f}")
