"""Run the P-wave detector over the cached splits and benchmark it.

Score = macro-average F1 across the four datasets (each dataset counts equally).
Usage: python src/bench.py [train|val|test]
"""
import os
import sys
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import in_scope            # noqa: E402
from metrics import match                # noqa: E402
from pwave import PParams, detect_pwaves  # noqa: E402
import splits                             # noqa: E402

DATASETS = ["butpdb", "pwave", "ludb", "qtdb"]


def score_item(it, prm, tol=0.05, detector=detect_pwaves):
    fs = it["fs"]
    det = detector(it["signal"].astype(float), fs, prm, rpeaks=it["rpeaks"]) / fs
    lo, hi = it["range"]
    det = det[(det >= lo - tol) & (det <= hi + tol)]
    if it["label_scope"] == "beats":
        keep = np.zeros(len(det), bool)
        for q in it["qrs"]:
            keep |= (det >= q - prm.search - tol) & (det <= q)
        det = det[keep]
    gt = it["p"]
    scope = in_scope(gt, it["qrs"], prm.search)
    m = match(gt, det, tol)
    tp_idx = np.array([i for i, _, _ in m], int)
    return dict(n_gt=len(gt), n_scope=int(scope.sum()), tp=len(m),
                tp_scope=int(scope[tp_idx].sum()) if len(m) else 0,
                fp=len(det) - len(m), err=[d for *_, d in m],
                det=det, gt=gt, scope=scope, match=m)


def aggregate(rows):
    g = lambda k: sum(r[k] for r in rows)
    se = g("tp_scope") / max(1, g("n_scope"))
    ppv = g("tp") / max(1, g("tp") + g("fp"))
    f1 = 2 * se * ppv / max(1e-9, se + ppv)
    err = np.concatenate([np.asarray(r["err"]) for r in rows]) if rows else np.array([])
    return dict(se=se, ppv=ppv, f1=f1, n_scope=g("n_scope"), tp=g("tp"), fp=g("fp"),
                err_mean=1e3 * err.mean() if len(err) else np.nan,
                err_sd=1e3 * err.std() if len(err) else np.nan)


def run(items, prm, detector=detect_pwaves):
    per = {ds: [] for ds in DATASETS}
    for it in items:
        per[it["dataset"]].append(score_item(it, prm, detector=detector))
    res = {ds: aggregate(rows) for ds, rows in per.items() if rows}
    res["macro_f1"] = float(np.mean([res[ds]["f1"] for ds in DATASETS if ds in res]))
    return res


def fmt(res):
    lines = []
    for ds in DATASETS:
        if ds in res:
            r = res[ds]
            lines.append(f"  {ds:7s} Se={r['se']*100:5.1f}%  PPV={r['ppv']*100:5.1f}%  F1={r['f1']*100:5.1f}%  "
                         f"err={r['err_mean']:+5.1f}±{r['err_sd']:4.1f} ms  (P in scope {r['n_scope']})")
    lines.append(f"  macro F1 = {res['macro_f1']*100:.2f}%")
    return "\n".join(lines)


if __name__ == "__main__":
    sp = sys.argv[1] if len(sys.argv) > 1 else "val"
    if sp == "test" and "--final" not in sys.argv:
        sys.exit("test split is reserved for the final report: add --final")
    prm = PParams()
    print("params:", asdict(prm))
    print(f"[{sp}]\n" + fmt(run(splits.load(sp), prm)))
