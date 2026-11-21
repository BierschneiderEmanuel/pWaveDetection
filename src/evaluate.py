"""Score the P-wave detector against the BUT PDB JSON export (ECG, 128 Hz)."""
import argparse
import os
import sys
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data import load_records          # noqa: E402
from elgendi import elgendi_rpeaks      # noqa: E402
from metrics import match               # noqa: E402
from pwave import PParams, detect_pwaves  # noqa: E402


def in_scope(p, qrs, search):
    """Ground-truth P waves that fall within `search` s before a ground-truth QRS."""
    m = np.zeros(len(p), bool)
    for i, t in enumerate(p):
        nxt = qrs[qrs > t]
        m[i] = len(nxt) > 0 and nxt[0] - t <= search
    return m


def run(records, prm, tol=0.05):
    rows = []
    for r in records:
        fs = r["fs"]
        rp = elgendi_rpeaks(r["signal"], fs)
        det = detect_pwaves(r["signal"], fs, prm, rpeaks=rp) / fs
        # score only within the region where the labels are complete
        lo, hi = r.get("range", (0.0, np.inf))
        det = det[(det >= lo - tol) & (det <= hi + tol)]
        if r.get("label_scope") == "beats":
            # only the labelled beats are annotated: score detections inside their search windows
            keep = np.zeros(len(det), bool)
            for q in r["qrs"]:
                keep |= (det >= q - prm.search - tol) & (det <= q)
            det = det[keep]
        rp_s = rp / fs
        rp_s = rp_s[(rp_s >= lo - 0.075) & (rp_s <= hi + 0.075)]
        gt = r["p"]
        scope = in_scope(gt, r["qrs"], prm.search)
        m = match(gt, det, tol)
        tp_idx = np.array([i for i, _, _ in m], int)
        err = np.array([d for *_, d in m])
        tp = len(m)
        tp_scope = int(scope[tp_idx].sum()) if tp else 0
        if r.get("label_scope") == "beats":
            rp_s = np.array([t for t in rp_s if np.any(np.abs(r["qrs"] - t) <= 0.075)])
        rq = match(r["qrs"], rp_s, 0.075)
        rows.append(dict(record=r["record"], lead=r["lead"], n_gt=len(gt), n_scope=int(scope.sum()),
                         n_det=len(det), tp=tp, tp_scope=tp_scope, fp=len(det) - tp, err=err,
                         r_tp=len(rq), r_gt=len(r["qrs"]), r_det=len(rp_s)))
    return rows


def summarize(rows):
    g = lambda k: sum(x[k] for x in rows)
    tp, fp = g("tp"), g("fp")
    err = np.concatenate([x["err"] for x in rows]) if rows else np.array([])
    se_all = tp / max(1, g("n_gt"))
    se_scope = g("tp_scope") / max(1, g("n_scope"))
    ppv = tp / max(1, tp + fp)
    f1 = 2 * se_scope * ppv / max(1e-9, se_scope + ppv)
    return dict(n_gt=g("n_gt"), n_scope=g("n_scope"), n_det=g("n_det"), tp=tp, fp=fp,
                se_all=se_all, se_scope=se_scope, ppv=ppv, f1_scope=f1,
                err_mean_ms=1e3 * err.mean() if len(err) else np.nan,
                err_sd_ms=1e3 * err.std() if len(err) else np.nan,
                r_se=g("r_tp") / max(1, g("r_gt")), r_ppv=g("r_tp") / max(1, g("r_det")))


def fmt(s):
    return (f"GT P={s['n_gt']:5d} (in 300ms scope {s['n_scope']:5d})  det={s['n_det']:5d}  TP={s['tp']:5d}  FP={s['fp']:5d}\n"
            f"  Se(all)={s['se_all']*100:5.1f}%  Se(scope)={s['se_scope']*100:5.1f}%  PPV={s['ppv']*100:5.1f}%  "
            f"F1(scope)={s['f1_scope']*100:5.1f}%  loc err={s['err_mean_ms']:+.1f}±{s['err_sd_ms']:.1f} ms  "
            f"| Elgendi R: Se={s['r_se']*100:.1f}% PPV={s['r_ppv']*100:.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), ".."))
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--per-record", action="store_true")
    a = ap.parse_args()
    prm = PParams()
    print("params:", asdict(prm))
    recs = list(load_records(a.data))
    rows = run(recs, prm, a.tol)
    if a.per_record:
        for x in rows:
            print(f"{x['record']} L{x['lead']}  gt={x['n_gt']:4d} scope={x['n_scope']:4d} det={x['n_det']:4d} "
                  f"tp={x['tp']:4d} fp={x['fp']:4d}")
    for lead in sorted({x["lead"] for x in rows}):
        print(f"lead {lead}: " + fmt(summarize([x for x in rows if x['lead'] == lead])))
    print("all   : " + fmt(summarize(rows)))
