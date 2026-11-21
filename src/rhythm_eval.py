"""Rhythm-stratified results using the AF labels (verdict 1 = atrial fibrillation, 0 = other).

Every reference QRS, reference P and detection is given the rhythm of the AF label that
covers its time; times without an AF label count as "unknown" and are left out.
Per rhythm and version:
  beats        reference QRS complexes in scope (QTDB: only the annotated beats)
  FP/100 beats false P detections per 100 beats; in AF nearly every detection is false,
               because AF segments carry (almost) no P labels
  Se, PPV, F1  as computed in bench.py, pooled over all databases (micro)
95 % intervals are obtained from a patient bootstrap (B resamples of the patient groups that
have beats in that rhythm, stratified by database), as in bootstrap_ci.py.

Usage: python src/rhythm_eval.py [val|test ...] [--final] [--B 2000]
Writes results_rhythm_<splits>.txt and results_rhythm_<splits>.json.
"""
import glob
import json
import os
import sys
from collections import defaultdict
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import splits                             # noqa: E402
from bench import DATASETS, score_item     # noqa: E402
from bootstrap_ci import ci               # noqa: E402
from data import _record_id, find_channel      # noqa: E402
from pwave import PParams, detect_pwaves   # noqa: E402
from pwave_ml import MLDetector           # noqa: E402

RHYTHMS = ("non-AF", "AF")
KEYS = ("beats", "n_scope", "tp_scope", "tp", "fp")


def load_af():
    """{(dataset, record, lead): [(start, end, verdict), ...]} from the converted JSON files."""
    out = {}
    for ds, folder in splits.DATASETS.items():
        for path in sorted(glob.glob(os.path.join(folder, "*_annotations.json"))):
            rec = _record_id(path)
            with open(path) as fh:
                d = json.load(fh)
            for lead, snip in enumerate(d["recording"]):
                ch = find_channel(snip)
                out[(ds, rec, lead)] = [(l["coordinates"]["start"], l["coordinates"]["end"],
                                         int(l["decision"]["verdict"]))
                                        for l in ch["annotations"][0]["labels"] if l["name"] == "AF"]
    return out


def rhythm_at(t, iv):
    """Index into RHYTHMS for each time, -1 where no AF label covers it."""
    t = np.asarray(t, float)
    r = np.full(len(t), -1)
    for t0, t1, v in iv:
        r[(t >= t0) & (t < t1)] = v
    return r


def item_counts(it, prm, detector, iv):
    s = score_item(it, prm, detector=detector)
    lo, hi = it["range"]
    q = np.asarray(it["qrs"], float)
    q = q[(q >= lo) & (q <= hi)]
    det, gt, scope = s["det"], s["gt"], s["scope"]
    tp_gt = np.zeros(len(gt), bool); tp_det = np.zeros(len(det), bool)
    for i, j, _ in s["match"]:
        tp_gt[i] = True; tp_det[j] = True
    rq, rg, rd = rhythm_at(q, iv), rhythm_at(gt, iv), rhythm_at(det, iv)
    return {v: np.array([(rq == v).sum(), (scope & (rg == v)).sum(), (scope & tp_gt & (rg == v)).sum(),
                         (tp_det & (rd == v)).sum(), (~tp_det & (rd == v)).sum()], float)
            for v in (0, 1)}


def group_counts(items, prm, detector, af):
    """{rhythm: {dataset: (groups, array [n_groups, len(KEYS)])}}, only groups with beats in that rhythm."""
    acc = {v: {ds: defaultdict(lambda: np.zeros(len(KEYS))) for ds in DATASETS} for v in (0, 1)}
    for it in items:
        for v, c in item_counts(it, prm, detector, af[(it["dataset"], it["record"], it["lead"])]).items():
            if c[0] > 0:
                acc[v][it["dataset"]][it["group"]] += c
    return {v: {ds: (sorted(a), np.array([a[g] for g in sorted(a)]).reshape(-1, len(KEYS)))
                for ds, a in acc[v].items() if a} for v in (0, 1)}


def metrics(c):
    se = c[..., 2] / np.maximum(1, c[..., 1])
    ppv = c[..., 3] / np.maximum(1, c[..., 3] + c[..., 4])
    return dict(fp100=100 * c[..., 4] / np.maximum(1, c[..., 0]), se=se, ppv=ppv,
                f1=2 * se * ppv / np.maximum(1e-9, se + ppv))


def bootstrap(counts, B, seed=0):
    """counts: {method: {rhythm: {ds: (groups, arr)}}}; the same draws are used for every method."""
    rng = np.random.default_rng(seed)
    first = counts[next(iter(counts))]
    idx = {(v, ds): rng.integers(0, len(g), size=(B, len(g))) for v in first for ds, (g, _) in first[v].items()}
    out = {}
    for m, cm in counts.items():
        out[m] = {}
        for v, per in cm.items():
            tot = sum(a.sum(0) for _, a in per.values())
            boot = sum(a[idx[(v, ds)]].sum(1) for ds, (_, a) in per.items())
            out[m][v] = dict(point=metrics(tot), boot=metrics(boot), counts=tot,
                             groups={ds: len(g) for ds, (g, _) in per.items()})
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    B = int(args[args.index("--B") + 1]) if "--B" in args else 2000
    sps = [a for a in args if a in ("train", "val", "test")] or ["val"]
    if "test" in sps and "--final" not in args:
        sys.exit("test split is reserved for the final report: add --final")
    af = load_af()
    ml = MLDetector.load(os.path.join(splits.HERE, "models", "pwave_ml.pkl"))
    methods = {"rules": (PParams(), detect_pwaves),
               "learned": (PParams(), ml),
               # ablation: the rule detector without the PR-consistency pass (Section 3.7)
               "rules-nocons": (replace(PParams(), consistency=False), detect_pwaves)}
    lines = [f"Rhythm-stratified results, pooled over databases; patient bootstrap B={B}, 95 % intervals"]
    js = {}
    for sp in sps:
        items = splits.load(sp)
        bs = bootstrap({m: group_counts(items, prm, det, af) for m, (prm, det) in methods.items()}, B)
        js[sp] = {}
        for m, res in bs.items():
            js[sp][m] = {}
            for v in (0, 1):
                r = res[v]; c = r["counts"]
                row = {k: dict(point=float(r["point"][k]), lo=ci(r["boot"][k])[0], hi=ci(r["boot"][k])[1])
                       for k in ("fp100", "se", "ppv", "f1")}
                row.update({k: int(x) for k, x in zip(KEYS, c)}, patients=r["groups"])
                js[sp][m][RHYTHMS[v]] = row
                f = lambda k, s=1: (f"{row[k]['point']*s:5.1f} [{row[k]['lo']*s:5.1f}, {row[k]['hi']*s:5.1f}]")
                lines.append(f"[{sp:4s}] {m:12s} {RHYTHMS[v]:6s} beats={int(c[0]):6d} P in scope={int(c[1]):6d} "
                             f"patients={sum(r['groups'].values()):3d} {r['groups']}\n"
                             f"        FP/100 beats={f('fp100')}  Se={f('se', 100)}  PPV={f('ppv', 100)}  "
                             f"F1={f('f1', 100)}  TP={int(c[3])} FP={int(c[4])}")
        for a, b in (("learned", "rules"), ("rules", "rules-nocons")):
            d = bs[a][1]["boot"]["fp100"] - bs[b][1]["boot"]["fp100"]
            p = float(bs[a][1]["point"]["fp100"] - bs[b][1]["point"]["fp100"])
            lo, hi = ci(d)
            js[sp][f"AF fp100 {a}-{b}"] = dict(point=p, lo=lo, hi=hi)
            lines.append(f"[{sp:4s}] AF FP/100 beats {a}-{b} = {p:+.1f} [{lo:+.1f}, {hi:+.1f}]")
    print("\n".join(lines))
    tag = "_".join(sps)
    json.dump(js, open(os.path.join(splits.HERE, f"results_rhythm_{tag}.json"), "w"), indent=1)
    open(os.path.join(splits.HERE, f"results_rhythm_{tag}.txt"), "w").write("\n".join(lines) + "\n")
