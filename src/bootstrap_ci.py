"""Patient-level bootstrap confidence intervals for the final report.

Each lead of every record is scored once. The counts are then summed per patient group
(splits.group_of) and per database. One bootstrap replicate draws, for each database, as many
patient groups as that database has in the split, with replacement, and recomputes
Se, PPV, F1 and the macro F1 from the summed counts. Resampling is stratified by database
so that every replicate contains all four databases. Paired differences between methods
are based on the same draw for both methods.

Usage: python src/bootstrap_ci.py [val|test ...] [--final] [--B 2000]
Writes results_ci_<splits>.json and results_ci_<splits>.txt.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import splits                             # noqa: E402
from bench import DATASETS, score_item     # noqa: E402
from pwave import PParams, detect_pwaves   # noqa: E402
from pwave_ml import MLDetector           # noqa: E402

KEYS = ("tp_scope", "n_scope", "tp", "fp")
METRICS = ("se", "ppv", "f1")


def group_counts(items, prm, detector):
    """{dataset: (group names, array [n_groups, 4] of KEYS)}"""
    acc = {ds: defaultdict(lambda: np.zeros(len(KEYS))) for ds in DATASETS}
    for it in items:
        r = score_item(it, prm, detector=detector)
        acc[it["dataset"]][it["group"]] += [r[k] for k in KEYS]
    out = {}
    for ds in DATASETS:
        names = sorted(acc[ds])
        out[ds] = (names, np.array([acc[ds][g] for g in names]).reshape(-1, len(KEYS)))
    return out


def metrics(c):
    """c: [..., 4] summed counts -> dict of arrays."""
    se = c[..., 0] / np.maximum(1, c[..., 1])
    ppv = c[..., 2] / np.maximum(1, c[..., 2] + c[..., 3])
    f1 = 2 * se * ppv / np.maximum(1e-9, se + ppv)
    return dict(se=se, ppv=ppv, f1=f1)


def bootstrap(counts_by_method, B, seed=0):
    """Same resampling indices for all methods -> paired replicates.
    Returns {method: {ds or 'macro': {metric: (point, boot array)}}}."""
    rng = np.random.default_rng(seed)
    idx = {ds: rng.integers(0, len(counts_by_method[next(iter(counts_by_method))][ds][0]),
                            size=(B, len(counts_by_method[next(iter(counts_by_method))][ds][0])))
           for ds in DATASETS}
    out = {}
    for m, cnt in counts_by_method.items():
        res, f1s, f1b = {}, [], []
        for ds in DATASETS:
            c = cnt[ds][1]
            point = metrics(c.sum(0))
            boot = metrics(c[idx[ds]].sum(1))
            res[ds] = {k: (float(point[k]), boot[k]) for k in METRICS}
            res[ds]["n_groups"] = len(cnt[ds][0])
            f1s.append(point["f1"]); f1b.append(boot["f1"])
        res["macro"] = {"f1": (float(np.mean(f1s)), np.mean(f1b, axis=0))}
        out[m] = res
    return out


def ci(b, a=0.05):
    return float(np.percentile(b, 100 * a / 2)), float(np.percentile(b, 100 * (1 - a / 2)))


if __name__ == "__main__":
    args = sys.argv[1:]
    B = int(args[args.index("--B") + 1]) if "--B" in args else 2000
    sps = [a for a in args if a in ("train", "val", "test")] or ["val"]
    if "test" in sps and "--final" not in args:
        sys.exit("test split is reserved for the final report: add --final")
    ml = MLDetector.load(os.path.join(splits.HERE, "models", "pwave_ml.pkl"))
    methods = {"rules": (PParams(), detect_pwaves),
               "learned": (PParams(), ml)}
    js, lines = {}, [f"Patient bootstrap, B={B}, stratified by database, 95 % percentile intervals (values in %)"]
    for sp in sps:
        items = splits.load(sp)
        counts = {m: group_counts(items, prm, det) for m, (prm, det) in methods.items()}
        bs = bootstrap(counts, B)
        js[sp] = {}
        for m, res in bs.items():
            js[sp][m] = {}
            lines.append(f"[{sp}] {m}")
            for ds in DATASETS + ["macro"]:
                row = {}
                txt = []
                for k in (METRICS if ds != "macro" else ("f1",)):
                    p, b = res[ds][k]
                    lo, hi = ci(b)
                    row[k] = dict(point=p, lo=lo, hi=hi)
                    txt.append(f"{k.upper()}={p*100:5.1f} [{lo*100:5.1f}, {hi*100:5.1f}]")
                if ds != "macro":
                    row["n_groups"] = res[ds]["n_groups"]
                js[sp][m][ds] = row
                lines.append(f"  {ds:7s} " + "  ".join(txt) +
                             (f"  (patients {res[ds]['n_groups']})" if ds != "macro" else ""))
        for a, b in (("learned", "rules"),):
            d = bs[a]["macro"]["f1"][1] - bs[b]["macro"]["f1"][1]
            p = bs[a]["macro"]["f1"][0] - bs[b]["macro"]["f1"][0]
            lo, hi = ci(d)
            frac = float((d <= 0).mean())
            js[sp][f"{a}-{b}"] = dict(point=p, lo=lo, hi=hi, frac_le_0=frac)
            lines.append(f"[{sp}] macro F1 {a}-{b} = {p*100:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}]  "
                         f"share of replicates <= 0: {frac:.3f}")
    print("\n".join(lines))
    tag = "_".join(sps)
    json.dump(js, open(os.path.join(splits.HERE, f"results_ci_{tag}.json"), "w"), indent=1)
    open(os.path.join(splits.HERE, f"results_ci_{tag}.txt"), "w").write("\n".join(lines) + "\n")
