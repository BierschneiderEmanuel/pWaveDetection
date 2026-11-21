"""One-time evaluation on the held-out TEST split (with train/val for reference).

Compares the rule detector against the learned detector (rules + logistic-regression
candidate scorer). Nothing is tuned here.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import splits                       # noqa: E402
from bench import DATASETS, run      # noqa: E402
from pwave import PParams           # noqa: E402
from pwave_ml import MLDetector     # noqa: E402

if __name__ == "__main__":
    ml = MLDetector.load(os.path.join(splits.HERE, "models", "pwave_ml.pkl"))
    methods = {
        "rule detector": (PParams(), None),
        "learned detector": (PParams(), ml),
    }
    out = {}
    for sp in ("train", "val", "test"):
        items = splits.load(sp)
        for name, (prm, det) in methods.items():
            r = run(items, prm) if det is None else run(items, prm, detector=det)
            out[(sp, name)] = r
            print(f"[{sp:5s}] {name:32s} macro F1={r['macro_f1']*100:5.2f}  " +
                  "  ".join(f"{d}: Se={r[d]['se']*100:4.1f} PPV={r[d]['ppv']*100:4.1f} F1={r[d]['f1']*100:4.1f}"
                            for d in DATASETS), flush=True)
    json.dump({f"{sp}|{n}": {d: {k: float(v) for k, v in r[d].items()} for d in DATASETS} | {"macro_f1": r["macro_f1"]}
               for (sp, n), r in out.items()}, open(os.path.join(splits.HERE, "results_final.json"), "w"), indent=1)
