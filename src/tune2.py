"""Coordinate-descent tuning on the TRAIN split (macro F1 over datasets).

Validation is only evaluated at the end of each round, to pick the round to keep;
the test split is never touched here.
"""
import json
import os
import sys
from dataclasses import asdict, replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import splits                      # noqa: E402
from bench import run, fmt          # noqa: E402
from pwave import PParams          # noqa: E402

START = dict(min_amp=0.0, min_amp_rel=0.01, amp_cap=0.02, qrs_corr_min=0.7, consistency=True,
             relax=0.6, k_isolated=2.0, interp=True)
MIN_GAIN = 1e-3   # a change is kept only if it improves train macro F1 by >= 0.1 points (noise-level gains are rejected)
GRID = {
    "min_amp_rel": [0.005, 0.01, 0.015, 0.02],
    "amp_cap": [0.01, 0.015, 0.02, 0.03, 1e9],
    "k_noise": [0.5, 1.0, 1.5, 2.0, 3.0],
    "relax": [0.4, 0.5, 0.6, 0.8, 1.0],
    "min_corr": [0.4, 0.5, 0.6, 0.7, 0.8],
    "k_isolated": [1.5, 2.0, 3.0, 5.0, 99.0],
    "nb_tol": [0.025, 0.04, 0.06],
    "nb_sigma": [0.015, 0.025, 0.04],
    "nb_beats": [2, 4, 6],
    "nb_min": [1, 2, 3],
    "qrs_corr_min": [-1.0, 0.5, 0.7, 0.8],
    "lat_sigma": [0.02, 0.03, 0.045],
    "p_lp": [10.0, 12.0, 15.0, 20.0],
    "guard": [0.008, 0.016, 0.03],
    "neg_bias": [1.5, 3.0, 5.0],
    "lf_weight": [0.0, 0.5, 1.0],
    "onset_frac": [0.05, 0.1, 0.15],
}

if __name__ == "__main__":
    tr, va = splits.load("train"), splits.load("val")
    prm = replace(PParams(), **START)
    best = run(tr, prm)["macro_f1"]
    print(f"start train={best*100:.2f}", flush=True)
    history = []
    for rnd in range(2):
        for k, vals in GRID.items():
            for v in vals:
                if getattr(prm, k) == v:
                    continue
                cand = replace(prm, **{k: v})
                s = run(tr, cand)["macro_f1"]
                if s > best + MIN_GAIN:
                    best, prm = s, cand
            print(f"round {rnd} {k}={getattr(prm, k)} train={best*100:.2f}", flush=True)
        rv = run(va, prm)
        print(f"== round {rnd}: train={best*100:.2f} val={rv['macro_f1']*100:.2f}", flush=True)
        history.append(dict(round=rnd, params=asdict(prm), train=best, val=rv["macro_f1"]))
    keep = max(history, key=lambda h: h["val"])
    json.dump(dict(chosen=keep, history=history), open(os.path.join(splits.HERE, "tuned_params.json"), "w"), indent=1)
    print("chosen round", keep["round"], json.dumps(keep["params"]))
    print("[train]\n" + fmt(run(tr, PParams(**keep["params"]))))
    print("[val]\n" + fmt(run(va, PParams(**keep["params"]))))
