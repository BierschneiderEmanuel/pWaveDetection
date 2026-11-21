"""Evaluate parameter variants on a split: python src/probe.py train "min_amp=0,min_amp_rel=0.02" ..."""
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import splits                      # noqa: E402
from bench import run              # noqa: E402
from pwave import PParams          # noqa: E402


def parse(spec, base):
    kw = {}
    for kv in filter(None, spec.split(",")):
        k, v = kv.split("=")
        t = type(getattr(base, k))
        kw[k] = (v.lower() in ("1", "true")) if t is bool else t(v)
    return replace(base, **kw)


if __name__ == "__main__":
    sp = sys.argv[1]
    assert sp in ("train", "val"), "probing on the test split is not allowed"
    items = splits.load(sp)
    for spec in sys.argv[2:]:
        r = run(items, parse(spec, PParams()))
        print(f"{spec or 'baseline':60s} macro={r['macro_f1']*100:.2f}  " +
              "  ".join(f"{d}={r[d]['f1']*100:.1f}" for d in ("butpdb", "pwave", "ludb", "qtdb")), flush=True)
