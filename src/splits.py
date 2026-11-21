"""Patient-level train / validation / test split across all datasets.

Records that originate from the same source patient share a group and are always placed in
the same split. BUT PDB, PWAVE and QT Database all contain excerpts of MIT-BIH records.
Groups are assigned by a fixed hash: 60 % train, 20 % validation, 20 % test.
The test split is only used for the final report.

Also builds a compact cache (.cache_bench.pkl) of all ECG signals, labels and
Elgendi R peaks so that tuning does not have to re-parse the JSON files.
"""
import hashlib
import json
import os
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_records          # noqa: E402
from elgendi import elgendi_rpeaks     # noqa: E402

HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SPLIT_FILE = os.path.join(HERE, "splits.json")
CACHE = os.path.join(HERE, ".cache_bench.pkl")
DATASETS = {"butpdb": HERE, "pwave": os.path.join(HERE, "datasets", "pwave"),
            "ludb": os.path.join(HERE, "datasets", "ludb"), "qtdb": os.path.join(HERE, "datasets", "qtdb")}


def _but_sources():
    db = {"MIT-BIH Arrhythmia Database": "mitdb", "MIT-BIH Supraventricular Arrhythmia Database": "svdb",
          "Long Term AF Database": "ltafdb"}
    out = {}
    for line in open(os.path.join(HERE, "raw", "but-pdb", "README.txt")):
        m = re.match(r"^(\d+)\s+'([^']*)'\s+(.+?)\s+(\d+)\s+\[", line)
        if m:
            out[f"{int(m.group(1)):02d}"] = f"{db[m.group(3).strip()]}:{m.group(4)}"
    return out


def group_of(dataset, record, but_src):
    if dataset == "butpdb":
        return but_src[record]
    if dataset == "pwave":
        return "mitdb:" + record.split("_")[0]
    if dataset == "qtdb":
        m = re.match(r"^sel([128]\d\d)$", record)
        if m:
            return ("svdb:" if m.group(1)[0] == "8" else "mitdb:") + m.group(1)
        return "qtdb:" + record
    return f"{dataset}:{record}"


def split_of(group):
    u = int(hashlib.sha1(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if u < 0.6 else ("val" if u < 0.8 else "test")


def build():
    but_src = _but_sources()
    items, splits = [], {}
    for ds, folder in DATASETS.items():
        for r in load_records(folder):
            g = group_of(ds, r["record"], but_src)
            sp = split_of(g)
            splits[f"{ds}/{r['record']}"] = {"group": g, "split": sp}
            items.append(dict(dataset=ds, record=r["record"], lead=r["lead"], group=g, split=sp,
                              fs=r["fs"], signal=r["signal"].astype(np.float32), qrs=r["qrs"], p=r["p"],
                              range=r["range"], label_scope=r["label_scope"],
                              rpeaks=elgendi_rpeaks(r["signal"], r["fs"])))
        print(ds, "done", flush=True)
    json.dump(splits, open(SPLIT_FILE, "w"), indent=1, sort_keys=True)
    pickle.dump(items, open(CACHE, "wb"), protocol=4)
    return items


def load(split=None, dataset=None):
    items = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else build()
    return [it for it in items if (split is None or it["split"] == split)
            and (dataset is None or it["dataset"] == dataset)]


if __name__ == "__main__":
    items = build()
    import collections
    c = collections.Counter((it["dataset"], it["split"]) for it in items if it["lead"] == 0)
    pc = collections.Counter()
    for it in items:
        pc[(it["dataset"], it["split"])] += len(it["p"])
    for ds in DATASETS:
        print(ds, {sp: (c[(ds, sp)], pc[(ds, sp)]) for sp in ("train", "val", "test")}, "(records, P labels)")
    leak = collections.defaultdict(set)
    for it in items:
        leak[it["group"]].add(it["split"])
    print("groups:", len(leak), "groups in >1 split:", sum(len(v) > 1 for v in leak.values()))
