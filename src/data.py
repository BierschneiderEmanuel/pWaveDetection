"""Loader for the BUT PDB JSON layout (ECG channel, 128 Hz).

Also reads the converted datasets under datasets/<name>/ (see convert.py).
"""
import glob
import json
import os

import numpy as np

CHANNEL = "ECG"


def find_channel(snip, channel=CHANNEL):
    return next(c for c in snip["ch"] if c["id"] == channel)


def _record_id(path):
    name = os.path.basename(path)
    name = name[:name.index("_annotations.json")]
    for suffix in ("_but-pdb_buotesd", "_pwave", "_ludb", "_qtdb"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def load_records(folder, channel=CHANNEL):
    """Yield dicts: record, lead, lead_name, fs, signal, qrs (s), p (s), range (s, s)."""
    for path in sorted(glob.glob(os.path.join(folder, "*_annotations.json"))):
        rec = _record_id(path)
        with open(path) as fh:
            d = json.load(fh)
        for lead, snip in enumerate(d["recording"]):
            ch = find_channel(snip, channel)
            ann = ch["annotations"][0]
            labels = ann["labels"]
            qrs = np.array(sorted(l["coordinates"]["peak"] for l in labels if l["name"] == "QRS"))
            p = np.array(sorted(l["coordinates"]["peak"] for l in labels if l["name"] == "P-Wave"))
            fs = float(ch["sample_rate"])
            y = np.asarray(ch["y"], dtype=float)
            rng = ann.get("annotated_range")
            rng = (rng["start"], rng["end"]) if rng else (0.0, len(y) / fs)
            yield dict(record=rec, lead=lead, lead_name=ann.get("workflow", {}).get("name", ""),
                       fs=fs, signal=y, qrs=qrs, p=p, range=rng,
                       label_scope=ann.get("label_scope", "all"))
