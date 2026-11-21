"""Attach "AF" rhythm labels to the original BUT PDB JSON files (in place, keeping a backup).

Authoritative rhythm source per record (the README.txt of BUT PDB on PhysioNet states the origin segment):
  * records 1-38  : MIT-BIH Arrhythmia DB rhythm annotations (.atr, aux "(AFIB" etc.)
  * records 44-50 : Long-Term AF DB rhythm annotations
  * records 39-43 : MIT-BIH Supraventricular Arrhythmia DB provides no rhythm annotations;
                    the BUT PDB pathology table lists no AFIB for them -> verdict 0
Each AF label spans a time interval: {"name": "AF", "coordinates": {"start", "end"},
"decision": {"verdict": 1 (atrial fibrillation) | 0 (other rhythm)}}.
It is safe to run the script twice: any existing AF labels are replaced.
"""
import glob
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rhythm import af_intervals, af_labels, rhythm_changes  # noqa: E402

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW = os.path.join(HERE, "raw")
DBS = {"MIT-BIH Arrhythmia Database": "mitdb",
       "MIT-BIH Supraventricular Arrhythmia Database": "svdb",
       "Long Term AF Database": "ltafdb"}


def read_but_readme(path):
    rows = {}
    for line in open(path):
        m = re.match(r"^(\d+)\s+'([^']*)'\s+(.+?)\s+(\d+)\s+\[(\d+),(\d+)\]\s+(\d+)", line)
        if m:
            rec, path_, db, src, s0, s1, fs = m.groups()
            rows[int(rec)] = dict(pathology=path_, db=DBS[db.strip()], src=src,
                                  start=int(s0), end=int(s1), fs=int(fs))
    return rows


def main():
    meta = read_but_readme(os.path.join(RAW, "but-pdb", "README.txt"))
    backup = os.path.join(RAW, "but-pdb-json-original")
    os.makedirs(backup, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(HERE, "*_but-pdb_buotesd_annotations.json"))):
        name = os.path.basename(path)
        if not os.path.exists(os.path.join(backup, name)):
            shutil.copy2(path, os.path.join(backup, name))
        rec = int(name[:2]); m = meta[rec]
        atr = os.path.join(RAW, "src_atr", m["db"], m["src"])
        changes, _ = rhythm_changes(atr)
        af, _ = af_intervals(changes, m["start"], m["end"], m["fs"])
        if af is None:  # the source carries no rhythm annotation: fall back to the BUT PDB pathology table
            dur = (m["end"] - m["start"]) / m["fs"]
            af = [(0.0, dur, 1 if "AFIB" in m["pathology"] else 0)]
            how = "pathology table"
        else:
            how = f"{m['db']} rhythm"
        with open(path) as fh:
            d = json.load(fh)
        for snip in d["recording"]:
            for ch in snip["ch"]:
                for ann in ch["annotations"]:
                    labels = [l for l in ann["labels"] if l["name"] != "AF"]
                    for i, l in enumerate(labels):
                        l["id"] = str(i)
                    ann["labels"] = labels + af_labels(af, len(labels))
        with open(path, "w") as fh:
            json.dump(d, fh)
        print(f"{name[:2]} ({how}): " + ", ".join(f"{a:.1f}-{b:.1f}s:{v}" for a, b, v in af))


if __name__ == "__main__":
    main()
