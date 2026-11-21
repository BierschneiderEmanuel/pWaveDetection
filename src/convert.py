"""Convert PhysioNet P-wave datasets (PWAVE, LUDB, QTDB) to the BUT PDB JSON layout.

Output layout per file (identical to *_but-pdb_buotesd_annotations.json):
{"recording": [ {"ch": [
    {"id": "initial", "x": [...s], "y": [...mV], "annotations": [ANN]},
    {"id": "ECG", "sample_rate": 128, "x": [...], "y": [...], "annotations": [ANN]}
]} , ... one entry per lead ]}
ANN = {"type": "MANUAL", "diagnostician": "MASTER", "version": {...},
       "labels": [{"id", "name": "P-Wave"|"QRS", "coordinates": {"peak": t_s},
                   "decision": {"verdict": 1}}],
       "workflow": {...},
       "annotated_range": {"start": s, "end": s},   # extra: region where labels are complete
       "label_scope": "all" | "beats"}              # "beats": only labelled QRS beats are annotated

Usage: python src/convert.py {pwave|ludb|qtdb|all} [--raw raw] [--out datasets]
"""
import argparse
import json
import os
from fractions import Fraction

import numpy as np
import wfdb
from scipy.signal import resample_poly

import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rhythm import af_intervals, af_labels, rhythm_changes  # noqa: E402

FS_OUT = 128
SEG_S = 120.0
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")


def resample(y, fs):
    fr = Fraction(FS_OUT, int(round(fs))).limit_denominator(1000)
    return resample_poly(y, fr.numerator, fr.denominator)


def make_ann(p_times, q_times, dataset, rng, label_scope="all", af=None):
    ev = sorted([(float(t), "P-Wave") for t in p_times] + [(float(t), "QRS") for t in q_times])
    labels = [{"id": str(i), "name": n, "coordinates": {"peak": t},
               "decision": {"verdict": 1}} for i, (t, n) in enumerate(ev)]
    if af:  # AF rhythm intervals: verdict 1 = atrial fibrillation, 0 = other rhythm
        labels += af_labels(af, len(labels))
    return {"type": "MANUAL", "diagnostician": "MASTER",
            "version": {"timestamp": 0, "tool": f"convert.py/{dataset}"},
            "labels": labels,
            "workflow": {"name": dataset, "version": "1"},
            "annotated_range": {"start": float(rng[0]), "end": float(rng[1])},
            "label_scope": label_scope}


def make_snippet(y, fs, p_times, q_times, dataset, rng, label_scope="all", af=None):
    y = np.asarray(y, float)
    y = np.nan_to_num(y)
    ann = make_ann(p_times, q_times, dataset, rng, label_scope, af)
    yr = resample(y, fs)
    n_out = int(round(len(y) / fs * FS_OUT))
    yr = yr[:n_out]
    r6 = lambda a: [round(float(v), 6) for v in a]
    return {"ch": [
        {"id": "initial", "x": r6(np.arange(len(y)) / fs), "y": r6(y), "annotations": [ann]},
        {"id": "ECG", "sample_rate": FS_OUT, "x": r6(np.arange(len(yr)) / FS_OUT), "y": r6(yr),
         "annotations": [ann]},
    ]}


def write(out_dir, name, snippets):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name), "w") as fh:
        json.dump({"recording": snippets}, fh, separators=(",", ":"))


def segments(p, q, fs, n, seg_s=SEG_S):
    """Split a long record into pieces of seg_s seconds; yield (s0, s1, p_rel, q_rel)."""
    seg = int(round(seg_s * fs))
    for k, s0 in enumerate(range(0, n - seg + 1, seg)):
        s1 = s0 + seg
        pm = (p >= s0) & (p < s1); qm = (q >= s0) & (q < s1)
        yield k, s0, s1, (p[pm] - s0) / fs, (q[qm] - s0) / fs


# ---------------------------------------------------------------- PWAVE (MIT-BIH)
def convert_pwave(raw, out):
    d = os.path.join(raw, "pwave")
    recs = sorted(f[:-len(".pwave")] for f in os.listdir(d) if f.endswith(".pwave"))
    n_files = 0
    for rec in recs:
        sig = wfdb.rdrecord(os.path.join(d, rec))
        fs = sig.fs
        p = wfdb.rdann(os.path.join(d, rec), "pwave").sample
        a = wfdb.rdann(os.path.join(raw, "mitdb_atr", rec), "atr")
        q = np.array([s for s, sy in zip(a.sample, a.symbol) if sy in BEAT_SYMBOLS])
        changes, _ = rhythm_changes(os.path.join(raw, "mitdb_atr", rec))
        for k, s0, s1, pr, qr in segments(np.asarray(p), q, fs, sig.sig_len):
            af, _ = af_intervals(changes, s0, s1, fs)
            snips = [make_snippet(sig.p_signal[s0:s1, ch], fs, pr, qr, "pwave_mitdb", (0.0, SEG_S), af=af)
                     for ch in range(sig.n_sig)]
            write(os.path.join(out, "pwave"), f"{rec}_{k:02d}_pwave_annotations.json", snips)
            n_files += 1
    print("pwave:", n_files, "files")


# ---------------------------------------------------------------- LUDB
def convert_ludb(raw, out):
    d = os.path.join(raw, "ludb")
    # the archive extracts to lobachevsky-university-electrocardiography-database-1.0.1/data
    for root, dirs, files in os.walk(d):
        if "1.hea" in files:
            d = root; break
    recs = sorted({f[:-4] for f in os.listdir(d) if f.endswith(".hea")}, key=int)
    import csv
    meta_path = os.path.join(os.path.dirname(d), "ludb.csv")
    rhythms = {row["ID"].strip(): row["Rhythms"] for row in csv.DictReader(open(meta_path))}
    for rec in recs:
        sig = wfdb.rdrecord(os.path.join(d, rec))
        fs = sig.fs
        # record-level rhythm from ludb.csv, applied across the whole 10 s record
        is_af = "atrial fibrillation" in rhythms[rec].lower()
        af = [(0.0, sig.sig_len / fs, 1 if is_af else 0)]
        snips = []
        for ch, lead in enumerate(sig.sig_name):
            a = wfdb.rdann(os.path.join(d, rec), lead.lower())
            sm, sy = np.asarray(a.sample), np.asarray(a.symbol)
            p = sm[sy == "p"] / fs; q = sm[sy == "N"] / fs
            # labelled region: from the first to the last wave boundary of this lead
            rng = (sm.min() / fs, sm.max() / fs) if len(sm) else (0.0, 0.0)
            snips.append(make_snippet(sig.p_signal[:, ch], fs, p, q, f"ludb_{lead.lower()}", rng, af=af))
        write(os.path.join(out, "ludb"), f"{int(rec):03d}_ludb_annotations.json", snips)
    print("ludb:", len(recs), "files")


# ---------------------------------------------------------------- QTDB
def _mitdb_offset(qt_atr, mit_atr, qt_fs=250.0, mit_fs=360.0):
    """Offset [s] of a QTDB excerpt within its MIT-BIH source record (via beat matching).
    Sampling rates are supplied explicitly because the .atr files ship without headers."""
    q = wfdb.rdann(qt_atr, "atr"); m = wfdb.rdann(mit_atr, "atr")
    qt = np.array([s for s, y in zip(q.sample, q.symbol) if y in BEAT_SYMBOLS]) / qt_fs
    mt = np.array([s for s, y in zip(m.sample, m.symbol) if y in BEAT_SYMBOLS]) / mit_fs
    n = min(50, len(qt))
    for off in mt - qt[0]:
        if off < 0:
            continue
        idx = np.clip(np.searchsorted(mt, qt[:n] + off), 1, len(mt) - 1)
        d = np.minimum(np.abs(mt[idx] - qt[:n] - off), np.abs(mt[idx - 1] - qt[:n] - off))
        if np.mean(d < 0.02) > 0.98:
            return float(off), mit_fs
    return None, None


def qtdb_af(raw, rec, lo, hi, fs):
    """AF intervals for the cropped QTDB record, or None when no rhythm annotation exists."""
    base = os.path.join(raw, "qtdb", rec)
    mit = os.path.join(raw, "src_atr", "mitdb", rec[3:])
    if len(rec) == 6 and rec[3] in "12" and os.path.exists(mit + ".atr") and os.path.exists(base + ".atr"):
        off, mfs = _mitdb_offset(base, mit, fs, 360.0)
        if off is not None:
            changes, _ = rhythm_changes(mit)
            s0 = int(round((off + lo / fs) * mfs)); s1 = int(round((off + hi / fs) * mfs))
            af, _ = af_intervals(changes, s0, s1, mfs)
            return af
    if os.path.exists(base + ".atr"):
        changes, _ = rhythm_changes(base)
        af, _ = af_intervals(changes, lo, hi, fs)
        return af
    return None


def convert_qtdb(raw, out):
    d = os.path.join(raw, "qtdb")
    recs = [l.strip() for l in open(os.path.join(d, "RECORDS")) if l.strip()]
    n_files = 0
    for rec in recs:
        base = os.path.join(d, rec)
        if not os.path.exists(base + ".q1c"):
            continue
        sig = wfdb.rdrecord(base)
        fs = sig.fs
        a = wfdb.rdann(base, "q1c")
        sm, sy = np.asarray(a.sample), np.asarray(a.symbol)
        p = sm[sy == "p"]
        q = sm[sy == "N"]
        # only the manually annotated beats carry labels (label_scope="beats"):
        # P labels are complete only within the 300 ms preceding these QRS labels
        # crop to the manually annotated region (+-2 s context), dropping any labels outside
        lo = max(0, int(sm.min() - 2 * fs)); hi = min(sig.sig_len, int(sm.max() + 2 * fs))
        pr = (p[(p >= lo) & (p < hi)] - lo) / fs
        qr = (q[(q >= lo) & (q < hi)] - lo) / fs
        rng = ((sm.min() - lo) / fs, (sm.max() - lo) / fs)
        af = qtdb_af(raw, rec, lo, hi, fs)
        snips = [make_snippet(sig.p_signal[lo:hi, ch], fs, pr, qr, "qtdb_q1c", rng, "beats", af=af)
                 for ch in range(sig.n_sig)]
        write(os.path.join(out, "qtdb"), f"{rec}_qtdb_annotations.json", snips)
        n_files += 1
    print("qtdb:", n_files, "files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["pwave", "ludb", "qtdb", "all"])
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    ap.add_argument("--raw", default=os.path.join(here, "raw"))
    ap.add_argument("--out", default=os.path.join(here, "datasets"))
    a = ap.parse_args()
    for name, fn in [("pwave", convert_pwave), ("ludb", convert_ludb), ("qtdb", convert_qtdb)]:
        if a.which in (name, "all"):
            fn(a.raw, a.out)
