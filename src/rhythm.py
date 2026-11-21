"""Atrial-fibrillation intervals extracted from WFDB rhythm annotations ('+' with aux '(AFIB', ...)."""
import numpy as np
import wfdb

AF_CODES = {"(AFIB"}            # atrial fibrillation only; flutter '(AFL' counts as non-AF


def rhythm_changes(atr_path):
    """Return a list of (sample, rhythm_code) pairs from a WFDB .atr file."""
    a = wfdb.rdann(atr_path, "atr")
    out = []
    for s, sy, aux in zip(a.sample, a.symbol, a.aux_note):
        aux = (aux or "").strip().rstrip("\x00")
        if sy == "+" and aux.startswith("("):
            out.append((int(s), aux))
    return out, a.fs


def af_intervals(changes, start, end, fs):
    """AF / non-AF intervals in seconds relative to `start` (samples) for [start, end).
    Returns a list of (t0, t1, verdict) pairs and the rhythm codes seen, or None if no rhythm info exists."""
    if not changes:
        return None, set()
    # the rhythm in force at `start`
    cur = None
    for s, code in changes:
        if s <= start:
            cur = code
        else:
            break
    pts = [(start, cur)] + [(s, c) for s, c in changes if start < s < end]
    if pts[0][1] is None:
        # rhythm before the first marker is unknown: label only from the first marker onward
        pts = pts[1:]
        if not pts:
            return None, set()
    codes = {c for _, c in pts}
    iv = []
    for i, (s, c) in enumerate(pts):
        e = pts[i + 1][0] if i + 1 < len(pts) else end
        v = 1 if c in AF_CODES else 0
        t0, t1 = (s - start) / fs, (e - start) / fs
        if iv and iv[-1][2] == v:
            iv[-1] = (iv[-1][0], t1, v)          # merge consecutive equal verdicts
        else:
            iv.append((t0, t1, v))
    return iv, codes


def af_labels(intervals, first_id):
    return [{"id": str(first_id + i), "name": "AF",
             "coordinates": {"start": round(t0, 6), "end": round(t1, 6)},
             "decision": {"verdict": int(v)}} for i, (t0, t1, v) in enumerate(intervals)]
