import numpy as np


def match(ref, det, tol):
    """Greedy one-to-one matching by smallest distance. Times in seconds.
    Returns list of (i_ref, j_det, det-ref)."""
    ref = np.asarray(ref); det = np.asarray(det)
    if len(ref) == 0 or len(det) == 0:
        return []
    pairs = []
    for i, r in enumerate(ref):
        lo, hi = np.searchsorted(det, [r - tol, r + tol], side="left")
        for j in range(lo, min(hi + 1, len(det))):
            d = det[j] - r
            if abs(d) <= tol:
                pairs.append((abs(d), i, j, d))
    pairs.sort()
    used_r, used_d, out = set(), set(), []
    for _, i, j, d in pairs:
        if i in used_r or j in used_d:
            continue
        used_r.add(i); used_d.add(j); out.append((i, j, d))
    return out
