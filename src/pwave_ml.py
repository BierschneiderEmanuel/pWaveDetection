"""Learned P-wave candidate scorer (classical ML, no neural networks).

The rule-based detector (pwave.py) supplies, for each beat, the candidate peaks inside the
300 ms search window along with context: noise floor, template, first-pass PR latencies of
neighbouring beats, QRS shape correlation and RR intervals.
Every candidate is characterised by hand-crafted features. A gradient-boosted decision-tree
ensemble (sklearn HistGradientBoosting) or a logistic regression estimates the
probability that the candidate is the true P peak. Per beat the most probable
candidate is reported if its probability exceeds a threshold.
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pwave import PParams, detect_pwaves  # noqa: E402

FEATURES = [
    "log_prom_nf", "log_prom_ramp", "log_prom", "corr", "dlat_tmpl", "lat",
    "dlat_nb", "has_nb", "nb_frac", "rank", "prom_frac", "n_cand", "d_start", "d_end",
    "win_len", "qcorr", "log_nf_ramp", "rr_prev", "rr_next", "rr_ratio", "rr_irreg",
    "log_tmpl_prom_nf",
]


def extract(ecg, fs, prm=PParams(), rpeaks=None, nb_beats=4):
    """Return (X [n_cand, F], meta list of (beat_index, sample_index, r_sample), ctx)."""
    ctx = detect_pwaves(np.asarray(ecg, float), fs, prm, rpeaks=rpeaks, return_context=True)
    if ctx is None:
        return np.zeros((0, len(FEATURES))), [], None
    beats, cands, lat1 = ctx["beats"], ctx["cand_all"], ctx["lat1"]
    rp = ctx["rpeaks"]
    rr = np.diff(rp) / fs
    rr_med = float(np.median(rr)) if len(rr) else 1.0
    r_pos = {int(r): i for i, r in enumerate(rp)}
    r_amp = max(ctx["r_amp"], 1e-6)
    med_noise = max(float(np.median(ctx["noise"])), 1e-6)
    tp = np.log(max(ctx["tmpl_prom"], 1e-6) / med_noise)
    X, meta = [], []
    for bi, (r, ws, we) in enumerate(beats):
        cs = cands[bi]
        if not cs:
            continue
        lo, hi = max(0, bi - nb_beats), min(len(beats), bi + nb_beats + 1)
        nb = np.r_[lat1[lo:bi], lat1[bi + 1:hi]]
        n_nb = max(1, hi - lo - 1)
        nbv = nb[~np.isnan(nb)]
        nb_med = float(np.median(nbv)) if len(nbv) else np.nan
        k = r_pos.get(int(r), None)
        rr_p = rr[k - 1] if k is not None and k >= 1 else rr_med
        rr_n = rr[k] if k is not None and k < len(rr) else rr_med
        if k is not None:
            seg = rr[max(0, k - nb_beats):k + nb_beats]
            irr = float(np.median(np.abs(np.diff(seg)))) / rr_med if len(seg) > 2 else 0.0
        else:
            irr = 0.0
        proms = np.array([c[1] for c in cs])
        order = (-proms).argsort().argsort()
        for j, (idx, prom, corr, nf) in enumerate(cs):
            lat = (r - idx) / fs
            X.append([
                np.log(max(prom, 1e-6) / max(nf, 1e-6)),
                np.log(max(prom, 1e-6) / r_amp),
                np.log(max(prom, 1e-6)),
                corr,
                abs(r - idx - ctx["lat_t"]) / fs,
                lat,
                abs(r - idx - nb_med) / fs if len(nbv) else 0.3,
                float(len(nbv) > 0),
                len(nbv) / n_nb,
                order[j],
                prom / max(proms.max(), 1e-9),
                len(cs),
                (idx - ws) / fs,
                (we - idx) / fs,
                (we - ws) / fs,
                ctx["qcorr"][bi],
                np.log(max(nf, 1e-6) / r_amp),
                rr_p / rr_med,
                rr_n / rr_med,
                rr_p / max(rr_n, 1e-3),
                irr,
                tp,
            ])
            meta.append((bi, idx, r))
    return np.asarray(X, float), meta, ctx


def label(meta, fs, gt_p, tol=0.05):
    """1 when the candidate falls within tol of a reference P peak."""
    gt = np.asarray(gt_p) * fs
    if len(gt) == 0:
        return np.zeros(len(meta), int)
    return np.array([int(np.min(np.abs(gt - idx)) <= tol * fs) for _, idx, _ in meta])


def decide(prob, meta, ctx, thr, interp=True):
    """Per beat, keep the best candidate if prob >= thr. Returns sample positions (float)."""
    best = {}
    for p, (bi, idx, _) in zip(prob, meta):
        if bi not in best or p > best[bi][0]:
            best[bi] = (p, idx)
    out = []
    xp, pol = ctx["xp"], ctx["pol"]
    for p, idx in best.values():
        if p < thr:
            continue
        pos = float(idx)
        if interp and 0 < idx < len(xp) - 1:
            y0, y1, y2 = pol * xp[idx - 1], pol * xp[idx], pol * xp[idx + 1]
            den = y0 - 2 * y1 + y2
            if den < 0:
                pos += 0.5 * (y0 - y2) / den
        out.append(pos)
    return np.array(sorted(out))


class MLDetector:
    """Callable with the same signature as detect_pwaves (for bench.run)."""

    def __init__(self, model, thr=0.5):
        self.model, self.thr = model, thr

    def __call__(self, ecg, fs, prm=PParams(), rpeaks=None):
        X, meta, ctx = extract(ecg, fs, prm, rpeaks)
        if len(meta) == 0:
            return np.array([])
        return decide(self.model.predict_proba(X)[:, 1], meta, ctx, self.thr)

    def save(self, path):
        pickle.dump(dict(model=self.model, thr=self.thr, features=FEATURES), open(path, "wb"))

    @staticmethod
    def load(path):
        d = pickle.load(open(path, "rb"))
        return MLDetector(d["model"], d["thr"])
