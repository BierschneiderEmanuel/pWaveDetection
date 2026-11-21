"""Elgendi QRS / R-peak detector.

M. Elgendi, "Fast QRS Detection with an Optimized Knowledge-Based Method:
Evaluation on 11 Standard ECG Databases", PLoS ONE 8(9): e73557, 2013.

Steps: 8-20 Hz Butterworth band-pass -> squaring -> two moving averages
(W1 = 97 ms ~ QRS width, W2 = 611 ms ~ beat width) -> blocks of interest
where MA_qrs > MA_beat + beta*mean(y) -> keep only blocks wider than W1 ->
R peak = extremum of the ECG inside each block.
"""
import numpy as np
from scipy.signal import butter, filtfilt


def _moving_average(x, n):
    n = max(1, int(round(n)))
    k = np.ones(n) / n
    return np.convolve(x, k, mode="same")


def elgendi_rpeaks(ecg, fs, w1=0.097, w2=0.611, beta=0.08, refractory=0.25):
    """Return R-peak sample indices."""
    b, a = butter(3, [8.0 / (fs / 2), 20.0 / (fs / 2)], btype="band")
    f = filtfilt(b, a, ecg)
    y = f ** 2
    ma_qrs = _moving_average(y, w1 * fs)
    ma_beat = _moving_average(y, w2 * fs)
    thr = ma_beat + beta * np.mean(y)
    boi = ma_qrs > thr

    # signal on which the R peak is located: a lightly band-passed ECG
    b2, a2 = butter(2, [0.5 / (fs / 2), 40.0 / (fs / 2)], btype="band")
    xr = filtfilt(b2, a2, ecg)

    edges = np.diff(boi.astype(int), prepend=0, append=0)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    min_w = int(round(w1 * fs))
    peaks = []
    for s, e in zip(starts, ends):
        if e - s < min_w:
            continue
        seg = np.abs(xr[s:e])
        peaks.append(s + int(np.argmax(seg)))
    peaks = np.array(peaks, dtype=int)

    # dominant QRS polarity: re-locate the extremum carrying the majority sign
    if len(peaks):
        sign = 1.0 if np.median(xr[peaks]) >= 0 else -1.0
        half = int(round(0.05 * fs))
        out = []
        for p in peaks:
            lo, hi = max(0, p - half), min(len(xr), p + half + 1)
            seg = sign * xr[lo:hi]
            # keep the absolute maximum if the majority-sign extremum is negligible
            cand = lo + int(np.argmax(seg))
            out.append(cand if sign * xr[cand] > 0.3 * abs(xr[p]) else p)
        peaks = np.array(out, dtype=int)

    # refractory period: for two peaks closer than `refractory`, keep the larger
    keep = []
    for p in peaks:
        if keep and p - keep[-1] < refractory * fs:
            if abs(xr[p]) > abs(xr[keep[-1]]):
                keep[-1] = p
        else:
            keep.append(p)
    return np.unique(np.array(keep, dtype=int))
