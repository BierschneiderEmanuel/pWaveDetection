"""Plot detections for selected leads: python src/plot_examples.py out.png 05:0,14:0 [t0] [data_dir]"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from data import load_records  # noqa: E402
from pwave import PParams, detect_pwaves  # noqa: E402

out, sel = sys.argv[1], sys.argv[2]
t0 = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
want = [(a.split(":")[0], int(a.split(":")[1])) for a in sel.split(",")]
data_dir = sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.path.dirname(__file__), "..")
recs = {(r["record"], r["lead"]): r for r in load_records(data_dir)
        if (r["record"], r["lead"]) in want}
fig, axs = plt.subplots(len(want), 1, figsize=(16, 3.2 * len(want)), squeeze=False)
for ax, k in zip(axs[:, 0], want):
    r = recs[k]; fs = r["fs"]
    det, d = detect_pwaves(r["signal"], fs, PParams(), return_debug=True)
    xf, xp = d["xf"], d["xp"]; t = np.arange(len(xf)) / fs
    s = slice(int(t0 * fs), int((t0 + 8) * fs))
    ax.plot(t[s], xf[s], c="0.6", lw=.7, label="ECG 0.5-40 Hz")
    ax.plot(t[s], xp[s], c="b", lw=1, label="P band (QRS blanked)")
    for rr, ws, we in d["beats"]:
        if t0 < rr / fs < t0 + 8:
            ax.axvspan(ws / fs, we / fs, color="y", alpha=.25)
    rp = d["rpeaks"]; rp = rp[(rp >= s.start) & (rp < s.stop)]
    ax.plot(rp / fs, xf[rp], "ko", label="R (Elgendi)")
    p = r["p"][(r["p"] > t0) & (r["p"] < t0 + 8)]
    ax.plot(p, xp[np.round(p * fs).astype(int)], "m^", ms=9, label="P label (P-Wave)")
    dd = det[(det >= s.start) & (det < s.stop)]
    ax.plot(dd / fs, xp[dd], "rx", ms=10, mew=2, label="P detected")
    ax.set_title(f"record {k[0]} lead {k[1]}"); ax.legend(loc="upper right", fontsize=7)
plt.tight_layout(); plt.savefig(out, dpi=70)
