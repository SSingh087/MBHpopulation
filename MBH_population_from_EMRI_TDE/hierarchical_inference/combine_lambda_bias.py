#!/usr/bin/env python
"""Merge the per-job npz files from test_lambda_bias.py into
lambda_bias_test/summary_{A,B}.txt and lambda_bias_test/profile_pop{A,B}.png."""
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import SCENARIO_KEY, MODEL_LABELS

parts = {}
for p in glob.glob("lambda_bias_test/parts/*.npz"):
    d = np.load(p, allow_pickle=True)
    parts[(str(d["pop"]), str(d["scen"]), str(d["survey"]))] = d


def best(L, XC, LAM):
    """(xc, lam) of the maximum for pop A, lam for pop B; plus the lam profile (max over xc for A)."""
    if XC.size:
        i, j = np.unravel_index(L.argmax(), L.shape)
        prof = L.max(0)
        return (XC[i], LAM[j]), prof - prof.max()
    return (LAM[L.argmax()],), L - L.max()


def fmt(b):
    return "(" + ",".join(f"{v:.2f}" for v in b) + ")"


for pop in "AB":
    lines = [f"===== Population {pop} =====",
             f"{'model':5s}{'survey':6s}{'N':>7s}{'<sig>':>6s} | {'truth':>13s} | {'as_built':>13s} | "
             f"{'exact':>13s} | {'scattered mean+-std (reps)':>30s}"]
    fig, axes = plt.subplots(2, 8, figsize=(26, 6), sharey=True)
    for c, key in enumerate(sorted(SCENARIO_KEY, key=lambda t: int(MODEL_LABELS[t][1:]))):
        scen, tag = "_".join(key), MODEL_LABELS[key]
        for r, survey in enumerate(("ZTF", "LSST")):
            d = parts.get((pop, scen, survey))
            if d is None:
                lines.append(f"{tag:5s}{survey:6s}  [missing]")
                continue
            XC, LAM = d["XC"], d["LAM"]
            b_as, p_as = best(d["L_as"], XC, LAM)
            b_ex, p_ex = best(d["L_ex"], XC, LAM)
            sc = [best(L, XC, LAM) for L in d["L_sc"]]
            b_sc = np.array([s[0] for s in sc])
            mean, std = b_sc.mean(0), b_sc.std(0)
            sc_txt = "(" + ",".join(f"{m:.2f}±{s:.2f}" for m, s in zip(mean, std)) + ")"
            lines.append(f"{tag:5s}{survey:6s}{int(d['N']):7d}{float(d['sig_mean']):6.2f} | {fmt(d['truth']):>13s} | "
                         f"{fmt(b_as):>13s} | {fmt(b_ex):>13s} | {sc_txt:>30s}")
            ax = axes[r, c]
            ax.plot(LAM, p_as, color="C3", label="as built")
            ax.plot(LAM, p_ex, color="C2", label="exact")
            ax.plot(LAM, sc[0][1], color="C0", label="scattered")
            ax.axvline(d["truth"][-1], color="k", ls="--")
            ax.set_ylim(-60, 2)
            ax.set_title(f"{tag} {survey}", fontsize=10)
    axes[0, 0].legend(fontsize=8)
    fig.supxlabel("lam_schechter" if pop == "A" else "lambda_M")
    fig.supylabel("log L - max  (A: max over xc)")
    fig.savefig(f"lambda_bias_test/profile_pop{pop}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    open(f"lambda_bias_test/summary_{pop}.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
