#!/usr/bin/env python
"""
Violin plots for EMRI, EMRI+TDE (ZTF), EMRI+TDE (LSST), EMRI+TDE (ALL) with a ZOOM column.

Per hyperparameter row:
  left  : full range, raw value (black dash = injected truth)  -> shows prior-limited cases
  right : the same violins as DELTA = value - truth, y-limits set from the TIGHT cases
          (those with LSST), so they are readable; wider violins are clipped there.
          Delta makes one common zoom window valid for all 8 scenarios even though truths differ.

Usage: python plots/plot_violin_zoom.py --GALAXIES 100000 --POPULATION A [--ZOOM_CASES "EMRI+TDE (LSST),EMRI+TDE (ALL)"] [--QZOOM 0.5]
Output: {base}/PLOTS/violin-posteriors/{GALAXIES}_pop_{POP}_violin_zoom.png/.pdf
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, os
import numpy as np, pandas as pd
import matplotlib, matplotlib.pyplot as plt, seaborn as sns
from utils import load_available_combos, get_latex_labels, get_true_x_flat, SOURCE_SURVEY_COMBOS, SCENARIO_KEY, MODEL_LABELS

matplotlib.rc('font', family='serif', serif=['Computer Modern'], size=18)
matplotlib.rc('text', usetex=True)
ap = argparse.ArgumentParser()
ap.add_argument('--GALAXIES', required=True); ap.add_argument('--POPULATION', choices=['A', 'B'], required=True)
ap.add_argument("--CASES", default="EMRI,EMRI+TDE (ZTF),EMRI+TDE (LSST),EMRI+TDE (ALL)")
ap.add_argument("--ZOOM_CASES", default="EMRI+TDE (LSST),EMRI+TDE (ALL)", help="cases that define the zoom window")
ap.add_argument("--QZOOM", type=float, default=0.5, help="percentile (each tail) of the zoom cases' delta used for the window")
ap.add_argument("--ASINH", type=int, default=1, help="1 = right panel uses an asinh y-axis (linear inside the zoom window, log-like outside) so nothing is clipped; 0 = hard zoom")
ap.add_argument("--PAD", type=float, default=0.25, help="fractional padding of the zoom window")
a = ap.parse_args()
keep = [s.strip() for s in a.CASES.split(",")]; zoom_cases = {s.strip() for s in a.ZOOM_CASES.split(",")}
base = f"./data/pseudo_inference_data/{a.GALAXIES}"
outdir = f"{base}/PLOTS/violin-posteriors"; os.makedirs(outdir, exist_ok=True)
labels = get_latex_labels(a.POPULATION)
case_order = [l for (_s, _v, l, _c) in SOURCE_SURVEY_COMBOS if l in keep]
pal = {c: col for (_s, _v, c, col) in SOURCE_SURVEY_COMBOS}
tuples = sorted(SCENARIO_KEY, key=lambda t: int(MODEL_LABELS[t][1:])); scen_order = [MODEL_LABELS[t] for t in tuples]

recs = []
for t in tuples:
    tag = MODEL_LABELS[t]; tr = get_true_x_flat(a.POPULATION, *t)
    for c in load_available_combos(base, a.POPULATION, *t):
        if c["label"] not in keep: continue
        for k, s in c["samples"].items():
            if k not in tr or k not in labels: continue   # e.g. logP/logL have no injected truth
            recs.append(pd.DataFrame({"value": s, "delta": s - tr[k], "truth": tr[k], "param": k, "scenario": tag, "case": c["label"]}))
df = pd.concat(recs, ignore_index=True)
scens = [s for s in scen_order if (df.scenario == s).any()]
cases = [c for c in case_order if (df.case == c).any()]
params = [p for p in labels if (df.param == p).any()]
df["scenario"] = pd.Categorical(df["scenario"], scens, ordered=True)
df["case"] = pd.Categorical(df["case"], cases, ordered=True)

fig, axes = plt.subplots(len(params), 2, figsize=(2.0 * len(scens) * 2 + 2, 3.2 * len(params)), squeeze=False)
for r, p in enumerate(params):
    sub = df[df.param == p]; rc = [c for c in cases if (sub.case == c).any()]
    zs = sub[sub.case.isin(zoom_cases)]["delta"]
    if len(zs):
        lo, hi = np.percentile(zs, [a.QZOOM, 100 - a.QZOOM]); lo, hi = min(lo, 0.0), max(hi, 0.0)
        pad = a.PAD * (hi - lo if hi > lo else 1.0); zlim = (lo - pad, hi + pad)
    else:
        zlim = None
    sc = max(abs(zlim[0]), abs(zlim[1])) / 3 if zlim else 1.0     # asinh linear scale
    sub = sub.assign(asinh=np.arcsinh(sub["delta"] / sc))
    for col, (yc, ttl) in enumerate((("value", "full range"), ("asinh" if a.ASINH else "delta", "asinh zoom: value $-$ truth" if a.ASINH else "zoom: value $-$ truth"))):
        ax = axes[r, col]
        sns.violinplot(data=sub, x="scenario", y=yc, hue="case", ax=ax, cut=0, order=scens, hue_order=rc, palette=pal,
                       dodge=True, linewidth=0.8, saturation=0.9, inner="quart", density_norm="width", common_norm=False)
        for i, s in enumerate(scens):
            tv = sub[sub.scenario == s]["truth"]
            if len(tv):
                y = tv.iloc[0] if yc == "value" else 0.0   # delta=0 and asinh(0)=0
                ax.plot([i - .4, i + .4], [y, y], "k--", lw=2, zorder=5)
        if col == 1 and a.ASINH:
            rng = max(abs(zlim[0]), abs(zlim[1])) if zlim else sc
            far = max(np.abs(sub["delta"]).max(), rng)
            tk = [0.0] + [sg * v for v in (rng / 3, rng, far) for sg in (-1, 1)]
            tk = sorted(set(tk)); ax.set_yticks(np.arcsinh(np.array(tk) / sc))
            ax.set_yticklabels([f"{v:.2g}" for v in tk], fontsize=11)
            ax.axhspan(np.arcsinh(zlim[0] / sc), np.arcsinh(zlim[1] / sc), color="grey", alpha=0.08, zorder=0) if zlim else None
        elif col == 1 and zlim: ax.set_ylim(*zlim)
        ax.set_xlabel(""); ax.set_ylabel(labels[p] if col == 0 else r"$\Delta$", fontsize=16)
        ax.tick_params(axis='x', labelsize=14); ax.tick_params(axis='y', labelsize=13)
        if r == 0: ax.set_title(ttl, fontsize=16)
        if not (r == 0 and col == 1): ax.get_legend().remove()
        else: ax.legend(title="Case", loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=12, title_fontsize=13, frameon=False)
plt.subplots_adjust(hspace=0.6, wspace=0.25)
fn = f"{outdir}/{a.GALAXIES}_pop_{a.POPULATION}_violin_zoom"
fig.savefig(fn + ".png", dpi=200, bbox_inches="tight"); fig.savefig(fn + ".pdf", bbox_inches="tight"); print("Saved", fn)
