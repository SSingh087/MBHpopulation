#!/usr/bin/env python
"""
Alternative ways to show the hierarchical-inference results across the 8
universe scenarios (M1..M8), 7 source/survey cases and the 2 phenomenological
population models (A: Schechter+Beta-mixtures, B: power-law+Gaussian-mixtures).

Why: in the violin grid most rows (the Beta/Gaussian shape parameters) are
prior-dominated and mutually degenerate, so the violins just redraw the prior
box and the eye cannot find the signal.  These variants instead show

  V1  information gain        -- which hyperparameters are constrained at all
  V2  bias / pull heatmap     -- is the truth recovered where it IS constrained
  V3  pull plot               -- every hyperparameter on one common axis
  V4  observable-space bands  -- inferred population pdf vs truth (A | B)
  V5  derived summaries       -- <lgM>, <a>, <e>, lambda_z : directly comparable A vs B
  V6  derived-summary scorecard (z-score matrix)

Usage:
    python plots/plot_variants.py --GALAXIES 100000
Outputs go to ./plots/plot_variants/
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

from utils import (SOURCE_SURVEY_COMBOS, MODEL_LABELS, SCENARIO_KEY,
                   load_available_combos, get_latex_labels, get_true_x_flat)

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES", default="100000")
parser.add_argument("--NTHIN", type=int, default=1500, help="posterior draws used for derived quantities")
args = parser.parse_args()

matplotlib.rcParams.update({"font.size": 11, "mathtext.fontset": "cm", "axes.spines.top": False,
                            "axes.spines.right": False})
base_dir = f"./data/pseudo_inference_data/{args.GALAXIES}"
outdir = "plots/plot_variants"
os.makedirs(outdir, exist_ok=True)
rng = np.random.default_rng(0)

SCEN = sorted(SCENARIO_KEY, key=lambda t: int(MODEL_LABELS[t][1:]))
SCEN_TAGS = [MODEL_LABELS[t] for t in SCEN]
CASES = [c[2] for c in SOURCE_SURVEY_COMBOS]
CCOLOR = {c[2]: c[3] for c in SOURCE_SURVEY_COMBOS}

# Prior bounds actually used (pseudo_runs/run_inf_pseudo.sh).  All priors are uniform.
PRIOR = {
    "A": {"xc": (4, 7), "lam_schechter": (6.5, 22),
          "alpha_1_MBHSpin": (0.5, 5), "alpha_2_MBHSpin": (1, 15), "beta_1_MBHSpin": (0.5, 15),
          "beta_2_MBHSpin": (0.5, 15), "weight_MBHSpin": (0.001, 0.999),
          "alpha_1_e_pl": (1.001, 5), "alpha_2_e_pl": (2.5, 7), "beta_1_e_pl": (6.5, 16),
          "beta_2_e_pl": (4.5, 9), "weight_e_pl": (0.001, 0.999), "lam_z_gal": (-2, -1.01)},
    "B": {"lambda_M": (-3, 3),
          "mu_1_MBHSpin": (0.05, 0.5), "sigma_1_MBHSpin": (0.001, 0.3), "mu_2_MBHSpin": (0.5, 0.99),
          "sigma_2_MBHSpin": (0.001, 0.3), "weight_MBHSpin": (0.001, 0.999),
          "mu_1_e_pl": (0.1, 0.25), "sigma_1_e_pl": (0.01, 0.09), "mu_2_e_pl": (0.2, 0.4),
          "sigma_2_e_pl": (0.02, 0.2), "weight_e_pl": (0.001, 0.999), "lam_z_gal": (-2, -1.01)},
}

# ── load everything once ─────────────────────────────────────────────────────
DATA = {"A": {}, "B": {}}     # DATA[pop][scen]["case"] -> samples dict
TRUTH = {"A": {}, "B": {}}
for pop in "AB":
    for t, tag in zip(SCEN, SCEN_TAGS):
        combos = load_available_combos(base_dir, pop, *t)
        DATA[pop][tag] = {c["label"]: c["samples"] for c in combos}
        TRUTH[pop][tag] = get_true_x_flat(pop, *t)


def save(fig, name):
    p = f"{outdir}/{name}"
    fig.savefig(p + ".png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("saved", p + ".png")


def stat(x):
    lo, md, hi = np.percentile(x, [5, 50, 95])
    s68 = 0.5 * (np.percentile(x, 84) - np.percentile(x, 16))
    return lo, md, hi, s68


# ════════════════════════════════════════════════════════════════════════════
# V1 / V2 : heatmaps of information gain and bias
# ════════════════════════════════════════════════════════════════════════════
def info_and_pull(pop):
    labels = get_latex_labels(pop)
    params = list(labels)
    info = np.full((len(CASES), len(params), len(SCEN_TAGS)), np.nan)
    pull = np.full_like(info, np.nan)
    for ci, case in enumerate(CASES):
        for pi, p in enumerate(params):
            lo_p, hi_p = PRIOR[pop][p]
            for si, tag in enumerate(SCEN_TAGS):
                s = DATA[pop][tag].get(case)
                if s is None or p not in s:
                    continue
                lo, md, hi, s68 = stat(s[p])
                info[ci, pi, si] = 1 - (hi - lo) / (0.9 * (hi_p - lo_p))   # 90% width vs. 90% prior width
                pull[ci, pi, si] = (md - TRUTH[pop][tag][p]) / max(s68, 1e-12)
    return params, labels, info, pull


def heatmap_grid(pop, which):
    params, labels, info, pull = info_and_pull(pop)
    fig, axes = plt.subplots(2, 4, figsize=(19, 0.33 * len(params) * 2 + 3.2), sharey=True)
    axes = axes.ravel()
    for ci, case in enumerate(CASES):
        ax = axes[ci]
        if which == "info":
            M = info[ci]
            im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
            txt = lambda v, i, j: f"{v:.2f}"
        else:
            M = np.where(info[ci] > 0.15, pull[ci], np.nan)       # hide prior-dominated cells
            im = ax.imshow(M, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -5, 5))
            txt = lambda v, i, j: f"{v:+.1f}"
        ax.set_facecolor("#dddddd")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    dark = (which == "info" and M[i, j] < 0.55) or (which == "pull" and abs(M[i, j]) < 3)
                    ax.text(j, i, txt(M[i, j], i, j), ha="center", va="center", fontsize=8,
                            color="white" if (which == "info" and dark) or (which == "pull" and not dark) else "black")
        ax.set_xticks(range(len(SCEN_TAGS)), SCEN_TAGS)
        ax.set_yticks(range(len(params)), [labels[p] for p in params], fontsize=11)
        ax.set_title(case, color=CCOLOR[case], fontweight="bold")
    for ax in axes[len(CASES):]:
        ax.axis("off")
    cax = fig.add_axes([0.80, 0.14, 0.15, 0.022])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    if which == "info":
        cb.set_label("information gain  $1-w_{90}/w_{90}^{\\rm prior}$   (0 = prior, 1 = delta)")
        fig.suptitle(f"V1 · Population {pop}: how much does each hyperparameter learn?", y=0.995, fontsize=15)
    else:
        cb.set_label("pull  (median − truth)/$\\sigma_{68}$   (grey = prior-dominated, info<0.15)")
        fig.suptitle(f"V2 · Population {pop}: is the truth recovered where the data constrain it?", y=0.995, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


for pop in "AB":
    save(heatmap_grid(pop, "info"), f"V1_information_gain_pop{pop}")
    save(heatmap_grid(pop, "pull"), f"V2_pull_heatmap_pop{pop}")


# ════════════════════════════════════════════════════════════════════════════
# V3 : every hyperparameter on one common axis (offset from truth / prior width)
# ════════════════════════════════════════════════════════════════════════════
def pull_plot(pop):
    labels = get_latex_labels(pop)
    params = list(labels)
    fig, axes = plt.subplots(1, len(SCEN_TAGS), figsize=(20, 0.55 * len(params) + 2), sharey=True, sharex=True)
    off = np.linspace(-0.32, 0.32, len(CASES))
    for si, (ax, tag) in enumerate(zip(axes, SCEN_TAGS)):
        for pi, p in enumerate(params):
            ax.axhspan(pi - 0.5, pi + 0.5, color="0.95" if pi % 2 == 0 else "white", zorder=0)
            lo_p, hi_p = PRIOR[pop][p]
            for ci, case in enumerate(CASES):
                s = DATA[pop][tag].get(case)
                if s is None or p not in s:
                    continue
                lo, md, hi, _ = stat(s[p])
                tr, w = TRUTH[pop][tag][p], hi_p - lo_p
                y = pi + off[ci]
                ax.plot([(lo - tr) / w, (hi - tr) / w], [y, y], color=CCOLOR[case], lw=2, solid_capstyle="butt")
                ax.plot((md - tr) / w, y, "o", color=CCOLOR[case], ms=3.5)
        ax.axvline(0, color="k", lw=1.2, ls="--")
        ax.axvspan(-0.5, 0.5, color="none")
        ax.set_xlim(-1, 1)
        ax.set_title(tag, fontsize=13)
        ax.set_xlabel("(θ − truth)/prior width", fontsize=9)
    axes[0].set_yticks(range(len(params)), [labels[p] for p in params], fontsize=12)
    axes[0].set_ylim(len(params) - 0.5, -0.5)
    handles = [plt.Line2D([], [], color=CCOLOR[c], lw=3, label=c) for c in CASES]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"V3 · Population {pop}: median and 90% interval relative to truth, in units of prior width", y=1.0, fontsize=14)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    return fig


for pop in "AB":
    save(pull_plot(pop), f"V3_pull_plot_pop{pop}")


# ════════════════════════════════════════════════════════════════════════════
# Observable-space pdfs from hyperparameter draws
# ════════════════════════════════════════════════════════════════════════════
GRID = {"mass": np.linspace(4, 7, 300), "spin": np.linspace(0.01, 0.99, 300), "ecc": np.linspace(0.01, 0.99, 300)}
COMP = {"spin": "MBHSpin", "ecc": "e_pl"}


def _norm(pdf, g):
    return pdf / np.trapz(pdf, g, axis=-1)[..., None]


def pdf_from(pop, obs, S):
    """N x G array of population pdfs, or None if S does not constrain this observable."""
    g = GRID[obs]
    if obs == "mass":
        if pop == "A":
            if "xc" not in S:
                return None
            xc, lam = np.atleast_1d(S["xc"])[:, None], np.atleast_1d(S["lam_schechter"])[:, None]
            r = lam * g[None] / xc
            lp = lam * np.log(r) - r
            return _norm(np.exp(lp - lp.max(1, keepdims=True)), g)
        if "lambda_M" not in S:
            return None
        lam = np.atleast_1d(S["lambda_M"])[:, None]
        lp = lam * np.log(g[None])
        return _norm(np.exp(lp - lp.max(1, keepdims=True)), g)
    suf = COMP[obs]
    g = GRID[obs]
    if pop == "A":
        keys = [f"{k}_{suf}" for k in ("alpha_1", "alpha_2", "beta_1", "beta_2", "weight")]
        if keys[0] not in S:
            return None
        a1, a2, b1, b2, w = (np.atleast_1d(S[k])[:, None] for k in keys)
        return w * stats.beta.pdf(g[None], a1, b1) + (1 - w) * stats.beta.pdf(g[None], a2, b2)
    keys = [f"{k}_{suf}" for k in ("mu_1", "sigma_1", "mu_2", "sigma_2", "weight")]
    if keys[0] not in S:
        return None
    m1, s1, m2, s2, w = (np.atleast_1d(S[k])[:, None] for k in keys)
    pdf = w * stats.norm.pdf(g[None], m1, s1) + (1 - w) * stats.norm.pdf(g[None], m2, s2)
    return _norm(pdf, g)       # truncated to [0.01, 0.99] and renormalised for display


def thin(S):
    n = len(next(iter(S.values())))
    idx = rng.choice(n, size=min(args.NTHIN, n), replace=False)
    return {k: np.asarray(v)[idx] for k, v in S.items()}


def truth_dict(pop, tag):
    return {k: np.array([v]) for k, v in TRUTH[pop][tag].items()}


# ── V4 : posterior-predictive bands in observable space, A | B ───────────────
OBS_LABEL = {"mass": r"$\lg M_\bullet$", "spin": r"MBH spin $a$", "ecc": r"eccentricity $e$"}
OBS_CASES = {"mass": ["EMRI", "TDE (ALL)", "EMRI+TDE (ALL)"], "spin": ["EMRI", "EMRI+TDE (ALL)"],
             "ecc": ["EMRI", "EMRI+TDE (ALL)"]}


def band_plot(obs):
    g = GRID[obs]
    fig, axes = plt.subplots(len(SCEN_TAGS), 2, figsize=(10, 1.75 * len(SCEN_TAGS)), sharex=True)
    for col, pop in enumerate("AB"):
        for row, tag in enumerate(SCEN_TAGS):
            ax = axes[row, col]
            ymax = 0
            for case in OBS_CASES[obs]:
                S = DATA[pop][tag].get(case)
                pdf = pdf_from(pop, obs, thin(S)) if S else None
                if pdf is None:
                    continue
                lo, md, hi = np.percentile(pdf, [5, 50, 95], axis=0)
                ax.fill_between(g, lo, hi, color=CCOLOR[case], alpha=0.30, lw=0)
                ax.plot(g, md, color=CCOLOR[case], lw=1.6, label=case)
                ymax = max(ymax, hi.max())
            tp = pdf_from(pop, obs, truth_dict(pop, tag))[0]
            ax.plot(g, tp, "k--", lw=1.6, label="truth")
            ax.set_ylim(0, max(ymax, tp.max()) * 1.08)
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(tag, rotation=0, labelpad=18, fontsize=13, va="center")
            if row == 0:
                ax.set_title(f"Population {pop}", fontsize=14)
        axes[-1, col].set_xlabel(OBS_LABEL[obs])
    h, l = axes[0, 0].get_legend_handles_labels()
    hb, lb = axes[0, 1].get_legend_handles_labels()
    seen = dict(zip(l + lb, h + hb))
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"V4 · Inferred population distribution of {OBS_LABEL[obs]} (median, 90% band) vs. truth", y=0.995, fontsize=14)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    return fig


for obs in ("mass", "spin", "ecc"):
    save(band_plot(obs), f"V4_observable_bands_{obs}")


# ════════════════════════════════════════════════════════════════════════════
# Derived summaries, directly comparable between A and B
# ════════════════════════════════════════════════════════════════════════════
SUMMARIES = [("mass", r"$\langle \lg M_\bullet\rangle$"), ("spin", r"$\langle a\rangle$"),
             ("ecc", r"$\langle e\rangle$"), ("z", r"$\lambda_z$")]


def summary_samples(pop, key, S):
    if key == "z":
        return np.asarray(S["lam_z_gal"]) if "lam_z_gal" in S else None
    S = thin(S)
    pdf = pdf_from(pop, key, S)
    return None if pdf is None else np.trapz(GRID[key] * pdf, GRID[key], axis=-1)


def summary_truth(pop, key, tag):
    if key == "z":
        return TRUTH[pop][tag]["lam_z_gal"]
    return summary_samples(pop, key, truth_dict(pop, tag))[0]


SUMM = {}   # SUMM[(pop, key, tag, case)] = samples ; SUMT[(pop,key,tag)] = truth
SUMT = {}
for pop in "AB":
    for key, _ in SUMMARIES:
        for tag in SCEN_TAGS:
            SUMT[(pop, key, tag)] = summary_truth(pop, key, tag)
            for case, S in DATA[pop][tag].items():
                x = summary_samples(pop, key, S)
                if x is not None:
                    SUMM[(pop, key, tag, case)] = x


# ── V5 : forest plot of summaries ────────────────────────────────────────────
def forest():
    fig, axes = plt.subplots(len(SUMMARIES), 2, figsize=(11, 2.9 * len(SUMMARIES)), sharey=True)
    off = np.linspace(-0.34, 0.34, len(CASES))
    for r, (key, lab) in enumerate(SUMMARIES):
        allv = [v for (p, k, t, c), v in SUMM.items() if k == key]
        allv += [v for (p, k, t), v in SUMT.items() if k == key]
        lo_all = np.percentile(np.concatenate([np.atleast_1d(v) for v in allv]), 0.5)
        hi_all = np.percentile(np.concatenate([np.atleast_1d(v) for v in allv]), 99.5)
        for c, pop in enumerate("AB"):
            ax = axes[r, c]
            for si, tag in enumerate(SCEN_TAGS):
                ax.axhspan(si - 0.5, si + 0.5, color="0.95" if si % 2 == 0 else "white", zorder=0)
                for ci, case in enumerate(CASES):
                    x = SUMM.get((pop, key, tag, case))
                    if x is None:
                        continue
                    lo, md, hi, _ = stat(x)
                    ax.plot([lo, hi], [si + off[ci]] * 2, color=CCOLOR[case], lw=2.2, solid_capstyle="butt")
                    ax.plot(md, si + off[ci], "o", color=CCOLOR[case], ms=3.5)
                t = SUMT[(pop, key, tag)]
                ax.plot([t, t], [si - 0.45, si + 0.45], color="k", lw=2, zorder=5)
            ax.set_xlim(lo_all, hi_all)
            ax.set_ylim(len(SCEN_TAGS) - 0.5, -0.5)
            ax.set_yticks(range(len(SCEN_TAGS)), SCEN_TAGS)
            ax.set_xlabel(lab, fontsize=13)
            if r == 0:
                ax.set_title(f"Population {pop}", fontsize=14)
    handles = [plt.Line2D([], [], color=CCOLOR[c], lw=3, label=c) for c in CASES]
    handles.append(plt.Line2D([], [], color="k", lw=2, label="truth"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("V5 · Population summaries recovered by A and B on a common axis (median, 90%)", y=0.998, fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.985))
    return fig


save(forest(), "V5_summary_forest")


# ── V6 : z-score scorecard of the summaries ──────────────────────────────────
def scorecard():
    fig, axes = plt.subplots(2, len(SUMMARIES), figsize=(4.2 * len(SUMMARIES), 7.6), sharey=True)
    for r, pop in enumerate("AB"):
        for c, (key, lab) in enumerate(SUMMARIES):
            ax = axes[r, c]
            M = np.full((len(CASES), len(SCEN_TAGS)), np.nan)
            for ci, case in enumerate(CASES):
                for si, tag in enumerate(SCEN_TAGS):
                    x = SUMM.get((pop, key, tag, case))
                    if x is not None:
                        _, md, _, s68 = stat(x)
                        M[ci, si] = (md - SUMT[(pop, key, tag)]) / max(s68, 1e-12)
            im = ax.imshow(M, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -4, 4))
            ax.set_facecolor("#dddddd")
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if np.isfinite(M[i, j]):
                        ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=8,
                                fontweight="bold" if abs(M[i, j]) > 2 else "normal")
            ax.set_xticks(range(len(SCEN_TAGS)), SCEN_TAGS, fontsize=9)
            ax.set_yticks(range(len(CASES)), CASES, fontsize=10)
            ax.set_title(f"{lab}   (pop {pop})", fontsize=12)
    fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.03, pad=0.07,
                 label="pull  (median − truth)/σ$_{68}$   ·  bold = |pull|>2   ·  grey = case does not constrain it")
    fig.suptitle("V6 · Scorecard: does each case recover each population summary in each universe?", y=0.99, fontsize=14)
    return fig


save(scorecard(), "V6_summary_scorecard")
print("done")
