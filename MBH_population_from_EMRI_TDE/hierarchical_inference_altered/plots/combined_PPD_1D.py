#!/usr/bin/env python
"""
Merge PPD + 1D hyperposteriors for ONE population into a single figure,
paper-style layout:

    +------+-------------------------------+
    | PPD  |  1D hyperposteriors           |
    | col  |  3-column grid                |
    +------+-------------------------------+

Now supports all source/survey combinations (EMRI, TDE x {ZTF,LSST,ALL},
EMRI+TDE x {ZTF,LSST,ALL}) via the shared, modular combo list in
source_combos.py -- any combo whose result.hdf5 is missing/broken is
skipped with a warning rather than crashing the figure, and adding new
combinations only requires editing source_combos.py.

Usage:
    python plots/combined_PPD_1D.py --POPULATION A --GALAXIES 100000
    python plots/combined_PPD_1D.py --POPULATION B --GALAXIES 100000
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

from utils import *

matplotlib.rc('font', family='serif', serif=['Computer Modern'], size=22)
matplotlib.rc('text', usetex=True)

# Explicit size knobs
LABEL_FONTSIZE = 22
TICK_FONTSIZE = 18
ANNOT_FONTSIZE = 15   # the median +/- error text in each 1D posterior panel
LEGEND_FONTSIZE = 16

PPD_KEYS = ["lgMBH_mass", "MBHspin", "e_pl", "z_gal"]
PPD_LABELS = {
    "lgMBH_mass": r"\log_{10}(M_\mathrm{BH}/M_\odot)",
    "MBHspin": r"a",
    "e_pl": r"e_\mathrm{pl}",
    "z_gal": r"z",
}

parser = argparse.ArgumentParser(description="Merged PPD + 1D-posterior figure (paper-style layout)")
parser.add_argument('--POPULATION', choices=['A', 'B'], required=True)
parser.add_argument('--GALAXIES', required=True, help="Galaxies tag used in the base data path")
parser.add_argument("--mbh_mass_method", type=str, required=True, choices=["use_sigma", "use_lgMgal"])
parser.add_argument("--nucleation_method", type=str, required=True, choices=["hannah", "neumayer"])
parser.add_argument("--occupation_method", type=str, required=True, choices=["uniform", "varying"])
parser.add_argument('--NSAMPLES', type=int, default=500, help="posterior samples to draw for PPD")
parser.add_argument('--DRAW_SIZE', type=int, default=500, help="samples per catalogue draw for PPD")
parser.add_argument('--POST_NCOLS', type=int, default=3, help="columns in the 1D-posterior grid")
args = parser.parse_args()

base_dir = f"./data/altered_inference_data/pseudo_inference_data/{args.GALAXIES}"
scenario_tag = f"{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}"
outdir = f"{base_dir}/PLOTS/combined/{scenario_tag}"
os.makedirs(outdir, exist_ok=True)


# ---------------------------------------------------------------- helpers
def smart_format(q50, q_low, q_high, alpha=0.8):
    """Format a median with asymmetric errors using a sensible number of digits."""
    err_minus = q50 - q_low
    err_plus = q_high - q50
    u = min(abs(err_minus), abs(err_plus))
    d = 1 if u == 0 else max(int(round(-np.log10(u) + alpha)), 0)
    if 1 <= u < 2:
        d = max(d, 1)
    q50_r, em_r, ep_r = round(q50, d), round(err_minus, d), round(err_plus, d)
    fmt = f"{{:.{d}f}}"
    return f"${fmt.format(q50_r)}_{{-{fmt.format(em_r)}}}^{{+{fmt.format(ep_r)}}}$"


def compute_hist(data, bins=50, log=False):
    """Histogram of the draws on the given bins."""
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    data = data[data > 0] if log else data
    if len(data) == 0:
        return None, None
    if log:
        data = np.log10(data)
    hist, edges = np.histogram(data, bins=bins, density=False)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return centers, hist


def xlim_for_key(key, combos, truth, pad_frac=0.05):
    """0.5-99.5 percentile range across every source that has this
    parameter, padded a bit, widened to include truth if needed. Used for
    the 1D-posterior column."""
    vals = [c["samples"][key] for c in combos if key in c["samples"]]
    if not vals:
        return None
    all_vals = np.concatenate(vals)
    lo, hi = np.percentile(all_vals, [0.5, 99.5])
    if truth is not None:
        lo = min(lo, truth)
        hi = max(hi, truth)
    pad = (hi - lo) * pad_frac if hi > lo else max(abs(lo), 1.0) * pad_frac
    return lo - pad, hi + pad


# get_nested_from_flat is imported from utils (see utils_patch.py) -- it
# uses the SAME explicit flat<->nested name table as get_true_x_flat(), so
# it can't silently mismatch names. It also returns which components a
# given source's row did NOT actually sample, so we know to skip those
# rather than plot truth-pinned noise.


# ---------------------------------------------------------------- load data (once)
# Modular loading: any missing/broken combo is skipped with a warning
# instead of crashing the figure. Edit SOURCE_SURVEY_COMBOS in
# source_combos.py to add/remove source/survey combinations.
combos = load_available_combos(
    base_dir, args.POPULATION, args.mbh_mass_method, args.nucleation_method, args.occupation_method
)
if not combos:
    raise RuntimeError(
        f"No source/survey combinations loaded for pop {args.POPULATION} "
        f"({scenario_tag}) -- nothing to plot."
    )

popdist = make_dist(args.POPULATION)
true_x = get_true_x(
    args.POPULATION, args.mbh_mass_method, args.nucleation_method, args.occupation_method
)

labels_dict = get_latex_labels(args.POPULATION)
truths_dict = get_true_x_flat(
    args.POPULATION, args.mbh_mass_method, args.nucleation_method, args.occupation_method
)
post_keys = [k for k in labels_dict if any(k in c["samples"] for c in combos)]
for c in combos:
    missing = [k for k in labels_dict if k not in c["samples"]]
    if missing:
        print(f"[info] {c['label']} pop {args.POPULATION}: not sampled -> {missing}")

n_ppd = len(PPD_KEYS)                       # 5
ncols_post = args.POST_NCOLS                # 3
nrows_post = int(np.ceil(len(post_keys) / ncols_post))
nrows_grid = nrows_post                     # PPD column spans this full height too (see below)

# ---------------------------------------------------------------- figure / gridspec
ncols_total = 1 + ncols_post
fig = plt.figure(figsize=(4.6 * ncols_total, 2.5 * nrows_grid))
gs = gridspec.GridSpec(
    nrows_grid, ncols_total, figure=fig,
    width_ratios=[1.15] + [1.0] * ncols_post,
    wspace=0.40, hspace=0.55,
    left=0.06, right=0.98, top=0.98, bottom=0.05,
)

gs_ppd = gridspec.GridSpecFromSubplotSpec(n_ppd, 1, subplot_spec=gs[:, 0], hspace=0.55)
ppd_axes = [fig.add_subplot(gs_ppd[i, 0]) for i in range(n_ppd)]

post_axes = [fig.add_subplot(gs[r, c]) for c in range(1, ncols_total) for r in range(nrows_post)]
# ^ column-major fill: top-to-bottom within a column, then move to the next column

# ---------------------------------------------------------------- PPD column (left)
for key, ax in zip(PPD_KEYS, ppd_axes):
    label = PPD_LABELS[key]
    is_log = (key == "z_gal")

    xlim_lo, xlim_hi = None, None

    for c in combos:
        sdict = c["samples"]
        if len(sdict) == 0:
            continue
        any_key = list(sdict.keys())[0]
        N_total = len(sdict[any_key])
        idxs = np.random.choice(N_total, min(args.NSAMPLES, N_total), replace=False)

        hist_stack, centers = [], None
        skipped_as_unsampled = 0
        for i in idxs:
            x_sample = {k: sdict[k][i] for k in sdict}
            x_nested, missing_components = get_nested_from_flat(
                x_sample, args.POPULATION, args.mbh_mass_method,
                args.nucleation_method, args.occupation_method
            )
            if key in missing_components:
                skipped_as_unsampled += 1
                continue
            cat = popdist.draw_samples(x=x_nested, size=args.DRAW_SIZE)
            if key not in cat:
                continue
            cts, hist = compute_hist(cat[key], log=is_log)
            if cts is None:
                continue
            centers = cts
            hist_stack.append(hist)

        if skipped_as_unsampled == len(idxs):
            print(f"[info] {c['label']}: does not sample hyperparameters for '{key}' -> skipping PPD panel")
            continue
        if len(hist_stack) == 0:
            print(f"[info] {c['label']}: no support for {key}")
            continue

        hist_stack = np.array(hist_stack).T
        mean = np.mean(hist_stack, axis=1)
        p16, p84 = np.percentile(hist_stack, [16, 84], axis=1)
        ax.plot(centers, mean, color=c["color"], lw=2, linestyle='--')
        ax.fill_between(centers, p16, p84, color=c["color"], alpha=0.3)

        c_lo, c_hi = centers.min(), centers.max()
        xlim_lo = c_lo if xlim_lo is None else min(xlim_lo, c_lo)
        xlim_hi = c_hi if xlim_hi is None else max(xlim_hi, c_hi)

    if xlim_lo is not None and xlim_hi is not None and xlim_hi > xlim_lo:
        pad = (xlim_hi - xlim_lo) * 0.03
        ax.set_xlim(xlim_lo - pad, xlim_hi + pad)

    if key == "z_gal":
        ax.set_xlabel(r"$\log_{10} z$", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(rf"$dN/d(\log_{{10}}({label}))$", fontsize=LABEL_FONTSIZE)
    else:
        ax.set_xlabel(rf"${label}$", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(rf"$dN/d({label})$", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    ax.minorticks_on()
    ax.set_yticks([])

# Legend placement: use a spare 1D-posterior slot if one exists (grid may
# not divide evenly by ncols_post). With up to 7 combos now instead of 3,
# a 2-column legend fits more comfortably than 1.
handles = [plt.Line2D([], [], color=c["color"], lw=2, linestyle='--', label=c["label"])
           for c in combos]

n_post_slots = nrows_post * ncols_post
spare_post_axes = post_axes[len(post_keys):]

if len(spare_post_axes) > 0:
    legend_ax = spare_post_axes[0]
    legend_ax.legend(handles=handles, loc='center', frameon=False,
                      fontsize=LEGEND_FONTSIZE, ncol=2 if len(handles) > 4 else 1)
else:
    # No spare 1D-posterior slot available (grid exactly full) -- fall
    # back to a figure-level legend instead of silently dropping it.
    fig.legend(handles=handles, loc='upper right', frameon=False,
               fontsize=LEGEND_FONTSIZE, ncol=2 if len(handles) > 4 else 1)

# ---------------------------------------------------------------- 1D posteriors (right)
y_positions = [0.95 - 0.115 * i for i in range(len(combos))]
for idx, ax in enumerate(post_axes[:len(post_keys)]):
    key = post_keys[idx]
    label = labels_dict[key]
    for c, y_pos in zip(combos, y_positions):
        if key not in c["samples"]:
            continue
        samples = c["samples"][key]
        sns.kdeplot(samples, fill=True, color=c["color"], linewidth=1.6, ax=ax)
        q5, q50, q95 = np.percentile(samples, [5, 50, 95])
        err = smart_format(q50, q5, q95)
        ax.text(0.04, y_pos, err, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left',
                color=c["color"], fontsize=ANNOT_FONTSIZE)
    if key in truths_dict:
        ax.axvline(truths_dict[key], color='black', linestyle='--', linewidth=1.3)
    xlim = xlim_for_key(key, combos, truths_dict.get(key))
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_yticks([])
    ax.set_ylabel(f'$p$({label}$|\\textit{{\\textbf{{d}}}}$)', fontsize=LABEL_FONTSIZE)
    ax.set_xlabel(label, fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='x', labelsize=TICK_FONTSIZE)
    ax.minorticks_on()

for ax in post_axes[len(post_keys):]:
    ax.axis('off')

base_name = f"{outdir}/{args.GALAXIES}_pop_{args.POPULATION}_{scenario_tag}_PPD_plus_1D"
fig.savefig(f"{base_name}.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{base_name}.pdf", bbox_inches="tight")
print(f"Saved -> {base_name}.png")
print(f"Saved -> {base_name}.pdf")