#!/usr/bin/env python
"""
1D and 2D (corner) hyperposterior comparison for ONE population x scenario,
across every source/survey combination (EMRI / TDE-ZTF / TDE-LSST / TDE-ALL /
EMRI+TDE-ZTF / EMRI+TDE-LSST / EMRI+TDE-ALL).

  - 1D grid : one panel per hyperparameter, KDE per combo, median +/- 90%
              annotation.
  - 2D corner: overlaid corner.corner() per combo, restricted to the
              hyperparameters common to every loaded combo (required so the
              overlays share the same axes layout).

Both outputs are saved as PDF and PNG.

Usage:
    python plots/posterior_plots.py --GALAXIES 100000 --POPULATION A \\
        --mbh_mass_method use_sigma --nucleation_method hannah --occupation_method uniform
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import os

import corner
import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import *

# --- Matplotlib setup ---
matplotlib.rc('font', family='serif', serif=['Computer Modern'], size=22)
matplotlib.rc('text', usetex=True)

parser = argparse.ArgumentParser(
    description="1D + 2D posterior comparison across source/survey combinations"
)
parser.add_argument('--POPULATION', choices=['A', 'B'], required=True,
                     help="Population model whose posteriors to plot")
parser.add_argument('--GALAXIES', required=True,
                     help="Galaxies tag used in the base data path")
parser.add_argument("--mbh_mass_method", type=str, required=True, choices=["use_sigma", "use_lgMgal"])
parser.add_argument("--nucleation_method", type=str, required=True, choices=["hannah", "neumayer"])
parser.add_argument("--occupation_method", type=str, required=True, choices=["uniform", "varying"])
args = parser.parse_args()


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


def xlim_for_key(key, combos, pad_frac=0.05):
    """0.5-99.5 percentile range across every source that has this
    parameter, padded a bit."""
    vals = [c["samples"][key] for c in combos if key in c["samples"]]
    if not vals:
        return None
    all_vals = np.concatenate(vals)
    lo, hi = np.percentile(all_vals, [0.5, 99.5])
    pad = (hi - lo) * pad_frac if hi > lo else max(abs(lo), 1.0) * pad_frac
    return lo - pad, hi + pad


def save_fig(fig, base_path):
    """Save the figure as PNG and PDF."""
    fig.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base_path}.pdf", bbox_inches="tight")
    print(f"Saved: {base_path}.png / .pdf")


# ── Shared data loading ─────────────────────────────────────────────────────
base_dir = f"./data/inference_data/{args.GALAXIES}"
scenario_tag = f"{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}"

combos = load_available_combos(
    base_dir, args.POPULATION, args.mbh_mass_method, args.nucleation_method, args.occupation_method
)
if not combos:
    raise RuntimeError(
        f"No source/survey combinations loaded for pop {args.POPULATION} "
        f"({scenario_tag}) -- nothing to plot."
    )

labels_dict = get_latex_labels(args.POPULATION)

for c in combos:
    missing = [k for k in labels_dict if k not in c["samples"]]
    if missing:
        print(f"[info] {c['label']} pop {args.POPULATION} ({scenario_tag}): not sampled -> {missing}")


# ════════════════════════════════════════════════════════════════════════════
# 1D grid: any key at least one combo sampled -- per-panel skips are fine.
# ════════════════════════════════════════════════════════════════════════════
def make_1d_grid():
    """Grid of 1D marginal posteriors."""
    keys = [k for k in labels_dict if any(k in c["samples"] for c in combos)]
    labels_post = [labels_dict[k] for k in keys]

    n_params = len(keys)
    ncols = 4
    nrows = int(np.ceil((n_params + 1) / ncols))  # +1 for the reserved legend slot
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows))
    axes = axes.flatten()

    label_index = ncols - 1
    axes[label_index].axis('off')
    plot_slots = [i for i in range(len(axes)) if i != label_index][:n_params]

    y_positions = [0.95 - 0.115 * i for i in range(len(combos))]

    for idx, ax_i in enumerate(plot_slots):
        ax = axes[ax_i]
        key = keys[idx]
        label = labels_post[idx]
        for c, y_pos in zip(combos, y_positions):
            if key not in c["samples"]:
                continue
            samples = c["samples"][key]
            sns.kdeplot(samples, fill=True, color=c["color"], linewidth=1.8, ax=ax)
            q5, q50, q95 = np.percentile(samples, [5, 50, 95])
            err = smart_format(q50, q5, q95)
            ax.text(0.04, y_pos, err, transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    color=c["color"], fontsize=13)
        xlim = xlim_for_key(key, combos)
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.set_yticks([])
        ax.set_ylabel(f'$p$({label}$|\\textit{{\\textbf{{d}}}}$)')
        ax.set_xlabel(label, fontsize=16)
        ax.tick_params(axis='x', labelsize=16)
        ax.minorticks_on()

    used = set(plot_slots) | {label_index}
    for j in range(len(axes)):
        if j not in used:
            axes[j].axis('off')

    handles = [plt.Line2D([], [], color=c["color"], lw=2, label=c["label"]) for c in combos]
    fig.legend(handles=handles, loc='upper right', fontsize=18, title='Sources',
               title_fontsize=20, frameon=False, bbox_to_anchor=(0.85, 0.93), ncol=1)

    plt.subplots_adjust(left=0.08, right=0.88, top=0.93, bottom=0.08, wspace=0.35, hspace=0.50)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# 2D corner: overlays require identical dimensionality -- restrict to keys
# common to EVERY loaded combo.
# ════════════════════════════════════════════════════════════════════════════
def make_2d_corner():
    """2D corner plot of the posteriors."""
    keys = [k for k in labels_dict if all(k in c["samples"] for c in combos)]
    dropped = [k for k in labels_dict if k not in keys]
    if dropped:
        print(f"[info] excluded from corner plot (not common to all combos): {dropped}")
    if len(keys) < 2:
        print(f"[warn] fewer than 2 shared parameters ({keys}) -- skipping corner plot")
        return None

    labels = [labels_dict[k] for k in keys]

    default_kwargs = dict(
        bins=32,
        smooth=0.9,
        plot_density=False,
        plot_datapoints=False,
        fill_contours=False,
        max_n_ticks=4,
        show_titles=False,
        label_kwargs=dict(fontsize=26),
        labelpad=0.2,
        title_kwargs={"fontsize": 18},
    )

    fig_corner = None
    for c in combos:
        samp = np.vstack([c["samples"][k] for k in keys]).T
        fig_corner = corner.corner(
            samp,
            fig=fig_corner,
            color=c["color"],
            labels=labels,
            hist_kwargs=dict(density=True, color=c["color"], linewidth=2),
            contour_kwargs=dict(linewidths=2),
            **default_kwargs,
        )

    handles = [mlines.Line2D([], [], color=c["color"], linewidth=2, label=c["label"]) for c in combos]
    fig_corner.legend(handles=handles, loc='upper right', fontsize=40,
                       title_fontsize=50, frameon=False, title='Sources',
                       bbox_to_anchor=(0.85, 0.93))
    return fig_corner


# ── Run ──────────────────────────────────────────────────────────────────
outdir_1d = f"{base_dir}/PLOTS/1D-posteriors/{scenario_tag}"
outdir_2d = f"{base_dir}/PLOTS/2D-posteriors/{scenario_tag}"
os.makedirs(outdir_1d, exist_ok=True)
os.makedirs(outdir_2d, exist_ok=True)

fig_1d = make_1d_grid()
save_fig(fig_1d, f"{outdir_1d}/{args.GALAXIES}_pop_{args.POPULATION}_{scenario_tag}_1D_posteriors")

fig_2d = make_2d_corner()
if fig_2d is not None:
    save_fig(fig_2d, f"{outdir_2d}/{args.GALAXIES}_pop_{args.POPULATION}_{scenario_tag}_corner")
