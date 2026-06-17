#!/usr/bin/env python
import argparse
import os, sys

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import *

# --- Matplotlib setup ---
matplotlib.rc('font', family='serif', serif=['Computer Modern'], size=22)
matplotlib.rc('text', usetex=True)

pal = sns.color_palette(palette="colorblind").as_hex()

# --- Source combinations to overlay (this replaces the old event-count axis) ---
SOURCE_ORDER = ['EMRI', 'TDE']  # add 'EMRI_TDE' back in once that run exists
SOURCE_LABELS = {'EMRI': 'EMRI', 'TDE': 'TDE', 'EMRI_TDE': 'EMRI+TDE'}
SOURCE_COLORS = [pal[4], pal[2], pal[0]]


parser = argparse.ArgumentParser(
        description="1D posterior comparison across source combinations (EMRI/TDE/EMRI+TDE)"
    )
parser.add_argument('--POPULATION', choices=['A', 'B'], required=True,
                    help="Population model whose posteriors to plot")
parser.add_argument('--GALAXIES', required=True,
                help="Galaxies tag used in the base data path")

args = parser.parse_args()


def load_samples_dict(result_path):
    """Read every field actually present in this run's posterior_samples
    table, instead of going through get_posterior_samples()'s hardcoded key
    list. Different sources can legitimately sample different subsets of
    hyperparameters (e.g. a TDE-only run may not constrain MBH spin at all),
    so this just reports whatever columns exist rather than assuming a
    fixed schema.
    """
    with h5py.File(result_path, "r") as hf:
        ds = hf["posterior_samples"]
        return {name: np.asarray(ds[name]) for name in ds.dtype.names}


def smart_format(q50, q_low, q_high, alpha=0.8):
    err_minus = q50 - q_low
    err_plus = q_high - q50
    u = min(abs(err_minus), abs(err_plus))
    d = 1 if u == 0 else max(int(round(-np.log10(u) + alpha)), 0)
    if 1 <= u < 2:
        d = max(d, 1)
    q50_r, em_r, ep_r = round(q50, d), round(err_minus, d), round(err_plus, d)
    fmt = f"{{:.{d}f}}"
    return f"${fmt.format(q50_r)}_{{-{fmt.format(em_r)}}}^{{+{fmt.format(ep_r)}}}$"


base_dir = f"./data/pseudo_inference_data/{args.GALAXIES}"
outdir = f"{base_dir}/PLOTS/1D-posteriors"

samples_all = []  # list of dicts, one per source -- not a fixed-width array
for source in SOURCE_ORDER:
    result_path = f"{base_dir}/{source}_pop_{args.POPULATION}/result.hdf5"
    samples_all.append(load_samples_dict(result_path))

labels_dict = get_latex_labels(args.POPULATION)
truths_dict = get_true_x_flat(args.POPULATION)

# Canonical parameter order for this population, restricted to params that
# at least one source actually has -- so we never render an empty panel.
keys = [k for k in labels_dict if any(k in s for s in samples_all)]

# Surface exactly the situation that used to crash, as information instead
# of a traceback.
for source, sdict in zip(SOURCE_ORDER, samples_all):
    missing = [k for k in labels_dict if k not in sdict]
    if missing:
        print(f"[info] {source}_pop_{args.POPULATION}: not sampled -> {missing}")

labels_post = [labels_dict[k] for k in keys]
truths = [truths_dict[k] for k in keys]

n_params = len(keys)
ncols = 4
nrows = int(np.ceil((n_params + 1) / ncols))  # +1 for the reserved legend slot
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows))
axes = axes.flatten()

# Reserve one slot (top-right of the first row) for the legend.
label_index = ncols - 1
axes[label_index].axis('off')
plot_slots = [i for i in range(len(axes)) if i != label_index][:n_params]

y_positions = [0.95, 0.75, 0.55]
for idx, ax_i in enumerate(plot_slots):
    ax = axes[ax_i]
    key = keys[idx]
    label = labels_post[idx]
    for sdict, color, y_pos in zip(samples_all, SOURCE_COLORS, y_positions):
        if key not in sdict:
            continue  # this source doesn't constrain this parameter -- skip
        samples = sdict[key]
        sns.kdeplot(samples, fill=True, color=color, linewidth=1.8, ax=ax)
        q5, q50, q95 = np.percentile(samples, [5, 50, 95])
        err = smart_format(q50, q5, q95)
        ax.text(0.04, y_pos, err, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left',
                color=color, fontsize=14)
    ax.axvline(truths[idx], color='black', linestyle='--', linewidth=1.5)
    ax.set_yticks([])
    ax.set_ylabel(f'$p$({label}$|\\textit{{\\textbf{{d}}}}$)')
    ax.set_xlabel(label, fontsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.minorticks_on()

# Hide any leftover unused subplots.
used = set(plot_slots) | {label_index}
for j in range(len(axes)):
    if j not in used:
        axes[j].axis('off')

handles = [plt.Line2D([], [], color=c, lw=2, label=SOURCE_LABELS[s])
            for c, s in zip(SOURCE_COLORS, SOURCE_ORDER)]
fig.legend(handles=handles, loc='upper right', fontsize=20, title='Sources',
            title_fontsize=22, frameon=False, bbox_to_anchor=(0.85, 0.93))

plt.subplots_adjust(left=0.08, right=0.88, top=0.93, bottom=0.08, wspace=0.35, hspace=0.50)

os.makedirs(outdir, exist_ok=True)
outpath = f"{outdir}/{args.GALAXIES}_pop_{args.POPULATION}_1D_posteriors.png"
plt.savefig(outpath, dpi=300, bbox_inches="tight")
print(f"Saved: {outpath}")