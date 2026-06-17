#!/usr/bin/env python
import argparse
import os, sys

import corner
import h5py
import matplotlib
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns

from utils import *

# --- Matplotlib setup ---
matplotlib.rc('font', family='serif', serif=['Computer Modern'], size=22)
matplotlib.rc('text', usetex=True)

pal = sns.color_palette(palette="colorblind").as_hex()

SOURCE_ORDER = ['EMRI', 'TDE']  # add 'EMRI_TDE' back in once that run exists
SOURCE_LABELS = {'EMRI': 'EMRI', 'TDE': 'TDE', 'EMRI_TDE': 'EMRI+TDE'}
SOURCE_COLORS = [pal[4], pal[2], pal[0]]


parser = argparse.ArgumentParser(
        description="2D corner posterior comparison across source combinations (EMRI/TDE/EMRI+TDE)"
    )
parser.add_argument('--POPULATION', choices=['A', 'B'], required=True,
                    help="Population model whose posteriors to plot")
parser.add_argument('--GALAXIES', required=True,
                help="Galaxies tag used in the base data path")

args = parser.parse_args()


def load_samples_dict(result_path):
    """Read every field actually present in this run's posterior_samples
    table -- see posterior_1D.py for why this doesn't go through
    get_posterior_samples()."""
    with h5py.File(result_path, "r") as hf:
        ds = hf["posterior_samples"]
        return {name: np.asarray(ds[name]) for name in ds.dtype.names}


base_dir = f"./data/pseudo_inference_data/{args.GALAXIES}"
outdir = f"{base_dir}/PLOTS/2D-posteriors"

samples_all = []
for source in SOURCE_ORDER:
    result_path = f"{base_dir}/{source}_pop_{args.POPULATION}/result.hdf5"
    samples_all.append(load_samples_dict(result_path))

labels_dict = get_latex_labels(args.POPULATION)
truths_dict = get_true_x_flat(args.POPULATION)

# Corner overlays need the SAME dimensionality for every color, so unlike
# the 1D script (which can skip a source per-panel), here we can only plot
# parameters that EVERY source in SOURCE_ORDER actually sampled.
keys = [k for k in labels_dict if all(k in s for s in samples_all)]
dropped = [k for k in labels_dict if k not in keys]
if dropped:
    print(f"[info] excluded from corner plot (not common to all sources): {dropped}")

labels = [labels_dict[k] for k in keys]
truths = [truths_dict[k] for k in keys]

default_kwargs = dict(
    bins=32,
    smooth=0.9,
    truth_color='black',
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
for sdict, color in zip(samples_all, SOURCE_COLORS):
    samp = np.vstack([sdict[k] for k in keys]).T
    fig_corner = corner.corner(
        samp,
        fig=fig_corner,
        color=color,
        truths=truths,
        labels=labels,
        hist_kwargs=dict(density=True, color=color, linewidth=2),
        contour_kwargs=dict(linewidths=2),
        **default_kwargs,
    )

handles = [mlines.Line2D([], [], color=c, linewidth=2, label=SOURCE_LABELS[s])
           for c, s in zip(SOURCE_COLORS, SOURCE_ORDER)]
fig_corner.legend(handles=handles, loc='upper right', fontsize=40,
                   title_fontsize=50, frameon=False, title='Sources',
                   bbox_to_anchor=(0.85, 0.93))

os.makedirs(outdir, exist_ok=True)
base_name = f"{outdir}/{args.GALAXIES}_pop_{args.POPULATION}_corner"
fig_corner.savefig(f"{base_name}.pdf", dpi=500, bbox_inches='tight')
fig_corner.savefig(f"{base_name}.png", dpi=500, bbox_inches='tight')
print(f"Saved: {base_name}.pdf / .png")