#!/usr/bin/env python
import argparse
import os, sys

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import *

matplotlib.rc('font', family='serif', serif=['Computer Modern'], size=20)
matplotlib.rc('text', usetex=True)

pal = sns.color_palette("colorblind").as_hex()

SOURCE_ORDER = ['EMRI', 'TDE']#, 'EMRI_TDE']
SOURCE_LABELS = {'EMRI': 'EMRI', 'TDE': 'TDE', 'EMRI_TDE': 'EMRI+TDE'}
SOURCE_COLORS = [pal[4], pal[2], pal[0]]


parser = argparse.ArgumentParser(description="Posterior Predictive Distributions (PPD)")
parser.add_argument('--POPULATION', choices=['A', 'B'], required=True)
parser.add_argument('--GALAXIES', required=True)
parser.add_argument('--NSAMPLES', type=int, default=200, help="posterior samples to draw")
parser.add_argument('--DRAW_SIZE', type=int, default=100, help="samples per catalogue draw")

args = parser.parse_args()

base_dir = f"./data/pseudo_inference_data/{args.GALAXIES}"
outdir = f"{base_dir}/PLOTS/PPD"

os.makedirs(outdir, exist_ok=True)

def load_samples_dict(result_path):
    with h5py.File(result_path, "r") as hf:
        ds = hf["posterior_samples"]
        return {name: np.asarray(ds[name]) for name in ds.dtype.names}

samples_all = []
for source in SOURCE_ORDER:
    path = f"{base_dir}/{source}_pop_{args.POPULATION}/result.hdf5"
    samples_all.append(load_samples_dict(path))


popdist = make_dist(args.POPULATION)
true_x = get_true_x(args.POPULATION)
truth_flat = get_true_x_flat(args.POPULATION)

TRUE_SIZE = 100
catalogue_true = popdist.draw_samples(x=true_x, size=TRUE_SIZE)


keys = list(catalogue_true.keys())

labels_dict = {
    "lgMBH_mass": r"\log_{10}(M_\mathrm{BH}/M_\odot)",
    "MBHspin": r"a",
    "CO_mass": r"m_\mathrm{CO}\,[M_\odot]",
    "e_pl": r"e_\mathrm{pl}",
    "z_gal": r"z",
}

def compute_hist(data, bins=50):
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return None, None
    hist, edges = np.histogram(data, bins=bins, density=False)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return centers, hist

n_params = len(keys)
ncols = 3
nrows = int(np.ceil((n_params + 1) / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
axes = axes.flatten()

# Reserve legend slot
legend_idx = ncols - 1
axes[legend_idx].axis('off')

plot_slots = [i for i in range(len(axes)) if i != legend_idx][:n_params]

def get_nested_from_flat(flat_dict, population):
    true = get_true_x(population)
    out = {}

    for key in true:
        out[key] = {}
        for subkey in true[key]:
            flat_name = subkey
            if subkey in flat_dict:
                out[key][subkey] = flat_dict[subkey]
            else:
                out[key][subkey] = true[key][subkey]  # fallback

    return out

# --- Main loop ---
for idx, ax_i in enumerate(plot_slots):
    ax = axes[ax_i]
    key = keys[idx]
    label = labels_dict.get(key, key)

    # --- Truth ---
    centers_true, hist_true = compute_hist(catalogue_true[key])
    if centers_true is not None:
        ax.plot(centers_true, hist_true, color='black', lw=2, label='Truth')

    # --- Loop over sources ---
    for sdict, source, color in zip(samples_all, SOURCE_ORDER, SOURCE_COLORS):

        # Skip if hyperparams not available
        if len(sdict) == 0:
            continue

        # Identify number of posterior samples to use
        any_key = list(sdict.keys())[0]
        N_total = len(sdict[any_key])
        idxs = np.random.choice(N_total, min(args.NSAMPLES, N_total), replace=False)

        hist_stack = []

        for i in idxs:

            # Build hyperparameter dict
            x_sample = {k: sdict[k][i] for k in sdict}

            # Convert to nested format expected by popdist
            x_nested = get_nested_from_flat(x_sample, args.POPULATION)

            # Draw catalogue
            cat = popdist.draw_samples(x=x_nested, size=args.DRAW_SIZE)

            if key not in cat:
                continue

            centers, hist = compute_hist(cat[key])
            if centers is None:
                continue

            hist_stack.append(hist)

        if len(hist_stack) == 0:
            print(f"[info] {source}: no support for {key}")
            continue

        hist_stack = np.array(hist_stack).T

        mean = np.mean(hist_stack, axis=1)
        p16, p84 = np.percentile(hist_stack, [16, 84], axis=1)

        ax.plot(centers, mean, color=color, lw=2, linestyle='--')
        ax.fill_between(centers, p16, p84, color=color, alpha=0.3)

    ax.set_xlabel(rf"${label}$")
    ax.set_ylabel(rf"$dN/d({label})$")
    ax.minorticks_on()
    ax.set_yticks([])

# --- Cleanup unused axes ---
used = set(plot_slots) | {legend_idx}
for j in range(len(axes)):
    if j not in used:
        axes[j].axis('off')

# --- Legend ---
handles = [
    plt.Line2D([], [], color='black', lw=2, label='Truth')
] + [
    plt.Line2D([], [], color=c, lw=2, linestyle='--', label=SOURCE_LABELS[s])
    for c, s in zip(SOURCE_COLORS, SOURCE_ORDER)
]

fig.legend(handles=handles, loc='upper right',
           fontsize=18, title='Sources', frameon=False)

plt.tight_layout()

outpath = f"{outdir}/{args.GALAXIES}_pop_{args.POPULATION}_PPD.png"
plt.savefig(outpath, dpi=300)
print(f"Saved → {outpath}")