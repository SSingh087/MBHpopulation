"""Posterior predictive distributions of the inferred populations in observable space, with a disk cache of the histograms."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#!/usr/bin/env python
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from utils import *

matplotlib.rc('font', family='serif', serif=['Computer Modern'], size=20)
matplotlib.rc('text', usetex=True)

parser = argparse.ArgumentParser(description="Posterior Predictive Distributions (PPD)")
parser.add_argument('--POPULATION', choices=['A', 'B'], required=True)
parser.add_argument('--GALAXIES', required=True)
parser.add_argument("--mbh_mass_method", type=str, required=True, default="use_sigma", choices=["use_sigma", "use_lgMgal"])
parser.add_argument("--nucleation_method", type=str, required=True, default="hannah", choices=["hannah", "neumayer"])
parser.add_argument("--occupation_method", type=str, required=True, default="uniform", choices=["uniform", "varying"])
parser.add_argument('--NSAMPLES', type=int, default=500, help="posterior samples to draw")
parser.add_argument('--DRAW_SIZE', type=int, default=500, help="samples per catalogue draw")
parser.add_argument('--NO_CACHE', action='store_true',
                     help="force recompute even if a cached PPD histogram exists")

args = parser.parse_args()

base_dir = f"./data/pseudo_inference_data/{args.GALAXIES}"
scenario_tag = f"{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}"
outdir = f"{base_dir}/PLOTS/PPD/{scenario_tag}"
os.makedirs(outdir, exist_ok=True)

# Modular loading: any missing/broken combo is skipped with a warning
# instead of crashing the whole script. Edit SOURCE_SURVEY_COMBOS in
# utils.py to add/remove combinations -- nothing here changes.
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

TRUE_SIZE = 500
catalogue_true = popdist.draw_samples(x=true_x, size=TRUE_SIZE)

keys = list(catalogue_true.keys())

labels_dict = {
    "lgMBH_mass": r"\log_{10}(M_\mathrm{BH}/M_\odot)",
    "MBHspin": r"a",
    "e_pl": r"e_\mathrm{pl}",
    "z_gal": r"z",
}


def compute_hist(data, bins=50, log=False):
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


# get_nested_from_flat lives in utils.py (same explicit flat<->nested name
# table used by get_true_x_flat()) and returns (nested_dict,
# missing_components) -- missing_components lists which top-level
# population components (e.g. "MBHspin") this particular posterior row
# did NOT actually sample, so we skip drawing a PPD contribution for
# those rather than plotting truth-pinned noise.

n_params = len(keys)
ncols = 3
nrows = int(np.ceil((n_params + 1) / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
axes = axes.flatten()

# Reserve legend slot
legend_idx = ncols - 1
axes[legend_idx].axis('off')

plot_slots = [i for i in range(len(axes)) if i != legend_idx][:n_params]

# --- Main loop ---
for idx, ax_i in enumerate(plot_slots):
    ax = axes[ax_i]
    key = keys[idx]
    label = labels_dict.get(key, key)
    is_log = (key == "z_gal")

    xlim_lo, xlim_hi = None, None

    # --- Loop over source/survey combos ---
    for c in combos:
        sdict = c["samples"]
        if len(sdict) == 0:
            continue

        cache_path = ppd_cache_path(base_dir, args.POPULATION, args.mbh_mass_method,
                                     args.nucleation_method, args.occupation_method, c["label"], key)
        cached = None if args.NO_CACHE else load_ppd_cache(cache_path)

        if cached is not None:
            centers, hist_stack = cached
            print(f"[cache hit] {c['label']} / {key} ({scenario_tag}) <- {cache_path}")
        else:
            any_key = list(sdict.keys())[0]
            N_total = len(sdict[any_key])
            idxs = np.random.choice(N_total, min(args.NSAMPLES, N_total), replace=False)

            hist_stack_list = []
            centers = None
            skipped_as_unsampled = 0

            for i in idxs:
                x_sample = {k: sdict[k][i] for k in sdict}
                x_nested, missing_components = get_nested_from_flat(
                    x_sample, args.POPULATION, args.mbh_mass_method,
                    args.nucleation_method, args.occupation_method
                )

                if key in missing_components:
                    # This source/survey combo never sampled the hyperparameters
                    # feeding this component -- skip rather than plot
                    # truth-pinned catalogue noise.
                    skipped_as_unsampled += 1
                    continue

                cat = popdist.draw_samples(x=x_nested, size=args.DRAW_SIZE)
                if key not in cat:
                    continue

                cts, hist = compute_hist(cat[key], log=is_log)
                if cts is None:
                    continue

                centers = cts
                hist_stack_list.append(hist)

            if skipped_as_unsampled == len(idxs):
                print(f"[info] {c['label']}: does not sample hyperparameters for '{key}' -> skipping")
                continue
            if len(hist_stack_list) == 0:
                print(f"[info] {c['label']}: no support for {key}")
                continue

            hist_stack = np.array(hist_stack_list).T
            save_ppd_cache(cache_path, centers, hist_stack)
            print(f"[cache write] {c['label']} / {key} ({scenario_tag}) -> {cache_path}")

        mean = np.mean(hist_stack, axis=1)
        p16, p84 = np.percentile(hist_stack, [16, 84], axis=1)

        ax.plot(centers, mean, color=c["color"], lw=2, linestyle='--')
        ax.fill_between(centers, p16, p84, color=c["color"], alpha=0.3)

        # Track the actual plotted data range on this panel across all
        # combos, so the x-axis can be limited to it instead of whatever
        # empty margin matplotlib's default autoscale leaves.
        c_lo, c_hi = centers.min(), centers.max()
        xlim_lo = c_lo if xlim_lo is None else min(xlim_lo, c_lo)
        xlim_hi = c_hi if xlim_hi is None else max(xlim_hi, c_hi)

    if xlim_lo is not None and xlim_hi is not None and xlim_hi > xlim_lo:
        pad = (xlim_hi - xlim_lo) * 0.03
        ax.set_xlim(xlim_lo - pad, xlim_hi + pad)

    if key == "z_gal":
        # centers are log10(z); relabel axis instead of set_xscale
        ax.set_xlabel(r"$\log_{10} z$")
        ax.set_ylabel(rf"$dN/d(\log_{{10}}({label}))$")
    else:
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
    plt.Line2D([], [], color=c["color"], lw=2, linestyle='--', label=c["label"])
    for c in combos
]

fig.legend(handles=handles, loc='upper right',
           fontsize=16, title='Sources', frameon=False, ncol=1)

plt.tight_layout()

base_name = f"{outdir}/{args.GALAXIES}_pop_{args.POPULATION}_{scenario_tag}_PPD"
plt.savefig(f"{base_name}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{base_name}.pdf", bbox_inches="tight")
print(f"Saved -> {base_name}.png / .pdf")