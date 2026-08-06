# #!/usr/bin/env python
# import argparse
# import os

# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns

# from utils import *

# # --- Matplotlib setup ---
# matplotlib.rc('font', family='serif', serif=['Computer Modern'], size=22)
# matplotlib.rc('text', usetex=True)

# parser = argparse.ArgumentParser(
#         description="1D posterior comparison across source+survey combinations "
#                      "(EMRI / TDE-ZTF / TDE-LSST / TDE-ALL / EMRI+TDE-ZTF / "
#                      "EMRI+TDE-LSST / EMRI+TDE-ALL)"
#     )
# parser.add_argument('--POPULATION', choices=['A', 'B'], required=True,
#                     help="Population model whose posteriors to plot")
# parser.add_argument('--GALAXIES', required=True,
#                 help="Galaxies tag used in the base data path")

# args = parser.parse_args()


# def smart_format(q50, q_low, q_high, alpha=0.8):
#     err_minus = q50 - q_low
#     err_plus = q_high - q50
#     u = min(abs(err_minus), abs(err_plus))
#     d = 1 if u == 0 else max(int(round(-np.log10(u) + alpha)), 0)
#     if 1 <= u < 2:
#         d = max(d, 1)
#     q50_r, em_r, ep_r = round(q50, d), round(err_minus, d), round(err_plus, d)
#     fmt = f"{{:.{d}f}}"
#     return f"${fmt.format(q50_r)}_{{-{fmt.format(em_r)}}}^{{+{fmt.format(ep_r)}}}$"


# def xlim_for_key(key, combos, truth, pad_frac=0.05):
#     """0.5-99.5 percentile range across every source that has this
#     parameter, padded a bit, and widened if needed so the truth line
#     stays visible. This is what removes the large empty-whitespace
#     margins that KDE's automatic axis scaling leaves on long-tailed or
#     tightly-peaked posteriors."""
#     vals = [c["samples"][key] for c in combos if key in c["samples"]]
#     if not vals:
#         return None
#     all_vals = np.concatenate(vals)
#     lo, hi = np.percentile(all_vals, [0.5, 99.5])
#     if truth is not None:
#         lo = min(lo, truth)
#         hi = max(hi, truth)
#     pad = (hi - lo) * pad_frac if hi > lo else max(abs(lo), 1.0) * pad_frac
#     return lo - pad, hi + pad


# base_dir = f"./data/pseudo_inference_data/{args.GALAXIES}"
# outdir = f"{base_dir}/PLOTS/1D-posteriors"

# # Modular loading: any missing/broken combo (e.g. a condor job that hasn't
# # finished, or a source/survey pairing that was never run) is skipped with
# # a warning instead of crashing the whole script. To add/remove a
# # combination, edit SOURCE_SURVEY_COMBOS in source_combos.py -- nothing
# # here needs to change.
# combos = load_available_combos(base_dir, args.POPULATION)
# if not combos:
#     raise RuntimeError(f"No source/survey combinations loaded for pop {args.POPULATION} -- nothing to plot.")

# labels_dict = get_latex_labels(args.POPULATION)
# truths_dict = get_true_x_flat(args.POPULATION)

# # Canonical parameter order for this population, restricted to params that
# # at least one loaded combo actually has -- so we never render an empty panel.
# keys = [k for k in labels_dict if any(k in c["samples"] for c in combos)]

# for c in combos:
#     missing = [k for k in labels_dict if k not in c["samples"]]
#     if missing:
#         print(f"[info] {c['label']} pop {args.POPULATION}: not sampled -> {missing}")

# labels_post = [labels_dict[k] for k in keys]
# truths = [truths_dict[k] for k in keys]

# n_params = len(keys)
# ncols = 4
# nrows = int(np.ceil((n_params + 1) / ncols))  # +1 for the reserved legend slot
# fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows))
# axes = axes.flatten()

# # Reserve one slot (top-right of the first row) for the legend.
# label_index = ncols - 1
# axes[label_index].axis('off')
# plot_slots = [i for i in range(len(axes)) if i != label_index][:n_params]

# # Vertical stagger for the per-source median +/- error text annotations.
# # Built to fit however many combos actually loaded (up to 7 now, not a
# # hardcoded 3) rather than silently dropping annotations for extra sources.
# y_positions = [0.95 - 0.115 * i for i in range(len(combos))]

# for idx, ax_i in enumerate(plot_slots):
#     ax = axes[ax_i]
#     key = keys[idx]
#     label = labels_post[idx]
#     for c, y_pos in zip(combos, y_positions):
#         if key not in c["samples"]:
#             continue
#         samples = c["samples"][key]
#         sns.kdeplot(samples, fill=True, color=c["color"], linewidth=1.8, ax=ax)
#         q5, q50, q95 = np.percentile(samples, [5, 50, 95])
#         err = smart_format(q50, q5, q95)
#         ax.text(0.04, y_pos, err, transform=ax.transAxes,
#                 verticalalignment='top', horizontalalignment='left',
#                 color=c["color"], fontsize=13)
#     ax.axvline(truths[idx], color='black', linestyle='--', linewidth=1.5)
#     xlim = xlim_for_key(key, combos, truths[idx])
#     if xlim is not None:
#         ax.set_xlim(*xlim)
#     ax.set_yticks([])
#     ax.set_ylabel(f'$p$({label}$|\\textit{{\\textbf{{d}}}}$)')
#     ax.set_xlabel(label, fontsize=16)
#     ax.tick_params(axis='x', labelsize=16)
#     ax.minorticks_on()

# # Hide any leftover unused subplots.
# used = set(plot_slots) | {label_index}
# for j in range(len(axes)):
#     if j not in used:
#         axes[j].axis('off')

# handles = [plt.Line2D([], [], color=c["color"], lw=2, label=c["label"]) for c in combos]
# fig.legend(handles=handles, loc='upper right', fontsize=18, title='Sources',
#             title_fontsize=20, frameon=False, bbox_to_anchor=(0.85, 0.93), ncol=1)

# plt.subplots_adjust(left=0.08, right=0.88, top=0.93, bottom=0.08, wspace=0.35, hspace=0.50)

# os.makedirs(outdir, exist_ok=True)
# outpath = f"{outdir}/{args.GALAXIES}_pop_{args.POPULATION}_1D_posteriors.png"
# plt.savefig(outpath, dpi=300, bbox_inches="tight")
# print(f"Saved: {outpath}")



#!/usr/bin/env python
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import *

# --- Matplotlib setup ---
matplotlib.rc('font', family='serif', serif=['Computer Modern'], size=22)
matplotlib.rc('text', usetex=True)

parser = argparse.ArgumentParser(
        description="1D posterior comparison across source+survey combinations "
                     "(EMRI / TDE-ZTF / TDE-LSST / TDE-ALL / EMRI+TDE-ZTF / "
                     "EMRI+TDE-LSST / EMRI+TDE-ALL)"
    )
parser.add_argument('--POPULATION', choices=['A', 'B'], required=True,
                    help="Population model whose posteriors to plot")
parser.add_argument('--GALAXIES', required=True,
                help="Galaxies tag used in the base data path")
parser.add_argument('--mbh_mass_method', required=True, choices=['use_sigma', 'use_lgMgal'])
parser.add_argument('--nucleation_method', required=True, choices=['hannah', 'neumayer'])

args = parser.parse_args()


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


def xlim_for_key(key, combos, truth, pad_frac=0.05):
    """0.5-99.5 percentile range across every source that has this
    parameter, padded a bit, and widened if needed so the truth line
    stays visible. This is what removes the large empty-whitespace
    margins that KDE's automatic axis scaling leaves on long-tailed or
    tightly-peaked posteriors."""
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


base_dir = f"./data/pseudo_inference_data/{args.GALAXIES}"
scenario_tag = f"{args.mbh_mass_method}_{args.nucleation_method}"
outdir = f"{base_dir}/PLOTS/1D-posteriors/{scenario_tag}"

# Modular loading: any missing/broken combo (e.g. a condor job that hasn't
# finished, or a source/survey pairing that was never run) is skipped with
# a warning instead of crashing the whole script. To add/remove a
# combination, edit SOURCE_SURVEY_COMBOS in utils.py -- nothing here needs
# to change.
combos = load_available_combos(base_dir, args.POPULATION, args.mbh_mass_method, args.nucleation_method)
if not combos:
    raise RuntimeError(
        f"No source/survey combinations loaded for pop {args.POPULATION} "
        f"({scenario_tag}) -- nothing to plot."
    )

labels_dict = get_latex_labels(args.POPULATION)
truths_dict = get_true_x_flat(args.POPULATION, args.mbh_mass_method, args.nucleation_method)

# Canonical parameter order for this population, restricted to params that
# at least one loaded combo actually has -- so we never render an empty panel.
keys = [k for k in labels_dict if any(k in c["samples"] for c in combos)]

for c in combos:
    missing = [k for k in labels_dict if k not in c["samples"]]
    if missing:
        print(f"[info] {c['label']} pop {args.POPULATION} ({scenario_tag}): not sampled -> {missing}")

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

# Vertical stagger for the per-source median +/- error text annotations.
# Built to fit however many combos actually loaded (up to 7 now, not a
# hardcoded 3) rather than silently dropping annotations for extra sources.
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
    ax.axvline(truths[idx], color='black', linestyle='--', linewidth=1.5)
    xlim = xlim_for_key(key, combos, truths[idx])
    if xlim is not None:
        ax.set_xlim(*xlim)
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

handles = [plt.Line2D([], [], color=c["color"], lw=2, label=c["label"]) for c in combos]
fig.legend(handles=handles, loc='upper right', fontsize=18, title='Sources',
            title_fontsize=20, frameon=False, bbox_to_anchor=(0.85, 0.93), ncol=1)

plt.subplots_adjust(left=0.08, right=0.88, top=0.93, bottom=0.08, wspace=0.35, hspace=0.50)

os.makedirs(outdir, exist_ok=True)
outpath = f"{outdir}/{args.GALAXIES}_pop_{args.POPULATION}_{scenario_tag}_1D_posteriors.png"
plt.savefig(outpath, dpi=300, bbox_inches="tight")
print(f"Saved: {outpath}")