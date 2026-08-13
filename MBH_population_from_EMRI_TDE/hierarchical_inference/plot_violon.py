#!/usr/bin/env python
"""
Violin-grid comparison across source/survey combos AND both population models.
Rows = shared parameters, columns = source/survey combo, split violin = pop A/B.
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    load_available_combos, get_latex_labels, get_true_x_flat,
    SOURCE_SURVEY_COMBOS,  # assumed ordered list of combo tags, e.g.
                           # ["EMRI","TDE-ZTF","TDE-LSST","TDE-ALL",
                           #  "EMRI+TDE-ZTF","EMRI+TDE-LSST","EMRI+TDE-ALL"]
)

matplotlib.rc('font', family='serif', serif=['Computer Modern'], size=18)
matplotlib.rc('text', usetex=True)

parser = argparse.ArgumentParser()
parser.add_argument('--GALAXIES', required=True)
parser.add_argument("--mbh_mass_method", required=True, choices=["use_sigma", "use_lgMgal"])
parser.add_argument("--nucleation_method", required=True, choices=["hannah", "neumayer"])
parser.add_argument("--occupation_method", required=True, choices=["uniform", "varying"])
args = parser.parse_args()

base_dir = f"./data/pseudo_inference_data/{args.GALAXIES}"
scenario_tag = f"{args.mbh_mass_method}_{args.nucleation_method}"
outdir = f"{base_dir}/PLOTS/violin-posteriors/{scenario_tag}"
os.makedirs(outdir, exist_ok=True)

POPS = ["A", "B"]
POP_COLORS = {"A": "#3b6fa0", "B": "#c0603d"}

# --- Load everything into a tidy long-form frame -----------------------
records = []
truths_by_pop = {}       # pop -> {param: truth}
labels_by_pop = {}       # pop -> {param: latex label}
combos_present = set()

for pop in POPS:
    combos = load_available_combos(base_dir, pop, args.mbh_mass_method, args.nucleation_method)
    if not combos:
        print(f"[warn] no combos loaded for population {pop} -- skipping")
        continue
    labels_by_pop[pop] = get_latex_labels(pop)
    truths_by_pop[pop] = get_true_x_flat(pop, args.mbh_mass_method, args.nucleation_method)

    for c in combos:
        combos_present.add(c["label"])
        for key, samples in c["samples"].items():
            records.append(pd.DataFrame({
                "value": samples,
                "param": key,
                "combo": c["label"],
                "population": pop,
            }))

if not records:
    raise RuntimeError("No data loaded for either population -- nothing to plot.")

df = pd.concat(records, ignore_index=True)

# --- Shared parameters: present for BOTH populations across ALL combos --
# (use intersection so every violin panel is fully populated; anything
# partially missing gets reported, not silently dropped into a half-empty row)
params_per_pop = {
    pop: set(labels_by_pop[pop].keys()) for pop in labels_by_pop
}
shared_params = set.intersection(*params_per_pop.values()) if params_per_pop else set()

# further restrict to params actually present in the loaded samples for
# every combo x population pair that was loaded
fully_shared = []
for p in shared_params:
    sub = df[df["param"] == p]
    n_combo_pop_pairs = sub[["combo", "population"]].drop_duplicates().shape[0]
    n_expected = df[["combo", "population"]].drop_duplicates().shape[0]
    if n_combo_pop_pairs == n_expected:
        fully_shared.append(p)
    else:
        print(f"[info] '{p}' dropped from shared set: only in "
              f"{n_combo_pop_pairs}/{n_expected} combo x population cells")

# preserve canonical order from population A's label dict where possible
canon_order = list(labels_by_pop.get("A", labels_by_pop.get("B", {})).keys())
shared_params_ordered = [p for p in canon_order if p in fully_shared]

if not shared_params_ordered:
    raise RuntimeError("No parameters are shared across both populations and all combos.")

# combo column order: respect SOURCE_SURVEY_COMBOS canonical ordering,
# restricted to combos actually present in the data
combo_order = [c for c in SOURCE_SURVEY_COMBOS if c in combos_present]

df["combo"] = pd.Categorical(df["combo"], categories=combo_order, ordered=True)

# --- Plot: one row per parameter, x-axis = combo, split violin by pop --
n_rows = len(shared_params_ordered)
fig, axes = plt.subplots(n_rows, 1, figsize=(2.2 * len(combo_order), 3.2 * n_rows),
                          squeeze=False)
axes = axes[:, 0]

for row_i, param in enumerate(shared_params_ordered):
    ax = axes[row_i]
    sub = df[df["param"] == param]

    sns.violinplot(
        data=sub, x="combo", y="value", hue="population",
        split=True, inner="quart", ax=ax, cut=0,
        palette=POP_COLORS, order=combo_order, hue_order=POPS,
        linewidth=1.0, saturation=0.9,
    )

    # truth markers per population (they can differ between A and B)
    for i, combo in enumerate(combo_order):
        for pop in POPS:
            t = truths_by_pop.get(pop, {}).get(param)
            if t is None:
                continue
            # small dash offset to each half of the split violin
            dx = -0.18 if pop == "A" else 0.18
            ax.plot([i + dx - 0.12, i + dx + 0.12], [t, t],
                    color=POP_COLORS[pop], lw=2.2, solid_capstyle="round",
                    zorder=5)

    label = labels_by_pop.get("A", labels_by_pop.get("B"))[param]
    ax.set_ylabel(label, fontsize=16)
    ax.set_xlabel("")
    ax.tick_params(axis='x', labelrotation=25, labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    if row_i == 0:
        ax.legend(title="Population", loc='upper right', fontsize=12, title_fontsize=13)
    else:
        ax.get_legend().remove()

plt.subplots_adjust(hspace=0.55)
outpath = f"{outdir}/{args.GALAXIES}_{scenario_tag}_violin_grid.png"
plt.savefig(outpath, dpi=300, bbox_inches="tight")
print(f"Saved: {outpath}")