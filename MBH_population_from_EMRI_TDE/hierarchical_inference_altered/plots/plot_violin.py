#!/usr/bin/env python
"""
Violin-style hyperposterior comparisons across source/survey combos.

Three products, selected via --MODE:

  single   ONE population, ONE scenario -- one violin panel per shared
           hyperparameter, x-axis = source/survey combo.
  grouped  BOTH populations, ONE scenario -- split-violin (A vs B) per
           shared hyperparameter, x-axis = source/survey combo. (This is
           the "individual per scenario" product.)
  all      BOTH populations, EVERY scenario pooled together into one
           split-violin figure -- the grand combined view. Truth markers
           are omitted here since the injected truth differs per scenario.

--MODE single/grouped need --mbh_mass_method/--nucleation_method/--occupation_method.
--MODE all takes no scenario args (pools every entry in utils.SCENARIO_KEY).

Usage:
    python plots/plot_violin.py --MODE single  --GALAXIES 100000 --POPULATION A \\
        --mbh_mass_method use_sigma --nucleation_method hannah --occupation_method uniform
    python plots/plot_violin.py --MODE grouped --GALAXIES 100000 \\
        --mbh_mass_method use_sigma --nucleation_method hannah --occupation_method uniform
    python plots/plot_violin.py --MODE all --GALAXIES 100000
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import load_available_combos, get_latex_labels, get_true_x_flat, SOURCE_SURVEY_COMBOS, SCENARIO_KEY

matplotlib.rc('font', family='serif', serif=['Computer Modern'], size=18)
matplotlib.rc('text', usetex=True)

POPS = ["A", "B"]
POP_COLORS = {"A": "#3b6fa0", "B": "#c0603d"}

parser = argparse.ArgumentParser(description="Violin-style hyperposterior comparisons")
parser.add_argument('--MODE', choices=['single', 'grouped', 'all'], required=True)
parser.add_argument('--GALAXIES', required=True)
parser.add_argument('--POPULATION', choices=['A', 'B'], help="required for --MODE single")
parser.add_argument("--mbh_mass_method", choices=["use_sigma", "use_lgMgal"], help="required for single/grouped")
parser.add_argument("--nucleation_method", choices=["hannah", "neumayer"], help="required for single/grouped")
parser.add_argument("--occupation_method", choices=["uniform", "varying"], help="required for single/grouped")
args = parser.parse_args()

if args.MODE == "single" and args.POPULATION is None:
    parser.error("--MODE single requires --POPULATION")
if args.MODE in ("single", "grouped"):
    missing = [n for n in ("mbh_mass_method", "nucleation_method", "occupation_method")
               if getattr(args, n) is None]
    if missing:
        parser.error(f"--MODE {args.MODE} requires --{'/--'.join(missing)}")

base_dir = f"./data/altered_inference_data/pseudo_inference_data/{args.GALAXIES}"


def save_fig(fig, base_path):
    fig.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base_path}.pdf", bbox_inches="tight")
    print(f"Saved: {base_path}.png / .pdf")


def load_scenario(pop, mbh, nuc, occ):
    """Returns (records_df_or_None, labels_dict, truths_dict, combo_labels_present)."""
    combos = load_available_combos(base_dir, pop, mbh, nuc, occ)
    if not combos:
        print(f"[warn] no combos loaded for population {pop} ({mbh}_{nuc}_{occ}) -- skipping")
        return None, {}, {}, set()

    labels_dict = get_latex_labels(pop)
    truths_dict = get_true_x_flat(pop, mbh, nuc, occ)

    records = []
    combo_labels_present = set()
    for c in combos:
        combo_labels_present.add(c["label"])
        for key, samples in c["samples"].items():
            records.append(pd.DataFrame({
                "value": samples, "param": key, "combo": c["label"], "population": pop,
            }))
    df = pd.concat(records, ignore_index=True) if records else None
    return df, labels_dict, truths_dict, combo_labels_present


def shared_params_for(df, labels_by_pop, pops_present):
    """Params present for every (combo, population) cell actually in df,
    restricted to pops_present, in canonical order from the first available pop."""
    params_per_pop = {pop: set(labels_by_pop[pop].keys()) for pop in pops_present}
    shared = set.intersection(*params_per_pop.values()) if params_per_pop else set()

    fully_shared = []
    n_expected = df[["combo", "population"]].drop_duplicates().shape[0]
    for p in shared:
        sub = df[df["param"] == p]
        n_present = sub[["combo", "population"]].drop_duplicates().shape[0]
        if n_present == n_expected:
            fully_shared.append(p)
        else:
            print(f"[info] '{p}' dropped from shared set: only in {n_present}/{n_expected} combo x population cells")

    canon_order = list(labels_by_pop[pops_present[0]].keys())
    return [p for p in canon_order if p in fully_shared]


def combo_order_for(combo_labels_present):
    return [label for (_src, _srv, label, _color) in SOURCE_SURVEY_COMBOS if label in combo_labels_present]


def make_violin_grid(df, shared_params, combo_order, labels_by_pop, pops, split,
                      truths_by_pop=None, title_suffix=""):
    n_rows = len(shared_params)
    fig, axes = plt.subplots(n_rows, 1, figsize=(2.2 * len(combo_order), 3.2 * n_rows), squeeze=False)
    axes = axes[:, 0]

    df = df.copy()
    df["combo"] = pd.Categorical(df["combo"], categories=combo_order, ordered=True)

    for row_i, param in enumerate(shared_params):
        ax = axes[row_i]
        sub = df[df["param"] == param]

        if split and len(pops) == 2:
            sns.violinplot(
                data=sub, x="combo", y="value", hue="population",
                split=True, inner="quart", ax=ax, cut=0,
                palette=POP_COLORS, order=combo_order, hue_order=pops,
                linewidth=1.0, saturation=0.9,
            )
            if truths_by_pop is not None:
                for i, combo in enumerate(combo_order):
                    for pop in pops:
                        t = truths_by_pop.get(pop, {}).get(param)
                        if t is None:
                            continue
                        dx = -0.18 if pop == "A" else 0.18
                        ax.plot([i + dx - 0.12, i + dx + 0.12], [t, t],
                                color=POP_COLORS[pop], lw=2.2, solid_capstyle="round", zorder=5)
            if row_i == 0:
                ax.legend(title="Population", loc='upper right', fontsize=12, title_fontsize=13)
            else:
                ax.get_legend().remove()
        else:
            pop = pops[0]
            sns.violinplot(
                data=sub, x="combo", y="value", ax=ax, cut=0,
                order=combo_order, color=POP_COLORS.get(pop, "#3b6fa0"),
                linewidth=1.0, saturation=0.9,
            )
            if truths_by_pop is not None:
                t = truths_by_pop.get(pop, {}).get(param)
                if t is not None:
                    ax.axhline(t, color="black", lw=1.8, linestyle="--", zorder=5)

        label = labels_by_pop[pops[0]][param]
        ax.set_ylabel(label, fontsize=16)
        ax.set_xlabel("")
        ax.tick_params(axis='x', labelrotation=25, labelsize=13)
        ax.tick_params(axis='y', labelsize=13)

    plt.subplots_adjust(hspace=0.55)
    return fig


# ════════════════════════════════════════════════════════════════════════════
if args.MODE == "single":
    scenario_tag = f"{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}"
    outdir = f"{base_dir}/PLOTS/violin-posteriors/{scenario_tag}"
    os.makedirs(outdir, exist_ok=True)

    df, labels_dict, truths_dict, combo_labels = load_scenario(
        args.POPULATION, args.mbh_mass_method, args.nucleation_method, args.occupation_method
    )
    if df is None:
        raise RuntimeError(f"No data loaded for population {args.POPULATION} ({scenario_tag}).")

    shared_params = shared_params_for(df, {args.POPULATION: labels_dict}, [args.POPULATION])
    if not shared_params:
        raise RuntimeError("No parameters available to plot.")
    combo_order = combo_order_for(combo_labels)

    fig = make_violin_grid(
        df, shared_params, combo_order, {args.POPULATION: labels_dict},
        pops=[args.POPULATION], split=False, truths_by_pop={args.POPULATION: truths_dict},
    )
    base_name = f"{outdir}/{args.GALAXIES}_pop_{args.POPULATION}_{scenario_tag}_violin"
    save_fig(fig, base_name)


elif args.MODE == "grouped":
    scenario_tag = f"{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}"
    outdir = f"{base_dir}/PLOTS/violin-posteriors/{scenario_tag}"
    os.makedirs(outdir, exist_ok=True)

    dfs, labels_by_pop, truths_by_pop, combo_labels_present = [], {}, {}, set()
    for pop in POPS:
        d, labels_dict, truths_dict, combos_here = load_scenario(
            pop, args.mbh_mass_method, args.nucleation_method, args.occupation_method
        )
        if d is None:
            continue
        dfs.append(d)
        labels_by_pop[pop] = labels_dict
        truths_by_pop[pop] = truths_dict
        combo_labels_present |= combos_here

    if not dfs:
        raise RuntimeError(f"No data loaded for either population ({scenario_tag}).")
    df = pd.concat(dfs, ignore_index=True)
    pops_present = list(labels_by_pop.keys())

    shared_params = shared_params_for(df, labels_by_pop, pops_present)
    if not shared_params:
        raise RuntimeError("No parameters shared across both populations and all combos.")
    combo_order = combo_order_for(combo_labels_present)

    fig = make_violin_grid(
        df, shared_params, combo_order, labels_by_pop,
        pops=pops_present, split=True, truths_by_pop=truths_by_pop,
    )
    base_name = f"{outdir}/{args.GALAXIES}_{scenario_tag}_violin_grouped"
    save_fig(fig, base_name)


else:  # all
    outdir = f"{base_dir}/PLOTS/violin-posteriors/ALL_SCENARIOS"
    os.makedirs(outdir, exist_ok=True)

    dfs, labels_by_pop, combo_labels_present = [], {}, set()
    for pop in POPS:
        for (mbh, nuc, occ) in SCENARIO_KEY:
            d, labels_dict, _truths, combos_here = load_scenario(pop, mbh, nuc, occ)
            if d is None:
                continue
            d = d.copy()
            d["scenario"] = f"{mbh}_{nuc}_{occ}"
            dfs.append(d)
            labels_by_pop[pop] = labels_dict
            combo_labels_present |= combos_here

    if not dfs:
        raise RuntimeError("No data loaded for any population/scenario -- nothing to plot.")
    df = pd.concat(dfs, ignore_index=True)
    pops_present = list(labels_by_pop.keys())

    # Truths differ per scenario -- pooled figure omits truth markers (split=True
    # branch handles truths_by_pop=None gracefully).
    shared_params = shared_params_for(df, labels_by_pop, pops_present)
    if not shared_params:
        raise RuntimeError("No parameters shared across both populations and all combos/scenarios.")
    combo_order = combo_order_for(combo_labels_present)

    fig = make_violin_grid(
        df, shared_params, combo_order, labels_by_pop,
        pops=pops_present, split=True, truths_by_pop=None,
    )
    base_name = f"{outdir}/{args.GALAXIES}_violin_all_scenarios"
    save_fig(fig, base_name)
