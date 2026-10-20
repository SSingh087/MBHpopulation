#!/usr/bin/env python
"""
Violin comparison across every injected universe scenario, for ONE
population model at a time.

Population A and B use different parametrizations (Schechter+Beta-mixture
vs. power-law+Gaussian-mixture) -- their hyperparameters are NOT directly
comparable, so this deliberately never mixes A and B in one plot. (Model
comparison between A and B belongs in observable space -- see PPD.py.)

For a fixed population:
  - rows    = that population's own hyperparameters
  - x-axis  = the 8 injected universe scenarios, labeled M1..M8 per
              utils.MODEL_LABELS (matches the paper's model-name table)
  - hue     = the 7 source/survey cases, dodged side by side within each
              scenario tick (NOT split -- there are more than 2 to compare)
  - a black dash at each scenario tick marks that scenario's injected truth
    (truth depends only on population + scenario, not on case)

A case can be silenced from the overlay with --EXCLUDE_CASES, same as
posterior_plots.py.

Usage:
    python plots/plot_violin.py --GALAXIES 100000 --POPULATION A
    python plots/plot_violin.py --GALAXIES 100000 --POPULATION B --EXCLUDE_CASES "TDE (LSST)"
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import load_available_combos, get_latex_labels, get_true_x_flat, SOURCE_SURVEY_COMBOS, SCENARIO_KEY, MODEL_LABELS

matplotlib.rc('font', family='serif', serif=['Computer Modern'], size=18)
matplotlib.rc('text', usetex=True)

parser = argparse.ArgumentParser(description="Violin comparison across universe scenarios, one population at a time")
parser.add_argument('--GALAXIES', required=True)
parser.add_argument('--POPULATION', choices=['A', 'B'], required=True)
parser.add_argument("--EXCLUDE_CASES", type=str, default="",
                     help="Comma-separated combo labels to silence, e.g. 'TDE (LSST),EMRI+TDE (ALL)'.")
parser.add_argument("--CASES", type=str,
                     default="EMRI,EMRI+TDE (ZTF),EMRI+TDE (LSST),TDE (ALL)",
                     help="Comma-separated combo labels to keep (default: the four-case subset). "
                          "Pass 'all' for every case.")
args = parser.parse_args()
excluded_cases = {s.strip() for s in args.EXCLUDE_CASES.split(",") if s.strip()}
if args.CASES.strip().lower() != "all":
    keep = {s.strip() for s in args.CASES.split(",") if s.strip()}
    excluded_cases |= {label for (_s, _v, label, _c) in SOURCE_SURVEY_COMBOS if label not in keep}

base_dir = f"./data/pseudo_inference_data/{args.GALAXIES}"
outdir = f"{base_dir}/PLOTS/violin-posteriors"
os.makedirs(outdir, exist_ok=True)

labels_dict = get_latex_labels(args.POPULATION)
case_order = [label for (_src, _srv, label, _color) in SOURCE_SURVEY_COMBOS if label not in excluded_cases]

# scenario_tag is just the paper's M1..M8 name -- unambiguous and already
# gives the x-axis its correct order (unlike joining mbh/nuc/occ into a
# string, which is ambiguous since mbh itself contains underscores).
scenario_tuples = sorted(SCENARIO_KEY, key=lambda t: int(MODEL_LABELS[t][1:]))
scenario_order = [MODEL_LABELS[t] for t in scenario_tuples]

records = []
truths_by_scenario = {}  # "M#" -> {param: truth}
present_cells = set()    # (param, "M#", case) actually loaded

for (mbh, nuc, occ) in scenario_tuples:
    scen_tag = MODEL_LABELS[(mbh, nuc, occ)]
    combos = load_available_combos(base_dir, args.POPULATION, mbh, nuc, occ)
    combos = [c for c in combos if c["label"] not in excluded_cases]
    if not combos:
        print(f"[warn] no cases loaded for pop {args.POPULATION}, scenario {scen_tag} -- skipping")
        continue

    truths_by_scenario[scen_tag] = get_true_x_flat(args.POPULATION, mbh, nuc, occ)

    for c in combos:
        for key, samples in c["samples"].items():
            present_cells.add((key, scen_tag, c["label"]))
            records.append(pd.DataFrame({
                "value": samples, "param": key, "scenario": scen_tag, "case": c["label"],
            }))

if not records:
    raise RuntimeError(f"No data loaded for population {args.POPULATION} across any scenario -- nothing to plot.")

df = pd.concat(records, ignore_index=True)

scenarios_present = [s for s in scenario_order if any(s == sc for (_p, sc, _c) in present_cells)]
cases_present = [c for c in case_order if any(c == cc for (_p, _sc, cc) in present_cells)]

# A parameter row is drawn if ANY loaded (scenario, case) cell sampled it; cases
# that don't constrain it (e.g. TDE-only for spin/eccentricity) just leave a gap.
params_ordered = [p for p in labels_dict if any(pp == p for (pp, _sc, _cc) in present_cells)]

if not params_ordered:
    raise RuntimeError("No parameters found in any loaded scenario x case cell -- nothing to plot.")

df["scenario"] = pd.Categorical(df["scenario"], categories=scenarios_present, ordered=True)
df["case"] = pd.Categorical(df["case"], categories=cases_present, ordered=True)

# Palette keyed by case NAME (from utils). A positional list breaks on rows where some
# cases have no samples (TDE-only cases never sample spin/eccentricity): seaborn drops the
# empty hue levels and re-assigns colours by position, so EMRI+TDE violins were being
# drawn in the TDE colours.
pal = {c: col for (_s, _v, c, col) in SOURCE_SURVEY_COMBOS}

n_rows = len(params_ordered)
fig, axes = plt.subplots(n_rows, 1, figsize=(2.0 * len(scenarios_present), 3.2 * n_rows), squeeze=False)
axes = axes[:, 0]

n_cases = len(cases_present)
dodge_width = 0.8  # total width sns.violinplot spreads hues across, per its own convention

for row_i, param in enumerate(params_ordered):
    ax = axes[row_i]
    sub = df[df["param"] == param]
    row_cases = [c for c in cases_present if (sub["case"] == c).any()]  # only cases that sampled this param

    sns.violinplot(
        data=sub, x="scenario", y="value", hue="case", ax=ax, cut=0,
        order=scenarios_present, hue_order=row_cases, palette=pal,
        dodge=True, linewidth=0.8, saturation=0.9, inner="quart",
        density_norm="width", common_norm=False,
    )

    # One truth dash per scenario tick, spanning the dodged case cluster.
    for i, scen_tag in enumerate(scenarios_present):
        t = truths_by_scenario.get(scen_tag, {}).get(param)
        if t is None:
            continue
        half = dodge_width / 2
        ax.plot([i - half, i + half], [t, t], color="black", lw=2.0,
                linestyle="--", solid_capstyle="butt", zorder=5)

    label = labels_dict[param]
    ax.set_ylabel(label, fontsize=16)
    ax.set_xlabel("")
    ax.tick_params(axis='x', labelrotation=0, labelsize=14)
    ax.tick_params(axis='y', labelsize=13)
    if row_i == 0:
        ax.legend(title="Case", loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=12, title_fontsize=13,
                   ncol=1, frameon=False)
    else:
        ax.get_legend().remove()

plt.subplots_adjust(hspace=0.6)

base_name = f"{outdir}/{args.GALAXIES}_pop_{args.POPULATION}_violin_all_scenarios"
fig.savefig(f"{base_name}.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{base_name}.pdf", bbox_inches="tight")
print(f"Saved: {base_name}.png / .pdf")
