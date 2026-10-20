"""
Impact of the merger-rate model on simulated EMRI / TDE(ZTF) detection counts,
M1-M8 scenarios, boxplots over the realisations in each universe_report_all.csv.
Baseline is excluded. A dashed line marks the observed ZTF TDE count (TDE panel).

Usage:
    python plot_merger_model_impact.py [--obs_tde_ztf N]   # default: count in real_tde_detections.npz
                                                           # (written by data_generation/pull_real_tde_catalog.py)
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from plot_style import setup_style, savefig_all, fig_size, TEXT_W, SCEN_LABEL

REAL_TDE_NPZ = "./data/astrophysical_data/real_tde_detections.npz"
DATA = "./data/DATA_FOR_MERGER/universe_report"

MODELS = ["mundy17", "duan24", "oleary21"]
MODEL_LABEL = {"mundy17": "Mundy+17", "duan24": "Duan+25", "oleary21": "O'Leary+21"}
MODEL_COLOR = {"mundy17": "#8a8a86", "duan24": "#4a3aa7", "oleary21": "#eda100"}
OBS_COLOR = "#882255"

# M1..M8 <-> (mbh_mass_method, nucleation_method, occupation_method)
SCENARIOS = [
    ("M1", "use_lgMgal", "hannah", "uniform"),
    ("M2", "use_sigma", "hannah", "uniform"),
    ("M3", "use_lgMgal", "neumayer", "uniform"),
    ("M4", "use_sigma", "neumayer", "uniform"),
    ("M5", "use_lgMgal", "hannah", "varying"),
    ("M6", "use_sigma", "hannah", "varying"),
    ("M7", "use_lgMgal", "neumayer", "varying"),
    ("M8", "use_sigma", "neumayer", "varying"),
]


def collect(dfs, col):
    out = {}
    for m in MODELS:
        df = dfs[m]
        out[m] = [df[(df.mbh_mass_method == mbh) & (df.nucleation_method == nuc)
                     & (df.occupation_method == occ)][col].values
                  for _, mbh, nuc, occ in SCENARIOS]
    return out


def boxplot_panel(ax, data, ylabel):
    n_model = len(MODELS)
    width = 0.8 / n_model
    base = np.arange(len(SCENARIOS))
    for i, m in enumerate(MODELS):
        pos = base + (i - (n_model - 1) / 2) * width
        bp = ax.boxplot(data[m], positions=pos, widths=width * 0.9, patch_artist=True,
                        showfliers=True, flierprops=dict(marker='o', markersize=1.5, alpha=0.5),
                        boxprops=dict(linewidth=0.7), whiskerprops=dict(linewidth=0.7),
                        capprops=dict(linewidth=0.7))
        for box in bp['boxes']:
            box.set_facecolor(MODEL_COLOR[m]); box.set_alpha(0.6); box.set_edgecolor(MODEL_COLOR[m])
        for med in bp['medians']:
            med.set_color('black'); med.set_linewidth(1.0)
        for el in ('whiskers', 'caps'):
            for line in bp[el]:
                line.set_color(MODEL_COLOR[m])
    ax.set_xticks(base)
    ax.set_xticklabels([SCEN_LABEL[s[0]] for s in SCENARIOS])
    ax.set_yscale('log')
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.3, linewidth=0.4)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--obs_tde_ztf", type=float, default=None,
                   help="Override the observed ZTF TDE count (default: number of events in REAL_TDE_NPZ).")
    args = p.parse_args()
    if args.obs_tde_ztf is None:
        args.obs_tde_ztf = len(np.load(REAL_TDE_NPZ, allow_pickle=True)['ztf_mjd'])
    print(f"Observed ZTF TDE count: {args.obs_tde_ztf:g}")

    setup_style()
    dfs = {m: pd.read_csv(f"{DATA}/{m}/universe_report_all.csv") for m in MODELS}

    fig, axes = plt.subplots(2, 1, figsize=fig_size(TEXT_W, 0.62), sharex=True)
    boxplot_panel(axes[0], collect(dfs, "n_emri_passed"), r"$N_\mathrm{EMRI}$ (detected)")
    boxplot_panel(axes[1], collect(dfs, "n_tde_ztf_passed"), r"$N_\mathrm{TDE,\,ZTF}$ (detected)")
    axes[1].axhline(args.obs_tde_ztf, color=OBS_COLOR, ls='--', lw=1.2, zorder=5)
    axes[1].set_xlabel("population scenario")
    axes[0].set_ylim(top=axes[0].get_ylim()[1] * 2.2)
    axes[1].set_ylim(top=axes[1].get_ylim()[1] * 1.8)

    handles = [Patch(facecolor=MODEL_COLOR[m], edgecolor=MODEL_COLOR[m], alpha=0.6,
                     label=MODEL_LABEL[m]) for m in MODELS]
    axes[0].legend(handles=handles, loc='upper right', ncol=3)
    axes[1].legend(handles=[Line2D([0], [0], color=OBS_COLOR, ls='--', lw=1.2, label="observed (Hammerstein+23)")],
                   loc='upper right')

    fig.tight_layout()
    savefig_all(fig, "merger_model_impact_counts")
    plt.close(fig)
