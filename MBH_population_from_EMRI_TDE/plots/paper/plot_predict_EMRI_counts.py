"""
Compare all 8 scenarios' per-universe TDE/EMRI counts (from universe_report_*.csv)
against the real observed TDE count, and derive the ratio-scaled EMRI prediction
per scenario.

Box plot chosen over violin plot: with counts this skewed (Neumayer scenarios have
CV% > 300% in tab:counts_summary, and many near-zero-TDE universes), a violin's KDE
smoothing can imply density below zero or bimodal structure that isn't really there
for discrete, heavily-skewed count data. A box plot (median/IQR/whiskers + outlier
points) is more honest about what 100 discrete samples actually show. Swap
`plot_kind="violin"` below if you disagree once you see both.

Expects one CSV per scenario, e.g.:
    universe_report_use_lgMgal_hannah_uniform.csv
    universe_report_use_sigma_hannah_uniform.csv
    ...
each with columns: universe, mbh_mass_method, nucleation_method, occupation_method,
n_emri_passed, n_tde_ztf_passed, n_tde_lsst_passed, n_total_detected
"""

import sys, os
sys.path.insert(0, os.path.abspath('./'))

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_style import setup_style, savefig_all, fig_size, COL_W, SCEN_LABEL

# ------------------------------------------------------------------
# Model-name mapping, matching Table tab:model_names in the draft
# ------------------------------------------------------------------
MODEL_NAME_MAP = {
    ("use_lgMgal", "hannah", "uniform"):   "M1",
    ("use_sigma",  "hannah", "uniform"):   "M2",
    ("use_lgMgal", "neumayer", "uniform"): "M3",
    ("use_sigma",  "neumayer", "uniform"): "M4",
    ("use_lgMgal", "hannah", "varying"):   "M5",
    ("use_sigma",  "hannah", "varying"):   "M6",
    ("use_lgMgal", "neumayer", "varying"): "M7",
    ("use_sigma",  "neumayer", "varying"): "M8",
}


def load_all_scenarios(report_dir, pattern="universe_report_*.csv"):
    """Load and concatenate all per-scenario universe_report CSVs, tagging each
    row with a compact model name (M1-M8) from MODEL_NAME_MAP."""
    paths = sorted(glob.glob(f"{report_dir}/{pattern}"))
    # exclude the pre-combined "universe_report_all.csv" if present, to avoid
    # double-counting every universe
    paths = [p for p in paths if not p.endswith("universe_report_all.csv")]
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern} in {report_dir}")

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        key = (df["mbh_mass_method"].iloc[0], df["nucleation_method"].iloc[0],
               df["occupation_method"].iloc[0])
        model_name = MODEL_NAME_MAP.get(key)
        if model_name is None:
            print(f"[WARN] {p}: scenario {key} not in MODEL_NAME_MAP -- "
                  f"using raw label instead of M1-M8.")
            model_name = "_".join(key)
        df["model_name"] = model_name
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(paths)} scenario files, {len(all_df)} universe-rows total.")
    return all_df


def add_derived_columns(df, real_tde_ztf_obs=None, real_tde_lsst_obs=None):
    """Add per-universe EMRI/TDE ratios and (if real counts given) the
    ratio-scaled EMRI prediction for each universe."""
    df = df.copy()

    # Ratios: NaN where TDE count is 0 (undefined), matching tab:ratio_summary's
    # convention of excluding zero-TDE realizations from the ratio's mean.
    df["ratio_emri_tde_ztf"] = np.where(
        df["n_tde_ztf_passed"] > 0,
        df["n_emri_passed"] / df["n_tde_ztf_passed"],
        np.nan,
    )
    df["ratio_emri_tde_lsst"] = np.where(
        df["n_tde_lsst_passed"] > 0,
        df["n_emri_passed"] / df["n_tde_lsst_passed"],
        np.nan,
    )

    if real_tde_ztf_obs is not None:
        df["emri_pred_from_ztf"] = df["ratio_emri_tde_ztf"] * real_tde_ztf_obs
    if real_tde_lsst_obs is not None:
        df["emri_pred_from_lsst"] = df["ratio_emri_tde_lsst"] * real_tde_lsst_obs

    return df


def _order_models(df):
    present = df["model_name"].unique().tolist()
    ordered = [m for m in ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"] if m in present]
    ordered += [m for m in present if m not in ordered]  # fallback labels, if any
    return ordered


def plot_count_comparison(df, column, real_value=None, real_label="Observed",
                           title=None, ylabel=None, log_scale=True,
                           plot_kind="box", ax=None):
    """
    Box (or violin) plot of `column` across scenarios (x-axis), with an optional
    horizontal reference line for the real observed value.
    """
    order = _order_models(df)
    data = [df.loc[df["model_name"] == m, column].dropna().values for m in order]

    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size(COL_W, 0.72))

    if plot_kind == "box":
        bp = ax.boxplot(data, tick_labels=[SCEN_LABEL.get(m, m) for m in order], showfliers=True, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#0072B2")
            patch.set_alpha(0.55)
            patch.set_edgecolor("#0072B2")
        for whisker in bp["whiskers"]:
            whisker.set_color("#0072B2")
        for cap in bp["caps"]:
            cap.set_color("#0072B2")
        for median in bp["medians"]:
            median.set_color("#D55E00")
            median.set_linewidth(1.2)
        for flier in bp["fliers"]:
            flier.set_markeredgecolor("#0072B2")
    elif plot_kind == "violin":
        parts = ax.violinplot(data, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#0072B2")
            pc.set_alpha(0.55)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels([SCEN_LABEL.get(m, m) for m in order])
    else:
        raise ValueError("plot_kind must be 'box' or 'violin'")

    if real_value is not None:
        ax.axhline(real_value, color="#882255", ls="--", lw=1.2, label=real_label, zorder=5)
        ax.legend(loc="upper right")

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("population scenario")
    ax.set_ylabel(ylabel or column.replace("_", r"\_"))
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3, axis="y", linewidth=0.4)
    return ax


if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument("--merger_model", type=str, required=True, choices=["mundy17", "oleary21", "duan24", "baseline"],
                    help="Galaxy merger-rate model whose universe_report/<merger_model>/ CSVs are plotted.")
    _mm = _p.parse_args().merger_model
    REPORT_DIR = ("./data/astrophysical_data/"
                   f"100000/PLOTS/universe_report/{_mm}")

    setup_style()

    df = load_all_scenarios(REPORT_DIR)

    fig, axes = plt.subplots(1, 1, figsize=fig_size(COL_W, 0.72))

    # Panel 1: simulated EMRI counts per scenario
    plot_count_comparison(
        df, "n_emri_passed", real_value=None,
        ylabel=r"EMRI count, $N_{\rm EMRI}$", log_scale=True, ax=axes,
    )

    plt.tight_layout()
    savefig_all(fig, "EMRI_counts_per_scenario")
    plt.close(fig)