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

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        fig, ax = plt.subplots(figsize=(10, 6))

    if plot_kind == "box":
        bp = ax.boxplot(data, tick_labels=order, showfliers=True, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#7FA8C9")
            patch.set_alpha(0.6)
    elif plot_kind == "violin":
        parts = ax.violinplot(data, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#7FA8C9")
            pc.set_alpha(0.6)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(order)
    else:
        raise ValueError("plot_kind must be 'box' or 'violin'")

    if real_value is not None:
        ax.axhline(real_value, color="crimson", ls="--", lw=2, label=real_label, zorder=5)
        ax.legend(loc="upper right", fontsize=9)

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel or column)
    ax.set_title(title or column)
    ax.grid(alpha=0.3, axis="y")
    return ax


MERGER_MODEL_CHOICES = ["mundy17", "oleary21", "duan24", "baseline"]

# Fixed color per merger model, reused across the grouped comparison plot.
MERGER_MODEL_COLORS = {
    "mundy17":  "#7FA8C9",
    "oleary21": "#E8A23D",
    "duan24":   "#6FB07F",
    "baseline": "#B07FC9",
}


def plot_grouped_count_comparison(dfs_by_model, column, real_value=None, real_label="Observed",
                                   title=None, ylabel=None, log_scale=True, ax=None):
    """
    Grouped box plot of `column`: one cluster of boxes per scenario (M1-M8),
    one box per merger model within each cluster (colored by MERGER_MODEL_COLORS),
    for side-by-side comparison of merger models on the same axes.

    dfs_by_model: dict {merger_model_name: dataframe} (already tagged with model_name column).
    """
    models = [m for m in MERGER_MODEL_CHOICES if m in dfs_by_model]
    order = _order_models(pd.concat(dfs_by_model.values(), ignore_index=True))

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))

    n_models = len(models)
    width = 0.8 / n_models
    group_centers = np.arange(1, len(order) + 1)

    for i, merger_model in enumerate(models):
        df = dfs_by_model[merger_model]
        data = [df.loc[df["model_name"] == m, column].dropna().values for m in order]
        positions = group_centers - 0.4 + width * (i + 0.5)
        bp = ax.boxplot(data, positions=positions, widths=width * 0.9,
                         showfliers=True, patch_artist=True)
        color = MERGER_MODEL_COLORS.get(merger_model, None)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        # dummy handle for the legend (box patches themselves aren't legend-friendly)
        ax.plot([], [], color=color, lw=6, alpha=0.7, label=merger_model)

    if real_value is not None:
        ax.axhline(real_value, color="crimson", ls="--", lw=2, label=real_label, zorder=5)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(order)
    ax.set_xlim(0.3, len(order) + 0.7)
    ax.legend(loc="upper right", fontsize=9, title="Merger model")
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel or column)
    ax.set_title(title or column)
    ax.grid(alpha=0.3, axis="y")
    return ax


if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument("--merger_model", type=str, choices=MERGER_MODEL_CHOICES,
                    help="Galaxy merger-rate model whose universe_report/<merger_model>/ CSVs are plotted. "
                         "Omit (or pass --all) to also produce the combined all-merger-model comparison plot.")
    _p.add_argument("--all", action="store_true",
                    help="Also (or instead, if --merger_model is omitted) plot every merger model's scenarios "
                         "side-by-side in one grouped comparison plot.")
    _args = _p.parse_args()
    if _args.merger_model is None and not _args.all:
        _p.error("pass --merger_model <name> and/or --all")

    BASE_DIR = "./data/astrophysical_data/100000/PLOTS/universe_report"

    if _args.merger_model is not None:
        REPORT_DIR = f"{BASE_DIR}/{_args.merger_model}"
        df = load_all_scenarios(REPORT_DIR)

        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        plot_count_comparison(
            df, "n_emri_passed", real_value=None,
            title=f"Simulated EMRI counts by scenario ({_args.merger_model})",
            ylabel="EMRI count", log_scale=True, ax=axes,
        )
        plt.tight_layout()
        out_path = f"{REPORT_DIR}/scenario_comparison_boxplots.png"
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {out_path}")

    if _args.all:
        dfs_by_model = {}
        for _mm in MERGER_MODEL_CHOICES:
            try:
                dfs_by_model[_mm] = load_all_scenarios(f"{BASE_DIR}/{_mm}")
            except FileNotFoundError as e:
                print(f"[SKIP] {_mm}: {e}")
        if not dfs_by_model:
            raise FileNotFoundError(f"No merger-model universe_report directories found under {BASE_DIR}")

        fig, axes = plt.subplots(1, 1, figsize=(14, 6))
        plot_grouped_count_comparison(
            dfs_by_model, "n_emri_passed", real_value=None,
            title="Simulated EMRI counts by scenario -- all merger models",
            ylabel="EMRI count", log_scale=True, ax=axes,
        )
        plt.tight_layout()
        out_path = f"{BASE_DIR}/scenario_comparison_boxplots_all_merger_models.png"
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {out_path}")