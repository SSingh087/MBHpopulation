#!/usr/bin/env python
"""
Realistic-catalog version of data_generation/compare_TDE_detection.py's
plots: same per-scenario envelope plots and per-TDE-model combined-scenario
plots, but reading the thinned catalogs from sweep_thin_TDE_ZTF.py /
sweep_thin_TDE_LSST.py instead of the raw over-predicted ones.

Reuses the plotting/aggregation logic from data_generation/compare_TDE_detection.py
directly (same math, only the input file paths change) -- see that module
for plot_scenario_envelope / plot_combined_models / aggregate_scenario.

Saves to ASTROPHYSICAL_DATA_DIR (same directory data_generation's plots use),
with a "_realistic" suffix:
  combined_per_model_cumulative_envelope_realistic_{survey}_{scenario}_{tde_model}.png
  combined_cumulative_realistic_{survey}_{merger_model}_{tde_model}.png
"""
import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_GEN_DIR = os.path.join(REPO_ROOT, "data_generation")
sys.path.insert(0, REPO_ROOT)       # compare_TDE_detection.py does `from utils import ...`,
sys.path.insert(0, DATA_GEN_DIR)    # which lives at the repo root, not in data_generation/
sys.path.insert(0, os.path.abspath("./"))

import numpy as np
import matplotlib.pyplot as plt

from compare_TDE_detection import (
    build_year_grid, cumulative_curve_on_grid, plot_scenario_envelope,
    plot_combined_models, mjd_to_year,
)
from common_realistic import (
    MERGER_MODEL_CHOICES, set_merger_model,
    ASTROPHYSICAL_DATA_DIR, GALAXIES_DEFAULT, SCENARIOS, SCENARIO_NAME_MAP,
    TDE_MODELS, discover_universes, load_h5_catalog, load_real_ztf_count,
    scenario_tag, tde_snr_path,
)

SURVEYS = ["ZTF", "LSST"]
N_BINS_PER_YEAR = 12


def load_t_detected_realistic(galaxies, survey, scenario, tde_model, universe):
    path = tde_snr_path(ASTROPHYSICAL_DATA_DIR, galaxies, survey, scenario, tde_model,
                         universe, realistic=True)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data, attrs = load_h5_catalog(path)
    return data.get("t0_mjd_transient", np.array([]))


def aggregate_scenario_realistic(galaxies, survey, scenario, tde_model, n_bins_per_year=12):
    tag = scenario_tag(scenario)
    glob_pat = tde_snr_path(ASTROPHYSICAL_DATA_DIR, galaxies, survey, scenario, tde_model,
                             "*", realistic=True)
    universes = discover_universes(glob_pat)
    if not universes:
        raise RuntimeError(f"No realistic universes found for {tag}/{tde_model}/{survey}")

    per_universe_years = []
    per_universe_t = []
    for u in universes:
        t_mjd = load_t_detected_realistic(galaxies, survey, scenario, tde_model, u)
        per_universe_t.append(t_mjd)
        if len(t_mjd) > 0:
            per_universe_years.append(mjd_to_year(t_mjd))

    if not per_universe_years:
        raise RuntimeError(f"No usable universes with >=1 detection for {tag}/{tde_model}/{survey}")

    year_grid = build_year_grid(per_universe_years, n_bins_per_year=n_bins_per_year)
    max_data_year = max(y.max() for y in per_universe_years)
    curves = np.stack([cumulative_curve_on_grid(t, year_grid) for t in per_universe_t])
    return year_grid, curves, max_data_year


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--GALAXIES", type=int, default=GALAXIES_DEFAULT)
    parser.add_argument("--merger_model", type=str, required=True, choices=MERGER_MODEL_CHOICES,
                        help="Galaxy merger-rate model whose universes are processed.")
    args = parser.parse_args()
    set_merger_model(args.merger_model)
    GALAXIES = args.GALAXIES

    REAL_COUNT = load_real_ztf_count()
    _real = np.load(f"{ASTROPHYSICAL_DATA_DIR}/real_tde_detections.npz", allow_pickle=True)
    REAL_T_MJD = {"ZTF": _real["ztf_mjd"], "LSST": _real["lsst_mjd"]}

    results = {}
    for survey in SURVEYS:
        for tde_model in TDE_MODELS:
            for scenario in SCENARIOS:
                tag = scenario_tag(scenario)
                label = SCENARIO_NAME_MAP[scenario]
                try:
                    year_grid, curves, max_data_year = aggregate_scenario_realistic(
                        GALAXIES, survey, scenario, tde_model, n_bins_per_year=N_BINS_PER_YEAR)
                except RuntimeError as e:
                    print(f"[SKIP] {survey}/{tag}/{tde_model}: {e}")
                    continue

                results[(scenario[0], scenario[1], scenario[2], tde_model, survey)] = \
                    (year_grid, curves, 0, max_data_year)

                ax = plot_scenario_envelope(
                    year_grid, curves, survey, f"{tag} ({tde_model}) [realistic]",
                    real_t_mjd=REAL_T_MJD[survey],
                )
                plt.tight_layout()
                out_path = (f"{ASTROPHYSICAL_DATA_DIR}/combined_per_model_cumulative_envelope_"
                            f"realistic_{survey}_{tag}_{tde_model}.png")
                plt.savefig(out_path, dpi=200)
                plt.close(ax.figure)
                print(f"Saved: {out_path} ({curves.shape[0]} universes)")

    for survey in SURVEYS:
        for tde_model in TDE_MODELS:
            try:
                ax = plot_combined_models(
                    results, survey, tde_model, palette="colorblind", show_band=(50, 95),
                    real_t_mjd=REAL_T_MJD[survey], real_label="Observed (TDE)",
                )
            except RuntimeError as e:
                print(f"[SKIP combined] {survey}/{tde_model}: {e}")
                continue

            survey_entries = [
                (year_grid, max_data_year)
                for (mbh, nuc, occ, model, surv), (year_grid, curves, n_missing, max_data_year)
                in results.items() if surv == survey and model == tde_model
            ]
            if survey_entries:
                x_min = min(grid.min() for grid, _ in survey_entries)
                x_max = max(mdy for _, mdy in survey_entries)
                if REAL_T_MJD[survey] is not None and len(REAL_T_MJD[survey]) > 0:
                    real_years = mjd_to_year(REAL_T_MJD[survey])
                    x_min = min(x_min, float(np.min(real_years)))
                    x_max = max(x_max, float(np.max(real_years)))
                ax.set_xlim(x_min, x_max)
            ax.set_yscale("log")
            ax.set_ylim(bottom=1)
            ax.set_title(ax.get_title() + " [realistic]")
            plt.tight_layout()
            out_path = f"{ASTROPHYSICAL_DATA_DIR}/combined_cumulative_realistic_{survey}_{args.merger_model}_{tde_model}.png"
            plt.savefig(out_path, dpi=200)
            plt.close(ax.figure)
            print(f"Saved: {out_path}")
