#!/usr/bin/env python
"""
Step 1 of the realistic-sweep pipeline.

For every (scenario M1-M8, TDE lightcurve model) pair, thin the ZTF
detected-event catalogs (data_generation/calculate_SNR_TDE.py output) down
towards the real observed ZTF TDE count (43, Yao+23), by binomial thinning
applied per redshift bin, with a single keep-probability per (scenario,
model) shared across all universes:

    ratio = min(1, real_count / mean_original_count_across_universes)

Outputs (under ASTROPHYSICAL_DATA_DIR/{GALAXIES}/):
  realistic/TDE_SNRs_ZTF_{scenario}_{tde_model}_{u}.h5   -- thinned catalogs
  realistic/ratio_summary_ZTF.csv                        -- one row per (scenario, model)
  realistic/per_bin_removed_ZTF.csv                      -- one row per (scenario, model, redshift bin)

ratio_summary_ZTF.csv is consumed by sweep_thin_TDE_LSST.py (same ratio,
reused verbatim) and sweep_thin_EMRI.py (averaged over the 4 TDE models).
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.abspath("./"))
import numpy as np

from common_realistic import (
    MERGER_MODEL_CHOICES, set_merger_model,
    ASTROPHYSICAL_DATA_DIR, GALAXIES_DEFAULT, N_REDSHIFT_BINS_DEFAULT,
    SCENARIOS, SCENARIO_NAME_MAP, TDE_MODELS,
    discover_universes, load_h5_catalog, load_real_ztf_count,
    redshift_bin_edges, rng_for, save_thinned_h5, scenario_tag,
    thin_by_redshift_bin, tde_snr_path,
)

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES", type=int, default=GALAXIES_DEFAULT)
parser.add_argument("--merger_model", type=str, required=True, choices=MERGER_MODEL_CHOICES,
                    help="Galaxy merger-rate model whose universes are processed.")
parser.add_argument("--N_BINS", type=int, default=N_REDSHIFT_BINS_DEFAULT)
args = parser.parse_args()
set_merger_model(args.merger_model)

GALAXIES = args.GALAXIES
REAL_COUNT = load_real_ztf_count()
print(f"Real observed ZTF TDE count (target): {REAL_COUNT}")

out_dir = os.path.join(ASTROPHYSICAL_DATA_DIR, str(GALAXIES), "realistic")
os.makedirs(out_dir, exist_ok=True)

ratio_rows = []
per_bin_rows = []

for scenario in SCENARIOS:
    tag = scenario_tag(scenario)
    label = SCENARIO_NAME_MAP[scenario]
    for tde_model in TDE_MODELS:
        glob_pat = tde_snr_path(ASTROPHYSICAL_DATA_DIR, GALAXIES, "ZTF", scenario, tde_model, "*")
        universes = discover_universes(glob_pat)
        if not universes:
            print(f"[SKIP] {label} ({tag}) / {tde_model}: no ZTF universes found")
            continue

        # Pass 1: load every universe's detected events, get original counts
        per_universe = {}
        for u in universes:
            path = tde_snr_path(ASTROPHYSICAL_DATA_DIR, GALAXIES, "ZTF", scenario, tde_model, u)
            data, attrs = load_h5_catalog(path)
            per_universe[u] = (data, attrs)

        orig_counts = np.array([len(d["z_gal"]) for d, _ in per_universe.values()])
        mean_orig = float(orig_counts.mean())
        ratio = min(1.0, REAL_COUNT / mean_orig) if mean_orig > 0 else 1.0

        flag = "no_reduction (already <= target)" if ratio >= 1.0 else "thinned"
        print(f"{label} ({tag}) / {tde_model}: mean_orig={mean_orig:.1f} over "
              f"{len(universes)} universes -> ratio={ratio:.4f} [{flag}]")

        # Common bin edges from ALL universes' pooled z, for this (scenario, model)
        pooled_z = np.concatenate([d["z_gal"] for d, _ in per_universe.values()]) \
            if per_universe else np.array([])
        bin_edges = redshift_bin_edges(pooled_z, n_bins=args.N_BINS)

        total_removed = 0
        total_original = 0
        agg_removed = np.zeros(args.N_BINS, dtype=int)
        agg_original = np.zeros(args.N_BINS, dtype=int)

        for u in universes:
            data, attrs = per_universe[u]
            rng = rng_for("ZTF", tag, tde_model, u)
            keep_mask, per_bin_removed, per_bin_original = thin_by_redshift_bin(
                data["z_gal"], ratio, bin_edges, rng)

            out_path = tde_snr_path(ASTROPHYSICAL_DATA_DIR, GALAXIES, "ZTF", scenario,
                                     tde_model, u, realistic=True)
            save_thinned_h5(out_path, data, attrs, keep_mask, extra_attrs={
                "n_events_passed_original": len(data["z_gal"]),
                "reduction_ratio": ratio,
                "real_target_count": REAL_COUNT,
            })

            total_removed += int(per_bin_removed.sum())
            total_original += int(per_bin_original.sum())
            agg_removed += per_bin_removed
            agg_original += per_bin_original

        ratio_rows.append({
            "model_name": label, "mbh_mass_method": scenario[0],
            "nucleation_method": scenario[1], "occupation_method": scenario[2],
            "tde_model": tde_model, "n_universes": len(universes),
            "mean_original_count": mean_orig, "real_target_count": REAL_COUNT,
            "reduction_ratio": ratio, "total_removed": total_removed,
            "total_original": total_original,
        })

        for b in range(args.N_BINS):
            per_bin_rows.append({
                "model_name": label, "tde_model": tde_model,
                "z_bin_lo": bin_edges[b], "z_bin_hi": bin_edges[b + 1],
                "n_original": int(agg_original[b]), "n_removed": int(agg_removed[b]),
            })

ratio_csv = os.path.join(out_dir, f"ratio_summary_ZTF_{args.merger_model}.csv")
with open(ratio_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(ratio_rows[0].keys()))
    writer.writeheader()
    writer.writerows(ratio_rows)
print(f"Saved: {ratio_csv}")

bin_csv = os.path.join(out_dir, f"per_bin_removed_ZTF_{args.merger_model}.csv")
with open(bin_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(per_bin_rows[0].keys()))
    writer.writeheader()
    writer.writerows(per_bin_rows)
print(f"Saved: {bin_csv}")
