#!/usr/bin/env python
"""
Step 2 of the realistic-sweep pipeline.

No real LSST TDE catalog exists yet, so there's no direct target to thin
towards. Instead, reuse each (scenario, TDE model)'s ZTF-derived reduction
ratio (from sweep_thin_TDE_ZTF.py's ratio_summary_ZTF.csv) verbatim, and
apply the same per-redshift-bin binomial thinning to that scenario/model's
LSST catalogs -- the working assumption being that whatever makes the
simulated ZTF rate overshoot the real one (by that ratio) affects the LSST
catalog from the same universe realization proportionally.

Outputs (under ASTROPHYSICAL_DATA_DIR/{GALAXIES}/realistic/):
  TDE_SNRs_LSST_{scenario}_{tde_model}_{u}.h5   -- thinned catalogs
  per_bin_removed_LSST.csv                      -- per (scenario, model, z bin) removed counts
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
    discover_universes, load_h5_catalog, redshift_bin_edges, rng_for,
    save_thinned_h5, tde_snr_path, thin_by_redshift_bin,
)

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES", type=int, default=GALAXIES_DEFAULT)
parser.add_argument("--merger_model", type=str, required=True, choices=MERGER_MODEL_CHOICES,
                    help="Galaxy merger-rate model whose universes are processed.")
parser.add_argument("--N_BINS", type=int, default=N_REDSHIFT_BINS_DEFAULT)
args = parser.parse_args()
set_merger_model(args.merger_model)

GALAXIES = args.GALAXIES
out_dir = os.path.join(ASTROPHYSICAL_DATA_DIR, str(GALAXIES), "realistic")

ratio_csv = os.path.join(out_dir, f"ratio_summary_ZTF_{args.merger_model}.csv")
if not os.path.exists(ratio_csv):
    raise FileNotFoundError(f"{ratio_csv} not found -- run sweep_thin_TDE_ZTF.py first.")

import pandas as pd
ratios_df = pd.read_csv(ratio_csv)

per_bin_rows = []

for _, row in ratios_df.iterrows():
    scenario = (row["mbh_mass_method"], row["nucleation_method"], row["occupation_method"])
    tde_model = row["tde_model"]
    label = row["model_name"]
    ratio = float(row["reduction_ratio"])
    tag = "_".join(scenario)

    glob_pat = tde_snr_path(ASTROPHYSICAL_DATA_DIR, GALAXIES, "LSST", scenario, tde_model, "*")
    universes = discover_universes(glob_pat)
    if not universes:
        print(f"[SKIP] {label} ({tag}) / {tde_model}: no LSST universes found")
        continue

    per_universe = {}
    for u in universes:
        path = tde_snr_path(ASTROPHYSICAL_DATA_DIR, GALAXIES, "LSST", scenario, tde_model, u)
        per_universe[u] = load_h5_catalog(path)

    pooled_z = np.concatenate([d["z_gal"] for d, _ in per_universe.values()])
    bin_edges = redshift_bin_edges(pooled_z, n_bins=args.N_BINS)

    agg_removed = np.zeros(args.N_BINS, dtype=int)
    agg_original = np.zeros(args.N_BINS, dtype=int)

    for u in universes:
        data, attrs = per_universe[u]
        rng = rng_for("LSST", tag, tde_model, u)
        keep_mask, per_bin_removed, per_bin_original = thin_by_redshift_bin(
            data["z_gal"], ratio, bin_edges, rng)

        out_path = tde_snr_path(ASTROPHYSICAL_DATA_DIR, GALAXIES, "LSST", scenario,
                                 tde_model, u, realistic=True)
        save_thinned_h5(out_path, data, attrs, keep_mask, extra_attrs={
            "n_events_passed_original": len(data["z_gal"]),
            "reduction_ratio": ratio,
            "reduction_ratio_source": "ZTF-derived, same (scenario, tde_model)",
        })

        agg_removed += per_bin_removed
        agg_original += per_bin_original

    print(f"{label} ({tag}) / {tde_model}: applied ratio={ratio:.4f} "
          f"({int(agg_original.sum())} -> {int(agg_original.sum() - agg_removed.sum())} "
          f"over {len(universes)} universes)")

    for b in range(args.N_BINS):
        per_bin_rows.append({
            "model_name": label, "tde_model": tde_model,
            "z_bin_lo": bin_edges[b], "z_bin_hi": bin_edges[b + 1],
            "n_original": int(agg_original[b]), "n_removed": int(agg_removed[b]),
        })

bin_csv = os.path.join(out_dir, f"per_bin_removed_LSST_{args.merger_model}.csv")
with open(bin_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(per_bin_rows[0].keys()))
    writer.writeheader()
    writer.writerows(per_bin_rows)
print(f"Saved: {bin_csv}")
