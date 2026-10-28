#!/usr/bin/env python
"""
Step 3 of the realistic-sweep pipeline.

EMRI catalogs have no TDE-lightcurve-model axis (an EMRI's detectability
doesn't depend on which redback TDE model was used for that universe's TDE
side) -- see sweep_thin_TDE_ZTF.py's ratio_summary_ZTF.csv, which has one
row per (scenario, tde_model). For EMRI, each scenario M1-M8 gets ONE ratio:
the mean of that scenario's 4 per-TDE-model ZTF reduction ratios. That
scenario-level ratio is applied, per redshift bin, to the scenario's EMRI
catalogs across all universes -- i.e. the SAME raw ratio used to derive
N_TDE_LSST, unmodified (matches "use the same ratio to divide EMRI").

NOTE: an earlier version of this script normalized the 8 ratios by their
max before applying them, aiming to fix what looked like a collapsed
between-scenario ordering. That was wrong: normalizing is a pure linear
rescale (dividing every ratio by the same constant), so it cannot change
relative ordering at all -- what it DID do is inflate N_EMRI ~15x and
distort the EMRI:TDE_ZTF proportion from ~0.02-0.04 (close to the true
unthinned ratio, ~0.039 for M1) to ~0.35-0.6, ~10-15x too generous to EMRI.
Reverted to the raw (unnormalized) ratio -- it keeps the EMRI:TDE
proportion realistic; the between-scenario ordering was always faintly
present in the raw ratio's continuous product, just compressed near 1 by
integer rounding (see the *_mean columns in ratio_summary_EMRI.csv for the
unrounded per-universe average, which shows it more clearly than the
floored integer counts do).

A floor of 2 EMRIs per universe is enforced after thinning: if a universe's
thinned catalog has exactly 1 event, that single row is duplicated so
downstream code (which needs at least 2 events for spread statistics)
doesn't break. This is a data-quality patch, not a physical effect.

Outputs (under ASTROPHYSICAL_DATA_DIR/{GALAXIES}/realistic/):
  EMRI_SNRs_data_realistic_{scenario}_{u}.h5   -- thinned catalogs (floor-of-2 applied)
  ratio_summary_EMRI.csv                       -- one row per scenario
  per_bin_removed_EMRI.csv                     -- one row per (scenario, z bin)
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.abspath("./"))
import numpy as np
import pandas as pd

from common_realistic import (
    MERGER_MODEL_CHOICES, set_merger_model,
    ASTROPHYSICAL_DATA_DIR, GALAXIES_DEFAULT, N_REDSHIFT_BINS_DEFAULT,
    SCENARIOS, SCENARIO_NAME_MAP,
    discover_universes, emri_snr_path, load_h5_catalog, redshift_bin_edges,
    rng_for, save_thinned_h5, scenario_tag, thin_by_redshift_bin,
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
ratios_df = pd.read_csv(ratio_csv)

scenario_ratio_rows = []
per_bin_rows = []

# ── Pass 1: raw per-scenario ratio (mean of the 4 ZTF TDE-model ratios) ──
raw_ratios = {}
for scenario in SCENARIOS:
    label = SCENARIO_NAME_MAP[scenario]
    sub = ratios_df[ratios_df["model_name"] == label]
    if sub.empty:
        print(f"[SKIP] {label}: no ZTF ratio rows found")
        continue
    raw_ratios[scenario] = float(sub["reduction_ratio"].mean())

print(f"Raw scenario ratios: { {SCENARIO_NAME_MAP[s]: round(r, 4) for s, r in raw_ratios.items()} }")

# ── Pass 2: thin using the raw ratio, unmodified ──
for scenario in SCENARIOS:
    tag = scenario_tag(scenario)
    label = SCENARIO_NAME_MAP[scenario]

    if scenario not in raw_ratios:
        continue
    raw_ratio = raw_ratios[scenario]
    scenario_ratio = raw_ratio

    glob_pat = emri_snr_path(ASTROPHYSICAL_DATA_DIR, GALAXIES, scenario, "*")
    universes = discover_universes(glob_pat)
    if not universes:
        print(f"[SKIP] {label} ({tag}): no EMRI universes found")
        continue

    per_universe = {}
    for u in universes:
        path = emri_snr_path(ASTROPHYSICAL_DATA_DIR, GALAXIES, scenario, u)
        per_universe[u] = load_h5_catalog(path)

    pooled_z = np.concatenate([d["z_gal"] for d, _ in per_universe.values()])
    bin_edges = redshift_bin_edges(pooled_z, n_bins=args.N_BINS)

    agg_removed = np.zeros(args.N_BINS, dtype=int)
    agg_original = np.zeros(args.N_BINS, dtype=int)

    n_floored = 0
    for u in universes:
        data, attrs = per_universe[u]
        rng = rng_for("EMRI", tag, u)
        keep_mask, per_bin_removed, per_bin_original = thin_by_redshift_bin(
            data["z_gal"], scenario_ratio, bin_edges, rng)

        out_path = emri_snr_path(ASTROPHYSICAL_DATA_DIR, GALAXIES, scenario, u, realistic=True)
        extra_attrs = {
            "n_events_passed_original": len(data["z_gal"]),
            "reduction_ratio": scenario_ratio,
            "reduction_ratio_source": "mean of 4 ZTF-derived TDE-model ratios for this scenario, "
                                       "unmodified (same ratio also used for N_TDE_LSST)",
        }

        if keep_mask.sum() == 1:
            # Floor of 2: duplicate the single surviving event so downstream
            # code that needs >=2 events per universe doesn't break. Data
            # quality patch, not a physical effect -- flagged via attrs.
            idx = np.where(keep_mask)[0]
            data = {k: np.concatenate([v[keep_mask], v[idx]]) for k, v in data.items()}
            keep_mask = np.ones(2, dtype=bool)
            extra_attrs["floor_of_2_applied"] = True
            n_floored += 1
            save_thinned_h5(out_path, data, attrs, keep_mask, extra_attrs=extra_attrs)
        else:
            extra_attrs["floor_of_2_applied"] = False
            save_thinned_h5(out_path, data, attrs, keep_mask, extra_attrs=extra_attrs)

        agg_removed += per_bin_removed
        agg_original += per_bin_original

    if n_floored:
        print(f"  [FLOOR] {label}: {n_floored}/{len(universes)} universes had exactly 1 EMRI "
              f"-- duplicated to 2")

    n_orig_total = int(agg_original.sum())
    n_kept_total = n_orig_total - int(agg_removed.sum())
    mean_kept_before_floor = n_kept_total / len(universes)
    mean_kept_after_floor = (n_kept_total + n_floored) / len(universes)  # each floor adds +1 event
    print(f"{label} ({tag}): ratio={scenario_ratio:.4f}  "
          f"({n_orig_total} -> {n_kept_total} over {len(universes)} universes; "
          f"mean/universe: {mean_kept_before_floor:.2f} before floor, "
          f"{mean_kept_after_floor:.2f} after floor)")

    scenario_ratio_rows.append({
        "model_name": label, "mbh_mass_method": scenario[0],
        "nucleation_method": scenario[1], "occupation_method": scenario[2],
        "n_universes": len(universes), "reduction_ratio": scenario_ratio,
        "total_original": n_orig_total, "total_kept_before_floor": n_kept_total,
        "n_floored": n_floored,
        "mean_per_universe_before_floor": mean_kept_before_floor,
        "mean_per_universe_after_floor": mean_kept_after_floor,
    })

    for b in range(args.N_BINS):
        per_bin_rows.append({
            "model_name": label,
            "z_bin_lo": bin_edges[b], "z_bin_hi": bin_edges[b + 1],
            "n_original": int(agg_original[b]), "n_removed": int(agg_removed[b]),
        })

ratio_out_csv = os.path.join(out_dir, f"ratio_summary_EMRI_{args.merger_model}.csv")
with open(ratio_out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(scenario_ratio_rows[0].keys()))
    writer.writeheader()
    writer.writerows(scenario_ratio_rows)
print(f"Saved: {ratio_out_csv}")

bin_csv = os.path.join(out_dir, f"per_bin_removed_EMRI_{args.merger_model}.csv")
with open(bin_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(per_bin_rows[0].keys()))
    writer.writeheader()
    writer.writerows(per_bin_rows)
print(f"Saved: {bin_csv}")
