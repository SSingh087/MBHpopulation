#!/usr/bin/env python
"""
Produces the per-scenario event COUNTS (not events) that this pipeline's
own make_events script should draw, replacing
N_EMRI = len(true_detected_data_EMRI_<scenario>.h5) etc.

Counts are computed directly from the realistic (redshift-bin-thinned)
catalogs written by data_gen_realistic/sweep_thin_*.py -- the mean event
count across that scenario's 30 realistic universes, rounded to the
nearest integer:

    N_EMRI_realistic     = round(mean over universes of len(EMRI_SNRs_data_realistic_<scenario>_<u>.h5))
    N_TDE_ZTF_realistic  = round(mean over universes of len(TDE_SNRs_realistic_ZTF_<scenario>_tde_fallback_<u>.h5))
    N_TDE_LSST_realistic = round(mean over universes of len(TDE_SNRs_realistic_LSST_<scenario>_tde_fallback_<u>.h5))

TDE_MODEL_FOR_COUNTS = "tde_fallback"

import argparse
_p = argparse.ArgumentParser()
_p.add_argument("--merger_model", type=str, required=True, choices=["mundy17", "oleary21", "duan24", "baseline"],
                help="Galaxy merger-rate model whose realistic catalogs are counted.")
MERGER_MODEL = _p.parse_args().merger_model per user's choice -- matches
prepare_data_MEDIAN.py's --tde_model selection, whose default the
original true_detected_data_TDE_* counts were built from.

Output: ASTROPHYSICAL_DATA_DIR/{GALAXIES}/realistic/realistic_event_counts.csv
(alongside ratio_summary_*.csv / per_bin_removed_*.csv from the sweep
scripts), one row per scenario (M1-M8), columns N_EMRI, N_TDE_ZTF,
N_TDE_LSST plus the raw per-universe mean/std for context. This is the
ONLY thing your own make_events script needs -- no event catalogs are
written here, and no separate hierarchical_inference_* pipeline copy is
needed just to carry this number -- point the existing
hierarchical_inference_altered pipeline's make_events step at this CSV
instead of true_detected_data_*.h5.
"""
import csv
import glob
import os
import re
import sys

import h5py
import numpy as np

ASTROPHYSICAL_DATA_DIR = "./data/astrophysical_data"
GALAXIES = 100000
REALISTIC_DIR = os.path.join(ASTROPHYSICAL_DATA_DIR, str(GALAXIES), "realistic")

TDE_MODEL_FOR_COUNTS = "tde_fallback"

SCENARIOS = [
    ("use_lgMgal", "hannah",   "uniform"),
    ("use_sigma",  "hannah",   "uniform"),
    ("use_lgMgal", "neumayer", "uniform"),
    ("use_sigma",  "neumayer", "uniform"),
    ("use_lgMgal", "hannah",   "varying"),
    ("use_sigma",  "hannah",   "varying"),
    ("use_lgMgal", "neumayer", "varying"),
    ("use_sigma",  "neumayer", "varying"),
]
SCENARIO_NAME_MAP = {s: f"M{i+1}" for i, s in enumerate(SCENARIOS)}


def scenario_tag(scenario):
    """Short tag identifying a scenario."""
    # (mbh, nuc, occ) + merger model, matching data_generation/*.py file names
    return "_".join(tuple(scenario) + (MERGER_MODEL,))


def discover_universes(glob_pattern):
    """List the universe indices matching a file glob."""
    idx = []
    for p in glob.glob(glob_pattern):
        m = re.search(r"_(\d+)\.h5$", os.path.basename(p))
        if m:
            idx.append(int(m.group(1)))
    return sorted(idx)


def counts_for(glob_pattern):
    """Detected-event counts per universe for the files matching the glob."""
    universes = discover_universes(glob_pattern)
    counts = []
    for u in universes:
        path = glob_pattern.replace("*", str(u))
        with h5py.File(path, "r") as f:
            counts.append(len(f["z_gal"]))
    return np.array(counts)


rows = []
for scenario in SCENARIOS:
    tag = scenario_tag(scenario)
    label = SCENARIO_NAME_MAP[scenario]

    emri_glob = os.path.join(REALISTIC_DIR, f"EMRI_SNRs_data_realistic_{tag}_*.h5")
    ztf_glob = os.path.join(REALISTIC_DIR, f"TDE_SNRs_ZTF_{tag}_{TDE_MODEL_FOR_COUNTS}_*.h5")
    lsst_glob = os.path.join(REALISTIC_DIR, f"TDE_SNRs_LSST_{tag}_{TDE_MODEL_FOR_COUNTS}_*.h5")

    n_emri = counts_for(emri_glob)
    n_ztf = counts_for(ztf_glob)
    n_lsst = counts_for(lsst_glob)

    if len(n_emri) == 0 or len(n_ztf) == 0 or len(n_lsst) == 0:
        print(f"[SKIP] {label} ({tag}): missing realistic catalogs "
              f"(EMRI={len(n_emri)}, ZTF={len(n_ztf)}, LSST={len(n_lsst)} universes) "
              f"-- run data_gen_realistic/sweep_thin_*.py first.")
        continue

    row = {
        "model_name": label,
        "mbh_mass_method": scenario[0],
        "nucleation_method": scenario[1],
        "occupation_method": scenario[2],
        "tde_model_used_for_counts": TDE_MODEL_FOR_COUNTS,
        "N_EMRI": int(round(n_emri.mean())),
        "N_TDE_ZTF": int(round(n_ztf.mean())),
        "N_TDE_LSST": int(round(n_lsst.mean())),
        "N_EMRI_mean": n_emri.mean(), "N_EMRI_std": n_emri.std(),
        "N_TDE_ZTF_mean": n_ztf.mean(), "N_TDE_ZTF_std": n_ztf.std(),
        "N_TDE_LSST_mean": n_lsst.mean(), "N_TDE_LSST_std": n_lsst.std(),
        "n_universes": len(n_emri),
    }
    rows.append(row)
    print(f"{label} ({tag}): N_EMRI={row['N_EMRI']}  N_TDE_ZTF={row['N_TDE_ZTF']}  "
          f"N_TDE_LSST={row['N_TDE_LSST']}  (mean over {row['n_universes']} universes, "
          f"TDE model={TDE_MODEL_FOR_COUNTS})")

out_path = os.path.join(REALISTIC_DIR, f"realistic_event_counts_{MERGER_MODEL}.csv")
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"\nSaved: {out_path}")
