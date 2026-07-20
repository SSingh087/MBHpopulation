#!/usr/bin/env python
"""
Runs once, after every batch node in the DAG has completed. Reads all
fisher_batch_*.h5 files and merges them into the same three output
files the old single-process script produced:
    fisher_results_TDE_{SURVEY}_{suffix}.h5
    true_data_TDE_{SURVEY}_{suffix}.h5
    noisy_data_TDE_{SURVEY}_{suffix}.h5
"""

import glob, argparse, sys, os
sys.path.insert(0, os.path.abspath('./'))
from utils import FISHER_DATA_DIR
import numpy as np
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES",      type=int, required=True)
parser.add_argument("--SURVEY",        type=str, default="ztf")
args = parser.parse_args()

loc      = f"{FISHER_DATA_DIR}/{args.GALAXIES}"

batch_files = sorted(glob.glob(f"{loc}/batches/fisher_batch_*.h5"))
if not batch_files:
    raise RuntimeError(f"No batch files found in {loc}/batches -- did the DAG run?")

print(f"Merging {len(batch_files)} batch files...")

PARAMS = ["redshift", "mbh_6", "stellar_mass", "eta", "alpha", "beta"]

fisher_out = h5py.File(f"{loc}/fisher_results_TDE_{args.SURVEY.upper()}.h5", "w")
true_samples_by_param = {p: [] for p in PARAMS}
noisy_samples_by_param = {p: [] for p in PARAMS}

n_events_total = 0
n_events_ok = 0

for bf in batch_files:
    with h5py.File(bf, "r") as hf:
        n_events_total += hf.attrs.get("n_events_in_batch", 0)
        for event_key in hf.keys():
            n_events_ok += 1
            eg = hf[event_key]
            out_g = fisher_out.create_group(event_key)
            for name in ["fisher_matrix", "covariance_matrix", "errors", "truths",
                         "fisher_matrix_new", "covariance_matrix_new", "errors_new", "truths_new"]:
                out_g.create_dataset(name, data=eg[name][()])

            true_samples_by_param_local = eg["true_samples"][()]   # (N_SAMPLES, n_params)
            noisy_samples_by_param_local = eg["noisy_samples"][()]
            for i, p in enumerate(PARAMS):
                true_samples_by_param[p].append(true_samples_by_param_local[:, i])
                noisy_samples_by_param[p].append(noisy_samples_by_param_local[:, i])

fisher_out.attrs["n_events_total"] = n_events_total
fisher_out.attrs["n_events_ok"]    = n_events_ok
fisher_out.close()

with h5py.File(f"{loc}/true_data_TDE_{args.SURVEY.upper()}.h5", "w") as hf_true, \
     h5py.File(f"{loc}/noisy_data_TDE_{args.SURVEY.upper()}.h5", "w") as hf_noisy:
    for p in PARAMS:
        true_arr = np.vstack(true_samples_by_param[p])
        noisy_arr = np.vstack(noisy_samples_by_param[p])
        hf_true.create_dataset(p, data=true_arr, compression="gzip")
        hf_noisy.create_dataset(p, data=noisy_arr, compression="gzip")
        print(f"[SAVED] '{p}' shape {true_arr.shape} (true) / {noisy_arr.shape} (noisy)")

print(f"\n[DONE] Merged {n_events_ok}/{n_events_total} events across {len(batch_files)} batches.")
print(f"Outputs written to {loc}/")