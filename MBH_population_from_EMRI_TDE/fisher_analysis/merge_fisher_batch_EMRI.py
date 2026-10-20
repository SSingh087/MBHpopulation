"""Merge the EMRI Fisher batch files into the posterior file read by the hierarchical inference."""
import argparse
import glob
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.abspath('./'))
from utils import INFERENCE_DATA_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES", type=int, required=True)
parser.add_argument("--SCENARIO", type=str, required=True,
                    help="{mbh_mass_method}_{nucleation_method}_{occupation_method}")
args = parser.parse_args()

loc = f"{INFERENCE_DATA_DIR}/{args.GALAXIES}"
tag = f"EMRI_{args.SCENARIO}"
files = sorted(glob.glob(f"{loc}/fisher_batches/{tag}_*.h5"))
if not files:
    raise RuntimeError(f"No batch files for {tag} in {loc}/fisher_batches -- did the Fisher DAG run?")

keys = ["lgMBH_mass", "MBHspin", "CO_mass", "z_gal", "e_pl"]
merged = {k: [] for k in keys}
for fn in files:
    with h5py.File(fn, "r") as hf:
        if "z_gal" not in hf:
            continue
        for k in keys:
            merged[k].append(hf[k][()])
merged = {k: np.concatenate(v).astype(np.float32) for k, v in merged.items()}

out = f"{loc}/fisher_posteriors_{tag}"
with h5py.File(out + ".h5", "w") as hf:
    for k, v in merged.items():
        hf.create_dataset(k, data=v, compression="gzip")
torch.save({k: torch.from_numpy(v) for k, v in merged.items()}, out + ".pt")
print(f"Merged {len(files)} batches, {merged['z_gal'].shape[0]} events -> {out}.(h5|pt)")
