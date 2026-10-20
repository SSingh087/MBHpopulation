#!/usr/bin/env python
"""EMRI SNR from the trained SNR network (training_EMRI_SNR/); keeps events above --SNR_THRESHOLD.

The network predicts y = minmax(log10(SNR / D_L[Gpc])) from the injection parameters
without distance and phases; SNR is recovered with the (y_min, y_max) saved in
snr_scaling.json by training_EMRI_SNR/prep_SNR_training_data.py.
"""

import argparse
import json
import os
import pickle
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.abspath('./'))
sys.path.insert(0, os.path.abspath('./astrophysical_setup'))
sys.path.insert(0, os.path.abspath('../poplar'))
from utils import ASTROPHYSICAL_DATA_DIR, EMRI_DATA_FILE_NAME, MERGER_MODEL_CHOICES

# Network input order (distance and the three initial phases are not inputs); must match
# the column order of injection_params_dist_REMOVED_COMBINED.npy in training_EMRI_SNR/.
FEATURES = ["lgMBH_mass", "MBHspin", "p0", "e0", "Y0", "qS", "phiS", "qK", "phiK",
            "T_SIGNAL_duration_years"]

parser = argparse.ArgumentParser(description="EMRI SNR from the trained network.")
parser.add_argument("--UNIVERSE_MODEL", type=int, required=True)
parser.add_argument("--GALAXIES", type=int, required=True)
parser.add_argument("--mbh_mass_method", type=str, required=True, choices=["use_sigma", "use_lgMgal"])
parser.add_argument("--nucleation_method", type=str, required=True, choices=["hannah", "neumayer"])
parser.add_argument("--occupation_method", type=str, required=True, choices=["uniform", "varying"])
parser.add_argument("--merger_model", type=str, required=True, choices=MERGER_MODEL_CHOICES)
parser.add_argument("--MODEL_PATH", type=str, required=True,
                    help="model.pth of the trained SNR network")
parser.add_argument("--SCALING_PATH", type=str, required=True,
                    help="snr_scaling.json written by prep_SNR_training_data.py")
parser.add_argument("--SNR_THRESHOLD", type=float, default=20.0, help="Detection SNR threshold")
parser.add_argument("--batch_size", type=int, default=100000)
args = parser.parse_args()

tag = f"{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}_{args.merger_model}_{args.UNIVERSE_MODEL}"
in_path = f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}/{EMRI_DATA_FILE_NAME}_{tag}.h5"
out_path = f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}/EMRI_SNRs_data_{tag}.h5"

with h5py.File(in_path, "r") as f:
    data = {key: np.array(f[key][:]) for key in f.keys()}

n_events = len(data["lgMBH_mass"])
X = np.stack([data[k] for k in FEATURES], axis=1).astype(np.float32)
dist_gpc = data["distance_Gpc"]

valid = np.isfinite(X).all(axis=1) & (dist_gpc > 0)
print(f"{valid.sum()}/{n_events} events have finite inputs; the rest get SNR = NaN.")

with open(args.SCALING_PATH) as f:
    scaling = json.load(f)
y_min, y_max = scaling["y_min"], scaling["y_max"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(args.MODEL_PATH, "rb") as f:
    model = pickle.load(f)
model.to(device)
model.eval()

y = np.full(n_events, np.nan)
Xv = torch.tensor(X[valid])
pred = []
with torch.no_grad():
    for i in range(0, len(Xv), args.batch_size):
        pred.append(model(Xv[i:i + args.batch_size].to(device)).reshape(-1).cpu().numpy())
if pred:
    y[valid] = np.concatenate(pred)

snr = dist_gpc * 10 ** (y * (y_max - y_min) + y_min)
snr = np.where(valid, snr, np.nan)

events_idx = np.where(snr > args.SNR_THRESHOLD)[0]
print(f"{len(events_idx)}/{n_events} events pass SNR threshold {args.SNR_THRESHOLD}")
if len(events_idx) == 0:
    print("WARNING: no events pass the threshold -- writing an empty file.")

with h5py.File(out_path, "w") as f:
    f.attrs["n_events_total"] = n_events
    f.attrs["n_events_passed"] = len(events_idx)
    f.attrs["rho_threshold"] = args.SNR_THRESHOLD
    for key, val in data.items():
        f.create_dataset(key, data=val[events_idx])
    f.create_dataset("SNR", data=snr[events_idx])
    # SNR of every generated event, kept to train the selection function
    f.create_dataset("SNR_all", data=snr)

print(f"Saved {len(events_idx)}/{n_events} SNR-passing EMRI rows to {out_path}")
