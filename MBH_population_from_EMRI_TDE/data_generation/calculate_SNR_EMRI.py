#!/usr/bin/env python
"""EMRI SNR from the trained SNR network (training_EMRI_SNR/); keeps events above --SNR_THRESHOLD.

The network predicts y = minmax(log10(SNR / D_L[Gpc])) from the injection parameters
without distance and phases; SNR is recovered with the (y_min, y_max) saved in
snr_scaling.json by training_EMRI_SNR/prep_SNR_training_data.py.
"""

import argparse
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.abspath('./'))
sys.path.insert(0, os.path.abspath('./astrophysical_setup'))
sys.path.insert(0, os.path.abspath('./data_generation'))
sys.path.insert(0, os.path.abspath('../poplar'))
from utils import ASTROPHYSICAL_DATA_DIR, EMRI_DATA_FILE_NAME, MERGER_MODEL_CHOICES
from emri_snr_model import load_snr_model, predict_snr

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, y_min, y_max = load_snr_model(args.MODEL_PATH, args.SCALING_PATH, device)
snr = predict_snr(model, y_min, y_max, data, device, args.batch_size)
print(f"{np.isfinite(snr).sum()}/{n_events} events have finite inputs; the rest get SNR = NaN.")

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
