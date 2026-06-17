#!/usr/bin/env python

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import h5py

# -----------------------------
# Argument parser
# -----------------------------
parser = argparse.ArgumentParser(description="Generate data for training.")
parser.add_argument("--events", type=int, required=True, help="total events")
args = parser.parse_args()

# -----------------------------
# Scaling functions
# -----------------------------
def rescale_between(x, target_min, target_max):
    """Rescale array x to [target_min, target_max]"""
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max == x_min:
        return np.full_like(x, target_min)  # avoid division by zero
    return (x - x_min) / (x_max - x_min) * (target_max - target_min) + target_min

def unrescale_between(x_scaled, original_min, original_max, target_min, target_max):
    """Inverse of rescale_between"""
    return (x_scaled - target_min) / (target_max - target_min) * (original_max - original_min) + original_min

# -----------------------------
# File paths
# -----------------------------
events_for_dir = f"{args.events:.0E}".replace("+0", "").replace(".0", "")
output_dir = f'./data/PRE_TRAIN_DATA_TDE/{events_for_dir}_events/'

# HDF5 files
params_ZTF_file = f'{output_dir}/TDE_params_1E{int(np.log10(args.events))}_ZTF.h5'
# params_LSST_file = f'{output_dir}/TDE_params_1E{int(np.log10(args.events))}_LSST.h5'

# -----------------------------
# Scaling rules
# -----------------------------
scaling_rules = {
    # linear scaling to [0,1]
    "lgMBH": (lambda x: x, 0.0, 1.0),
    "MBHspin": (lambda x: x, 0.0, 1.0),
    "z_gal": (lambda x: x, 0.0, 1.0),
    "ra_rad": (lambda x: x, 0.0, 1.0),
    "dec_rad": (lambda x: x, 0.0, 1.0),
    "beta": (lambda x: x, 0.0, 1.0),

    # log scaling then rescale to [-1,1]
    "eta": (np.log10, -1.0, 1.0),
    "alpha": (np.log10, -1.0, 1.0)
}

# -----------------------------
# Load HDF5 files
# -----------------------------
with h5py.File(params_ZTF_file, 'r') as params_TDE_ZTF: #, \
    #  h5py.File(params_LSST_file, 'r') as params_TDE_LSST:

    num_params = len(params_TDE_ZTF.keys())
    num_events = len(params_TDE_ZTF["lgMBH"])

    # Initialize arrays to hold scaled values
    param_ZTF_scaled = np.zeros((num_events, num_params))
    param_LSST_scaled = np.zeros((num_events, num_params))

    # Process each parameter
    for i, key in enumerate(params_TDE_ZTF.keys()):
        print(f"Processing {key}...")

        param_ZTF = params_TDE_ZTF[key][:]
        # param_LSST = params_TDE_LSST[key][:]

        if key in scaling_rules:
            transform_fn, target_min, target_max = scaling_rules[key]
            param_ZTF_scaled[:, i] = rescale_between(transform_fn(param_ZTF), target_min, target_max)
            # param_LSST_scaled[:, i] = rescale_between(transform_fn(param_LSST), target_min, target_max)
        else:
            # No scaling rule → keep original
            param_ZTF_scaled[:, i] = param_ZTF
            # param_LSST_scaled[:, i] = param_LSST

# -----------------------------
# Save scaled arrays as numpy
# -----------------------------
np.save(f"{output_dir}/TDE_params_scaled_1E{int(np.log10(args.events))}_ZTF.npy", param_ZTF_scaled)
# np.save(f"{output_dir}/TDE_params_scaled_1E{int(np.log10(args.events))}_LSST.npy", param_LSST_scaled)

print("All parameters rescaled and saved successfully!")