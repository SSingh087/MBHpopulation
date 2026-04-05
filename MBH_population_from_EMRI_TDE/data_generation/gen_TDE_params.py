#!/usr/bin/env python

import numpy as np
import argparse
import os, sys
import h5py

sys.path.insert(0, os.path.abspath('../astrophysical_setup'))
from cosmology import CosmologyModel

cosmo_model = CosmologyModel()

# Argument parsing
parser = argparse.ArgumentParser(description="Generate data for training.")

parser.add_argument("--eta", type=float, nargs=2, required=True, help="eta")
parser.add_argument("--alpha", type=float, nargs=2, required=True, help="alpha")
parser.add_argument("--beta", type=float, nargs=2, required=True, help="beta")
parser.add_argument("--OBSERVING_WINDOW", type=float, required=True, help="Observing window in days.")

parser.add_argument("--file_name", type=str, required=True, help="Output file name.")

args = parser.parse_args()

hf = h5py.File('./DATA/data_cusp_evolution.h5', 'r')
lgMgal_samples = hf['lgMgal'][:]
z_gal = hf['z_gal'][:]
lgMBH_mass = hf['lgMBH'][:]
MBHspin = hf['MBHspin'][:]
observed_TDEs = hf['observed_TDEs'][:]

hf.close()

distances = cosmo_model.luminosity_distance(z_gal).to('Gpc').value

eta_min, eta_max = args.eta
alpha_min, alpha_max = args.alpha
beta_min, beta_max = args.beta

param_mins = np.array([eta_min, alpha_min, beta_min])
param_maxs = np.array([eta_max, alpha_max, beta_max])
param_names = ["eta", "alpha", "beta"]

with h5py.File("./DATA/all_galaxies_TDE_events.h5", "w") as f:

    idx = np.where(observed_TDEs > 0)[0]

    for i in idx:
        N = observed_TDEs[i]
        print(f"Galaxy {i}: N_TDE_events = {N}")

        # Create a group for this galaxy
        group = f.create_group(f"galaxy_{i}")

        # Fixed galaxy properties
        group["lgMBH_mass"] = lgMBH_mass[i]
        group["distance_Gpc"] = distances[i]
        group["MBHspin"] = MBHspin[i]
        params = np.random.uniform(param_mins, param_maxs, size=(N, len(param_mins)))

        for i, name in enumerate(param_names):
            group[name] = params[:, i]