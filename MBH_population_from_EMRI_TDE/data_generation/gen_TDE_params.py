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

z_gal = np.array(hf['z_gal'][:])
ra = np.rad2deg(np.array(hf['ra_deg'][:]))
dec = np.rad2deg(np.array(hf['dec_deg'][:]))

lgMBH_mass = np.array(hf['lgMBH'][:])
MBHspin = np.array(hf['MBHspin'][:])

star_masses = np.array(hf['star_masses'])

observed_TDEs = np.array(hf['observed_TDEs'][:])

hf.close()

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
        group["z_gal"] = z_gal[i]
        group["MBHspin"] = MBHspin[i]
        group["ra"] = ra[i]
        group["dec"] = dec[i]
        group["star_mass"] = star_masses
        params = np.random.uniform(param_mins, param_maxs, size=(N, len(param_mins)))

        for i, name in enumerate(param_names):
            group[name] = params[:, i]