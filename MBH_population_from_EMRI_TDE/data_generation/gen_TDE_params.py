#!/usr/bin/env python

import numpy as np
import argparse
import os, sys
import h5py

sys.path.insert(0, os.path.abspath('./astrophysical_setup'))
from cosmology import CosmologyModel

cosmo_model = CosmologyModel()

# Argument parsing
parser = argparse.ArgumentParser(description="Generate data for training.")
parser.add_argument("--GALAXIES", type=int, required=True, help="Number of galaxies")
parser.add_argument("--eta", type=float, nargs=2, required=True, help="eta")
parser.add_argument("--alpha", type=float, nargs=2, required=True, help="alpha")
parser.add_argument("--beta", type=float, nargs=2, required=True, help="beta")
parser.add_argument("--is_clustering", type=int, required=True, help="Whether to use clustering")

args = parser.parse_args()

hf = h5py.File(f'./data/astrophysical_data/{args.GALAXIES}/data_cusp_evolution_{ "clustering" if args.is_clustering else "no_clustering" }.h5', 'r')

z_gal = np.array(hf['z_gal'][:])
ra_rad = np.rad2deg(np.array(hf['ra_deg'][:]))
dec_rad = np.rad2deg(np.array(hf['dec_deg'][:]))
lgMBH_mass = np.array(hf['lgMBH'][:])
initial_MBHspin = np.array(hf['initial_MBHspin'][:])
star_masses = np.array(hf['star_masses'])
generated_TDEs = np.array(hf['generated_TDEs'][:])

hf.close()

eta_min, eta_max = args.eta
alpha_min, alpha_max = args.alpha
beta_min, beta_max = args.beta

param_mins = np.array([eta_min, alpha_min, beta_min])
param_maxs = np.array([eta_max, alpha_max, beta_max])

def loguniform(low=0, high=1, size=None):
    return np.exp(np.random.uniform(low, high, size))


with h5py.File(f"./data/astrophysical_data/{args.GALAXIES}/all_galaxies_TDE_events_{ 'clustering' if args.is_clustering else 'no_clustering' }.h5", "w") as f:

    idx = np.where(generated_TDEs > 0)[0]

    for i in idx:
        N = generated_TDEs[i]
        print(f"Galaxy {i}: N_TDE_events = {N}")

        # Create a group for this galaxy
        group = f.create_group(f"galaxy_{i}")

        # Fixed galaxy properties
        group["lgMBH_mass"] = lgMBH_mass[i]
        group["z_gal"] = z_gal[i]
        group["initial_MBHspin"] = initial_MBHspin[i]
        group["ra_rad"] = ra_rad[i]
        group["dec_rad"] = dec_rad[i]
        group["star_mass"] = star_masses
        group["eta"] = loguniform(np.log(eta_min), np.log(eta_max), size=N)
        group["alpha"] = loguniform(np.log(alpha_min), np.log(alpha_max), size=N)
        group["beta"] = np.random.uniform(beta_min, beta_max, size=N)