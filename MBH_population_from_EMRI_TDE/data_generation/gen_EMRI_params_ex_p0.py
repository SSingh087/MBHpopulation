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

parser.add_argument("--a", type=float, nargs=2, required=True, help="a")
parser.add_argument("--e0", type=float, nargs=2, required=True, help="e0")
parser.add_argument("--Y0", type=float, nargs=2, required=True, help="Y0")
parser.add_argument("--T_SIGNAL", type=float, nargs=2, required=True, help="time in years")

parser.add_argument("--file_name", type=str, required=True, help="Output file name.")

args = parser.parse_args()

hf = h5py.File('data_cusp_evolution.h5', 'r')
lgMgal_samples = hf['lgMgal'][:]
z_gal = hf['z_gal'][:]
lgMBH_mass = hf['lgMBH'][:]
cusp_age = hf['cusp_age'][:]
accumulated_EMRIs = hf['accumulated_EMRIs'][:] * 1E12

hf.close()

distances = cosmo_model.luminosity_distance(z_gal).to('Gpc').value

a_min, a_max = args.a
e0_min, e0_max = args.e0
Y0_min, Y0_max = args.Y0
T_SIGNAL_min, T_SIGNAL_max = args.T_SIGNAL


qS_min, qS_max = 0.1, np.pi * 0.99
phiS_min, phiS_max = 0.1, 2 * np.pi * 0.99   
qK_min, qK_max = 0.1, np.pi * 0.99
phiK_min, phiK_max = 0.1, 2 * np.pi * 0.99
Phi_phi0_min, Phi_phi0_max = 0.1, 2 * np.pi * 0.99
Phi_theta0_min, Phi_theta0_max = 0.1, 2 * np.pi * 0.99
Phi_r0_min, Phi_r0_max = 0.1, 2 * np.pi * 0.99

param_mins = np.array([
    a_min, e0_min, Y0_min,
    qS_min, phiS_min, qK_min, phiK_min,
    Phi_phi0_min, Phi_theta0_min, Phi_r0_min,
    T_SIGNAL_min
])

param_maxs = np.array([
    a_max, e0_max, Y0_max,
    qS_max, phiS_max, qK_max, phiK_max,
    Phi_phi0_max, Phi_theta0_max, Phi_r0_max,
    T_SIGNAL_max
])

param_names = [
    "a", "e0", "Y0",
    "qS", "phiS",
    "qK", "phiK",
    "Phi_phi0", "Phi_theta0", "Phi_r0",
    "T_SIGNAL_years"
]

with h5py.File("all_galaxies_EMRI_events.h5", "w") as f:

    for g in range(len(accumulated_EMRIs)):
        N = int(accumulated_EMRIs[g])
        print(f"Galaxy {g}: N_EMRI_events = {N}")

        # Create a group for this galaxy
        group = f.create_group(f"galaxy_{g}")

        # Fixed galaxy properties
        group["lgMBH_mass"] = lgMBH_mass[g]
        group["distance_Gpc"] = distances[g]

        params = np.random.uniform(param_mins, param_maxs, size=(N, len(param_mins)))

        for i, name in enumerate(param_names):
            group[name] = params[:, i]