#!/usr/bin/env python

import numpy as np
import argparse
import os, sys
import h5py

# Argument parsing
parser = argparse.ArgumentParser(description="Generate data for training.")
parser.add_argument("--lgMBH", type=float, nargs=2, required=True, help="lgMBH range: min max")
parser.add_argument("--eta", type=float, nargs=2, required=True, help="eta range: min max")
parser.add_argument("--alpha", type=float, nargs=2, required=True, help="alpha range: min max")
parser.add_argument("--beta", type=float, nargs=2, required=True, help="beta range: min max")
parser.add_argument("--events", type=int, required=True, help="total events")
parser.add_argument("--z_gal", type=float, nargs=2, required=True, help="redshift range: min max")
parser.add_argument("--ra_deg", type=float, nargs=2, required=True, help="RA range: min max")
parser.add_argument("--dec_deg", type=float, nargs=2, required=True, help="DEC range: min max")
parser.add_argument("--MBHspin", type=float, nargs=2, required=True, help="MBH spin range: min max")
parser.add_argument("--SURVEY", type=str, default="ztf", help="Survey name (e.g., 'ztf' or 'lsst')")
args = parser.parse_args()

output_dir = f'./data/PRE_TRAIN_DATA_TDE/1E{np.log10(args.events):.0f}_events/'

os.makedirs(output_dir, exist_ok=True)

lgMBH_min, lgMBH_max = args.lgMBH
z_gal_min, z_gal_max = args.z_gal
ra_min, ra_max = args.ra_deg
ra_rad_min, ra_rad_max = np.deg2rad(ra_min), np.deg2rad(ra_max)
dec_min, dec_max = args.dec_deg
dec_rad_min, dec_rad_max = np.deg2rad(dec_min), np.deg2rad(dec_max)
MBHspin_min, MBHspin_max = args.MBHspin
star_masses = 1.0
eta_min, eta_max = args.eta
alpha_min, alpha_max = args.alpha
beta_min, beta_max = args.beta

def loguniform(low=0, high=1, size=None):
    return np.exp(np.random.uniform(low, high, size))


data = [
    np.random.uniform(lgMBH_min, lgMBH_max, size=args.events),
    np.random.uniform(MBHspin_min, MBHspin_max, size=args.events),
    np.random.uniform(z_gal_min, z_gal_max, size=args.events),
    np.random.uniform(ra_rad_min, ra_rad_max, size=args.events),
    np.random.uniform(dec_rad_min, dec_rad_max, size=args.events),
    loguniform(np.log(eta_min), np.log(eta_max), size=args.events),
    loguniform(np.log(alpha_min), np.log(alpha_max), size=args.events),
    np.random.uniform(beta_min, beta_max, size=args.events)
]

filename = f"TDE_params_1E{int(np.log10(args.events))}_{args.SURVEY}.h5"
file_path = os.path.join(output_dir, filename)

with h5py.File(file_path, "w") as f:
    f.create_dataset("lgMBH", data=data[0])
    f.create_dataset("MBHspin", data=data[1])
    f.create_dataset("z_gal", data=data[2])
    f.create_dataset("ra_rad", data=data[3])
    f.create_dataset("dec_rad", data=data[4])
    f.create_dataset("eta", data=data[5])
    f.create_dataset("alpha", data=data[6])
    f.create_dataset("beta", data=data[7])

f.close()