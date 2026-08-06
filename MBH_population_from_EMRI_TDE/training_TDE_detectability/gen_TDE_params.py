#!/usr/bin/env python

import numpy as np
import argparse
import os, sys
import h5py
import matplotlib.pyplot as plt

# Argument parsing
parser = argparse.ArgumentParser(description="Generate data for training.")
parser.add_argument("--lgMBH", type=float, nargs=2, required=True, help="lgMBH range: min max")
parser.add_argument("--eta", type=float, nargs=2, required=True, help="eta range: min max")
parser.add_argument("--alpha", type=float, nargs=2, required=True, help="alpha range: min max")
parser.add_argument("--beta", type=float, nargs=2, required=True, help="beta range: min max")
parser.add_argument("--events", type=int, required=True, help="total events")
parser.add_argument("--z_gal", type=float, nargs=2, required=True, help="redshift range: min max")
parser.add_argument("--SURVEY", type=str, default="ztf", help="Survey name (e.g., 'ztf' or 'lsst')")
args = parser.parse_args()

output_dir = f'./data/PRE_TRAIN_DATA_TDE/1E{np.log10(args.events):.0f}_events/'

os.makedirs(output_dir, exist_ok=True)

lgMBH_min, lgMBH_max = args.lgMBH
z_gal_min, z_gal_max = args.z_gal
star_masses = 1.0
eta_min, eta_max = args.eta
alpha_min, alpha_max = args.alpha
beta_min, beta_max = args.beta

ra_rad_min, ra_rad_max = 0.0, 2 * np.pi
dec_rad_min, dec_rad_max = -np.pi / 2, np.pi / 2

if args.SURVEY.lower() == "ztf":
    t0_mjd_transient = np.random.uniform(58200, 61396, size=args.events) # ZTF survey period Mar 2018 – Dec 2026 8.75 yr
elif args.SURVEY.lower() == "lsst":
    t0_mjd_transient = np.random.uniform(60827, 64423, size=args.events) #LSST survey period Jun 2025-Jun 2035 10 yr #60827-64423

def loguniform(low=0, high=1, size=None):
    return np.exp(np.random.uniform(low, high, size))


print(f"Generating {args.events} events with parameters:")
print(f"lgMBH: {lgMBH_min} to {lgMBH_max}")
print(f"z_gal: {z_gal_min} to {z_gal_max}")
print(f"RA: {np.rad2deg(ra_rad_min)} to {np.rad2deg(ra_rad_max)} degrees / {ra_rad_min} to {ra_rad_max} radians")
print(f"DEC: {np.rad2deg(dec_rad_min)} to {np.rad2deg(dec_rad_max)} degrees / {dec_rad_min} to {dec_rad_max} radians")
print(f"eta: {eta_min} to {eta_max}")
print(f"alpha: {alpha_min} to {alpha_max}")
print(f"beta: {beta_min} to {beta_max}")


data = [
    np.random.uniform(lgMBH_min, lgMBH_max, size=args.events),
    np.random.uniform(z_gal_min, z_gal_max, size=args.events),
    np.random.uniform(ra_rad_min, ra_rad_max, size=args.events),
    np.random.uniform(dec_rad_min, dec_rad_max, size=args.events),
    loguniform(np.log(eta_min), np.log(eta_max), size=args.events),
    loguniform(np.log(alpha_min), np.log(alpha_max), size=args.events),
    np.random.uniform(beta_min, beta_max, size=args.events),
    t0_mjd_transient
]

filename = f"TDE_params_1E{int(np.log10(args.events))}_{args.SURVEY}.h5"
file_path = os.path.join(output_dir, filename)

with h5py.File(file_path, "w") as f:
    f.create_dataset("lgMBH", data=data[0])
    f.create_dataset("z_gal", data=data[1])
    f.create_dataset("ra_rad", data=data[2])
    f.create_dataset("dec_rad", data=data[3])
    f.create_dataset("eta", data=data[4])
    f.create_dataset("alpha", data=data[5])
    f.create_dataset("beta", data=data[6])
    f.create_dataset("t0_mjd_transient", data=data[7])
f.close()

# plot generated data in one figure with 7 subplots
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
axs[0, 0].hist(data[0], bins='auto', color='blue', alpha=0.7)
axs[0, 0].set_title('lgMBH')
axs[0, 1].hist(data[1], bins='auto', color='orange', alpha=0.7)
axs[0, 1].set_title('z_gal')
axs[0, 2].hist(data[2], bins='auto', color='green', alpha=0.7)
axs[0, 2].set_title('ra_rad')
axs[1, 0].hist(data[3], bins='auto', color='red', alpha=0.7)
axs[1, 0].set_title('dec_rad')
axs[1, 1].hist(data[4], bins='auto', color='purple', alpha=0.7)
axs[1, 1].set_title('eta')
axs[1, 2].hist(data[5], bins='auto', color='brown', alpha=0.7)
axs[1, 2].set_title('alpha')
axs[2, 0].hist(data[6], bins='auto', color='pink', alpha=0.7)
axs[2, 0].set_title('beta')
axs[2, 1].hist(data[7], bins='auto', color='gray', alpha=0.7)
axs[2, 1].set_title('t0_mjd_transient')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"TDE_params_1E{int(np.log10(args.events))}_{args.SURVEY}.png"))
plt.close()
