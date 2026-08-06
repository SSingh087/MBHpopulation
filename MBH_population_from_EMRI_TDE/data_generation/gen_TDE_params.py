#!/usr/bin/env python

import numpy as np
import argparse
import os, sys
import h5py

sys.path.insert(0, os.path.abspath('./astrophysical_setup'))
sys.path.insert(0, os.path.abspath('./'))
from utils import ASTROPHYSICAL_DATA_DIR, MOCK_GALAXY_CATALOG_DATA, TDE_DATA_FILE_NAME
from cosmology import CosmologyModel

cosmo_model = CosmologyModel()

# ============================================================
# Argument parsing
# ============================================================
parser = argparse.ArgumentParser(description="Generate TDE transient parameters.")
parser.add_argument("--UNIVERSE_MODEL", type=int, required=True, help="Universe model index")
parser.add_argument("--GALAXIES", type=int, required=True, help="Monte Carlo sample size used to label the run directory")
parser.add_argument("--eta", type=float, nargs=2, required=True, help="eta range [min, max] (log-uniform)")
parser.add_argument("--alpha", type=float, nargs=2, required=True, help="alpha range [min, max] (log-uniform)")
parser.add_argument("--beta", type=float, nargs=2, required=True, help="beta range [min, max] (uniform)")
parser.add_argument("--mbh_mass_method", type=str, required=False, default="use_sigma", choices=["use_sigma", "use_lgMgal"])
parser.add_argument("--nucleation_method", type=str, required=False, default="hannah", choices=["hannah", "neumayer"])
parser.add_argument("--SEED", type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()

rng = np.random.default_rng(seed=args.SEED)

# ============================================================
# Load resampled TDE catalog (flat, one row = one event)
# ============================================================
loc = f'{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}'
catalog_path = f'{loc}/{MOCK_GALAXY_CATALOG_DATA}_{args.mbh_mass_method}_{args.nucleation_method}_{args.UNIVERSE_MODEL}.h5'

with h5py.File(catalog_path, 'r') as f:
    tde = {field: np.array(f['TDE_catalog'][field][:]) for field in f['TDE_catalog'].keys()}

n_events = len(tde['lgMBH'])
print(f"Loaded {n_events} TDE events from {catalog_path}")

# FIX: the `raise` below was not indented under the `if` (an
# IndentationError -- this is a SyntaxError that stops the file from
# even being parsed, for ANY input, regardless of n_events). Fixed.
if n_events == 0:
    raise RuntimeError(f"TDE_catalog in {catalog_path} is empty -- nothing to generate parameters for.")

z_gal = tde['z_gal']
ra_rad = np.deg2rad(tde['ra_deg'])
dec_rad = np.deg2rad(tde['dec_deg'])
lgMBH_mass = tde['lgMBH']
initial_MBHspin = tde['initial_MBHspin']
star_masses = tde['star_mass']
t0_mjd = tde['t0_mjd_transient']
distances_Gpc = cosmo_model.luminosity_distance(z_gal, unit='Gpc')

# ============================================================
# Sample transient-model parameters, one draw per event (no nested
# per-galaxy loop needed -- each catalog row is already one event).
# ============================================================
eta_min, eta_max = args.eta
alpha_min, alpha_max = args.alpha
beta_min, beta_max = args.beta


def loguniform(low, high, size, rng):
    return np.exp(rng.uniform(np.log(low), np.log(high), size))


eta_samples = loguniform(eta_min, eta_max, n_events, rng)
alpha_samples = loguniform(alpha_min, alpha_max, n_events, rng)
beta_samples = rng.uniform(beta_min, beta_max, size=n_events)

# ============================================================
# Save flat output: one row per event, no galaxy grouping.
# ============================================================
out_path = f'{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}/{TDE_DATA_FILE_NAME}_{args.mbh_mass_method}_{args.nucleation_method}_{args.UNIVERSE_MODEL}.h5'

with h5py.File(out_path, "w") as f:
    f.attrs['n_events'] = n_events

    f.create_dataset("lgMBH_mass", data=lgMBH_mass)
    f.create_dataset("distance_Gpc", data=distances_Gpc)
    f.create_dataset("z_gal", data=z_gal)
    f.create_dataset("initial_MBHspin", data=initial_MBHspin)
    f.create_dataset("ra_rad", data=ra_rad)
    f.create_dataset("dec_rad", data=dec_rad)
    f.create_dataset("star_mass", data=star_masses)
    f.create_dataset("eta", data=eta_samples)
    f.create_dataset("alpha", data=alpha_samples)
    f.create_dataset("beta", data=beta_samples)
    f.create_dataset("t0_mjd_transient", data=t0_mjd)

print(f"Saved {n_events} TDE parameter rows to {out_path}")