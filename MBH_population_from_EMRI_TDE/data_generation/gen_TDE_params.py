#!/usr/bin/env python
"""Draw the full TDE parameter set for the chosen light-curve model for the generated TDE catalogue."""

import numpy as np
import argparse
import os, sys
import h5py

sys.path.insert(0, os.path.abspath('./astrophysical_setup'))
sys.path.insert(0, os.path.abspath('./'))
from utils import ASTROPHYSICAL_DATA_DIR, MOCK_GALAXY_CATALOG_DATA, TDE_DATA_FILE_NAME, MERGER_MODEL_CHOICES
from cosmology import CosmologyModel

cosmo_model = CosmologyModel()

# ============================================================
# Per-TDE-model config -- MUST match training_TDE_detectability/
# gen_TDE_params.py and prep_TDE_params_detectability.py exactly (same
# MODELS list, same extra-param ranges, same key names), since
# calculate_SNR_TDE.py feeds these events through the classifiers
# trained on that data.
# bpl_cooling_envelope and gaussianrise_cooling_envelope are excluded:
# both crash deterministically (KeyError: 'bands') under redback's
# SimulateOpticalTransient (confirmed still broken in redback 1.18.0).
# ============================================================
MODELS = [
    "cooling_envelope",
    "stream_stream_tde",
    "tde_analytical",
    "tde_fallback",
]

# ============================================================
# Argument parsing
# ============================================================
parser = argparse.ArgumentParser(description="Generate TDE transient parameters.")
parser.add_argument("--UNIVERSE_MODEL", type=int, required=True, help="Universe model index")
parser.add_argument("--GALAXIES", type=int, required=True, help="Monte Carlo sample size used to label the run directory")
parser.add_argument("--TDE_model", type=str, required=True, choices=MODELS,
                     help="redback TDE model to sample extra params for. Every event in the "
                          "source TDE_catalog gets modeled by THIS one model -- comparing "
                          "models means running this script once per model, not splitting "
                          "events across models.")
parser.add_argument("--eta", type=float, nargs=2, required=True, help="eta range [min, max] (log-uniform)")
parser.add_argument("--alpha", type=float, nargs=2, required=True, help="alpha range [min, max] (log-uniform)")
parser.add_argument("--beta", type=float, nargs=2, required=True, help="beta range [min, max] (uniform)")
parser.add_argument("--mbh_mass_method", type=str, required=True, default="use_sigma", choices=["use_sigma", "use_lgMgal"])
parser.add_argument("--nucleation_method", type=str, required=True, default="hannah", choices=["hannah", "neumayer"])
parser.add_argument("--occupation_method", type=str, required=True, default="uniform", choices=["uniform", "varying"])
parser.add_argument("--merger_model", type=str, required=True, choices=MERGER_MODEL_CHOICES,
                    help="Galaxy major-merger-rate model (see astrophysical_setup/cosmology.py MERGER_MODELS)")
parser.add_argument("--SEED", type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()

rng = np.random.default_rng(seed=args.SEED)

# ============================================================
# Load resampled TDE catalog (flat, one row = one event)
# ============================================================
loc = f'{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}'
catalog_path = f'{loc}/{MOCK_GALAXY_CATALOG_DATA}_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}_{args.merger_model}_{args.UNIVERSE_MODEL}.h5'

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
    """Log-uniform samples between low and high."""
    return np.exp(rng.uniform(np.log(low), np.log(high), size))


eta_samples = loguniform(eta_min, eta_max, n_events, rng)
alpha_samples = loguniform(alpha_min, alpha_max, n_events, rng)
beta_samples = rng.uniform(beta_min, beta_max, size=n_events)

# ============================================================
# Sample this ONE model's extra parameters for EVERY event in the
# source catalog (not a per-event split) -- comparing models means
# running this script once per --TDE_model on the same full event set,
# so detection rates are comparable model-to-model on identical events.
# Ranges/citations match training_TDE_detectability/gen_TDE_params.py.
# ============================================================
MODEL = args.TDE_model
extra_data = {}

if MODEL == "stream_stream_tde":
    extra_data["peak_time"] = loguniform(5.0, 40.0, n_events, rng)
    extra_data["sigma_t"] = loguniform(5.0, 30.0, n_events, rng)

elif MODEL == "tde_analytical":
    extra_data["mej"] = loguniform(0.1, 5.0, n_events, rng)
    extra_data["vej"] = loguniform(3000.0, 15000.0, n_events, rng)
    extra_data["kappa"] = rng.uniform(0.1, 1.0, size=n_events)
    extra_data["kappa_gamma"] = loguniform(0.0001, 10000.0, n_events, rng)
    extra_data["temperature_floor"] = rng.uniform(10000.0, 50000.0, size=n_events)
    extra_data["l0"] = loguniform(1e51, 1e60, n_events, rng)
    extra_data["t_0_turn"] = loguniform(0.0001, 500.0, n_events, rng)

elif MODEL == "tde_fallback":
    extra_data["tvisc"] = loguniform(1.0, 30.0, n_events, rng)
    extra_data["bb"] = rng.uniform(0.0, 2.0, size=n_events)
    extra_data["rph0"] = loguniform(1e-4, 1e4, n_events, rng)
    extra_data["lphoto"] = rng.uniform(0.0, 2.0, size=n_events)

# cooling_envelope: no extra params beyond the shared set.

print(f"Sampled extra params for model '{MODEL}': {list(extra_data.keys())}")

# ============================================================
# Save flat output: one row per event, no galaxy grouping.
# ============================================================
out_path = f'{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}/{TDE_DATA_FILE_NAME}_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}_{args.merger_model}_{MODEL}_{args.UNIVERSE_MODEL}.h5'

with h5py.File(out_path, "w") as f:
    f.attrs['n_events'] = n_events
    f.attrs['model'] = MODEL

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
    f.create_dataset("model", data=np.full(n_events, MODEL, dtype=h5py.string_dtype()))
    for key, arr in extra_data.items():
        f.create_dataset(key, data=arr)

print(f"Saved {n_events} TDE parameter rows (model={MODEL}) to {out_path}")