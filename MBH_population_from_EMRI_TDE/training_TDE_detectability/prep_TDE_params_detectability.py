#!/usr/bin/env python
"""Scale TDE parameters for the detectability classifier; must match the generation-time scaling."""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import h5py

# bpl_cooling_envelope and gaussianrise_cooling_envelope are excluded: both
# crash deterministically (KeyError: 'bands') under redback's
# SimulateOpticalTransient (confirmed still broken in redback 1.18.0).
MODELS = [
    "cooling_envelope",
    "stream_stream_tde",
    "tde_analytical",
    "tde_fallback",
]

SHARED_KEYS = ["lgMBH", "z_gal", "ra_rad", "dec_rad", "eta", "alpha", "beta", "t0_mjd_transient"]

# Model-specific extra parameters used as features, with their scaling rule.
# "log" -> log10 then rescale to [-1,1] (matches how eta/alpha are treated,
# since these were all sampled log-uniformly); "linear" -> rescale to [0,1].
# Fixed (constant) params sampled by gen_TDE_params.py (stream_stream_tde's
# c1/del_omega/f/h_r/inc_tcool, tde_fallback's leddlimit) carry no
# information and are excluded here.
EXTRA_PARAM_SCALING = {
    "cooling_envelope":              [],
    "stream_stream_tde":             [("peak_time", "log"), ("sigma_t", "log")],
    "tde_analytical":                [("mej", "log"), ("vej", "log"), ("kappa", "linear"),
                                       ("kappa_gamma", "log"), ("temperature_floor", "linear"),
                                       ("l0", "log"), ("t_0_turn", "log")],
    "tde_fallback":                  [("tvisc", "log"), ("bb", "linear"),
                                       ("rph0", "log"), ("lphoto", "linear")],
}

# -----------------------------
# Argument parser
# -----------------------------
parser = argparse.ArgumentParser(description="Generate per-model scaled training data.")
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

SHARED_SCALING_RULES = {
    # linear scaling to [0,1]
    "lgMBH": (lambda x: x, 0.0, 1.0),
    "z_gal": (lambda x: x, 0.0, 1.0),
    "ra_rad": (lambda x: x, 0.0, 1.0),
    "dec_rad": (lambda x: x, 0.0, 1.0),
    "beta": (lambda x: x, 0.0, 1.0),

    # log scaling then rescale to [-1,1]
    "eta": (np.log10, -1.0, 1.0),
    "alpha": (np.log10, -1.0, 1.0),
    "t0_mjd_transient": (lambda x: x, 0.0, 1.0),
}

def scale_column(x, kind):
    if kind == "log":
        return rescale_between(np.log10(x), -1.0, 1.0)
    return rescale_between(x, 0.0, 1.0)

# -----------------------------
# File paths
# -----------------------------
events_for_dir = f"{args.events:.0E}".replace("+0", "").replace(".0", "")
output_dir = f'./data/PRE_TRAIN_DATA_TDE/{events_for_dir}_events/'
n_str = f"{int(np.log10(args.events))}"

# -----------------------------
# Build one scaled feature matrix PER (survey, model) -- each model is
# trained as its own independent classifier, so there is no cross-model
# concatenation or one-hot column here.
# -----------------------------
for SURVEY in ["ZTF", "LSST"]:
    for model in MODELS:
        params_file = f"{output_dir}/TDE_params_1E{n_str}_{SURVEY}_{model}.h5"
        extra_cols = EXTRA_PARAM_SCALING[model]
        feature_keys = SHARED_KEYS + [k for k, _ in extra_cols]

        print(f"[{SURVEY}][{model}] Processing {len(feature_keys)} feature columns...")

        with h5py.File(params_file, "r") as hf:
            num_events = len(hf["lgMBH"])
            scaled = np.zeros((num_events, len(feature_keys)))

            for i, key in enumerate(SHARED_KEYS):
                transform_fn, target_min, target_max = SHARED_SCALING_RULES[key]
                scaled[:, i] = rescale_between(transform_fn(hf[key][:]), target_min, target_max)

            for j, (key, kind) in enumerate(extra_cols):
                scaled[:, len(SHARED_KEYS) + j] = scale_column(hf[key][:], kind)

        out_path = f"{output_dir}/TDE_params_scaled_1E{n_str}_{SURVEY}_{model}.npy"
        np.save(out_path, scaled)
        print(f"[{SURVEY}][{model}] Feature column order: {feature_keys}")
        print(f"[{SURVEY}][{model}] Saved -> {out_path}  shape={scaled.shape}")

print("All models/surveys rescaled and saved successfully!")
