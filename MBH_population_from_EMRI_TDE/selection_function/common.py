"""Shared pieces of the selection-function pipeline: hyperparameter names, prior bounds and detection probabilities."""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.abspath('./'))
sys.path.insert(0, os.path.abspath('./astrophysical_setup'))

CHANNELS = ["EMRI", "TDE_ZTF", "TDE_LSST"]

# Hyperparameters entering each channel, in the order used by the inference (PopulationModel._param_names)
SHARED = {"A": ["xc", "lam_schechter", "lam_z_gal"], "B": ["lambda_M", "lam_z_gal"]}
EMRI_ONLY = {
    "A": ["alpha_1_MBHSpin", "alpha_2_MBHSpin", "beta_1_MBHSpin", "beta_2_MBHSpin", "weight_MBHSpin",
          "alpha_1_e_pl", "alpha_2_e_pl", "beta_1_e_pl", "beta_2_e_pl", "weight_e_pl"],
    "B": ["mu_1_MBHSpin", "sigma_1_MBHSpin", "mu_2_MBHSpin", "sigma_2_MBHSpin", "weight_MBHSpin",
          "mu_1_e_pl", "sigma_1_e_pl", "mu_2_e_pl", "sigma_2_e_pl", "weight_e_pl"],
}
# run-script variable stem that differs from the hyperparameter name
PRIOR_STEM = {"xc": "XC_M", "lam_z_gal": "Z_GAL_LAM"}


def channel_tag(channel, tde_model=None):
    """Folder/file tag of a channel; TDE channels depend on the TDE light-curve model."""
    return f"{channel}_{tde_model}" if channel != "EMRI" and tde_model else channel


def channel_names(population, channel):
    """Hyperparameters the channel's selection function depends on."""
    return SHARED[population] + (EMRI_ONLY[population] if channel == "EMRI" else [])


def load_prior_bounds(population, priors_sh="hierarchical_inference/runs/run_inference.sh"):
    """{hyperparameter: (low, high)} parsed from the <POP>_PRIOR_<NAME>_LOW/HIGH lines of the inference run script."""
    values = {}
    with open(priors_sh) as f:
        for line in f:
            m = re.match(rf"{population}_PRIOR_(\w+?)_(LOW|HIGH)=([-+0-9.eE]+)", line)
            if m:
                values[(m.group(1).upper(), m.group(2))] = float(m.group(3))
    bounds = {}
    for name in SHARED[population] + EMRI_ONLY[population]:
        stem = PRIOR_STEM.get(name, name).upper()
        bounds[name] = (values[(stem, "LOW")], values[(stem, "HIGH")])
    return bounds


def emri_detection_probability(events, args):
    """1 where the SNR network gives SNR > threshold, else 0, for a dict of complete EMRI parameters.

    Uses the same network, inputs and scaling as data_generation/calculate_SNR_EMRI.py.
    """
    import torch
    sys.path.insert(0, os.path.abspath('./data_generation'))
    from emri_snr_model import load_snr_model, predict_snr
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, y_min, y_max = load_snr_model(args.MODEL_PATH, args.SCALING_PATH, device)
    snr = predict_snr(model, y_min, y_max, events, device)
    return (snr > args.SNR_THRESHOLD).astype(float)


_CLASSIFIERS = {}


def tde_detection_probability(draws, survey, args, rng):
    """Detection indicator (classifier probability > threshold) for TDEs with population draws `draws`.

    The remaining parameters (sky position, t0 inside the survey window, eta/alpha/beta and the
    model-specific ones) are drawn as in data_generation/gen_TDE_params.py; the classifier and its
    scaling are those used by data_generation/calculate_SNR_TDE.py.
    """
    import torch
    sys.path.insert(0, os.path.abspath('./data_generation'))
    from tde_detect_model import (SURVEY_WINDOWS, load_classifier, predict_detection_prob,
                                  sample_extra_params, sample_transient_params)
    from utils import TRAIN_DIR_TDE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    key = (args.TDE_MODEL, survey)
    if key not in _CLASSIFIERS:
        _CLASSIFIERS[key] = load_classifier(TRAIN_DIR_TDE, args.TDE_MODEL, survey,
                                            args.num_neurons, args.layers, device)
    classifier, scaling, feature_keys = _CLASSIFIERS[key]

    n = len(draws["z_gal"])
    t_lo, t_hi = SURVEY_WINDOWS[survey]
    catalog = {"lgMBH_mass": draws["lgMBH_mass"], "z_gal": draws["z_gal"],
               "ra_rad": rng.uniform(0, 2 * np.pi, n), "dec_rad": np.arcsin(rng.uniform(-1, 1, n)),
               "t0_mjd_transient": rng.uniform(t_lo, t_hi, n),
               **sample_transient_params(n, rng), **sample_extra_params(args.TDE_MODEL, n, rng)}
    prob, _ = predict_detection_prob(classifier, scaling, feature_keys, catalog, device)
    return ((prob > args.DETECTION_THRESHOLD) & (draws["z_gal"] <= args.Z_GAL_MAX)).astype(float)
