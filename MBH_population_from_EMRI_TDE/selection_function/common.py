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
# priors.sh variable stem that differs from the hyperparameter name
PRIOR_STEM = {"xc": "XC_M", "lam_z_gal": "Z_GAL_LAM"}


def channel_names(population, channel):
    """Hyperparameters the channel's selection function depends on."""
    return SHARED[population] + (EMRI_ONLY[population] if channel == "EMRI" else [])


def load_prior_bounds(population, priors_sh="EXECUTABLES/priors.sh"):
    """{hyperparameter: (low, high)} parsed from the <POP>_PRIOR_<NAME>_LOW/HIGH lines of priors.sh."""
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


def tde_detection_probability(events, survey, args):
    """Detection probability of the (TDE model, survey) classifier for a dict of complete TDE parameters.

    Uses the same classifier, inputs and scaling as data_generation/calculate_SNR_TDE.py.
    """
    raise NotImplementedError("wrap the classifier call of data_generation/calculate_SNR_TDE.py here")
