"""Loading and evaluation of the trained EMRI SNR network (shared by calculate_SNR_EMRI.py and the selection function)."""
import json
import pickle

import numpy as np
import torch

# Network input order (distance and the three initial phases are not inputs); must match
# the column order of injection_params_dist_REMOVED_COMBINED.npy in training_EMRI_SNR/.
FEATURES = ["lgMBH_mass", "MBHspin", "p0", "e0", "Y0", "qS", "phiS", "qK", "phiK",
            "T_SIGNAL_duration_years"]


def load_snr_model(model_path, scaling_path, device):
    """(model, y_min, y_max): the network and the bounds needed to invert its output."""
    with open(scaling_path) as f:
        scaling = json.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    model.to(device)
    model.eval()
    return model, scaling["y_min"], scaling["y_max"]


def predict_snr(model, y_min, y_max, events, device, batch_size=100000):
    """SNR of a dict of EMRI parameters (arrays); NaN where an input is not finite."""
    X = np.stack([events[k] for k in FEATURES], axis=1).astype(np.float32)
    dist_gpc = np.asarray(events["distance_Gpc"])
    valid = np.isfinite(X).all(axis=1) & (dist_gpc > 0)
    y = np.full(len(X), np.nan)
    Xv = torch.tensor(X[valid])
    out = []
    with torch.no_grad():
        for i in range(0, len(Xv), batch_size):
            out.append(model(Xv[i:i + batch_size].to(device)).reshape(-1).cpu().numpy())
    if out:
        y[valid] = np.concatenate(out)
    return np.where(valid, dist_gpc * 10 ** (y * (y_max - y_min) + y_min), np.nan)
