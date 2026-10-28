"""TDE light-curve models: extra-parameter sampling and the trained detectability classifier (shared by gen_TDE_params.py, calculate_SNR_TDE.py and the selection function).

Per-model configuration MUST match training_TDE_detectability/prep_TDE_params_detectability.py
(same MODELS, SHARED_KEYS order and EXTRA_PARAM_SCALING). bpl_cooling_envelope and
gaussianrise_cooling_envelope are excluded: both crash under redback's SimulateOpticalTransient.
"""
import pickle

import h5py
import numpy as np
import torch

MODELS = ["cooling_envelope", "stream_stream_tde", "tde_analytical", "tde_fallback"]

SHARED_KEYS = ["lgMBH", "z_gal", "ra_rad", "dec_rad", "eta", "alpha", "beta", "t0_mjd_transient"]

EXTRA_PARAM_SCALING = {
    "cooling_envelope":   [],
    "stream_stream_tde":  [("peak_time", "log"), ("sigma_t", "log")],
    "tde_analytical":     [("mej", "log"), ("vej", "log"), ("kappa", "linear"),
                           ("kappa_gamma", "log"), ("temperature_floor", "linear"),
                           ("l0", "log"), ("t_0_turn", "log")],
    "tde_fallback":       [("tvisc", "log"), ("bb", "linear"),
                           ("rph0", "log"), ("lphoto", "linear")],
}

# training key -> catalogue key
KEY_MAP = {"lgMBH": "lgMBH_mass"}

# Survey observing windows (MJD)
SURVEY_WINDOWS = {"ZTF": (58200, 61396), "LSST": (60827, 64423)}


def loguniform(low, high, size, rng):
    """Log-uniform samples between low and high."""
    return np.exp(rng.uniform(np.log(low), np.log(high), size))


def sample_extra_params(model, n, rng):
    """Draw the model-specific light-curve parameters for n events."""
    if model == "stream_stream_tde":
        return {"peak_time": loguniform(5.0, 40.0, n, rng), "sigma_t": loguniform(5.0, 30.0, n, rng)}
    if model == "tde_analytical":
        return {"mej": loguniform(0.1, 5.0, n, rng), "vej": loguniform(3000.0, 15000.0, n, rng),
                "kappa": rng.uniform(0.1, 1.0, n), "kappa_gamma": loguniform(0.0001, 10000.0, n, rng),
                "temperature_floor": rng.uniform(10000.0, 50000.0, n),
                "l0": loguniform(1e51, 1e60, n, rng), "t_0_turn": loguniform(0.0001, 500.0, n, rng)}
    if model == "tde_fallback":
        return {"tvisc": loguniform(1.0, 30.0, n, rng), "bb": rng.uniform(0.0, 2.0, n),
                "rph0": loguniform(1e-4, 1e4, n, rng), "lphoto": rng.uniform(0.0, 2.0, n)}
    return {}   # cooling_envelope: no extra parameters


def sample_transient_params(n, rng, eta=(0.001, 0.1), alpha=(0.01, 0.1), beta=(1.001, 30.0)):
    """Draw eta, alpha (log-uniform) and beta (uniform) as in gen_TDE_params.py."""
    return {"eta": loguniform(*eta, n, rng), "alpha": loguniform(*alpha, n, rng),
            "beta": rng.uniform(*beta, n)}


def rescale_between(x, x_min, x_max, t_min, t_max):
    """Exact replica of the training script rescale_between()."""
    if x_max == x_min:
        return np.full_like(x, t_min, dtype=float)
    return (x - x_min) / (x_max - x_min) * (t_max - t_min) + t_min


def load_classifier(train_dir, model_name, survey, num_neurons, layers, device):
    """(classifier, scaling, feature_keys) of the trained (TDE model, survey) detectability network."""
    model_path = (f"{train_dir}/1E6_events/TDE_DETECT_BCE_no_rescaler_{model_name}_{num_neurons}_{layers}_10_{survey}/"
                  f"best_final_model/model/model.pth")
    train_h5 = f"{train_dir}/1E6_events/TDE_params_1E6_{survey}_{model_name}.h5"
    extra_cols = EXTRA_PARAM_SCALING[model_name]
    feature_keys = SHARED_KEYS + [k for k, _ in extra_cols]

    scaling = {}
    with h5py.File(train_h5, "r") as f:
        for key in feature_keys:
            data = f[key][:]
            kind = "log" if key in ("eta", "alpha") or dict(extra_cols).get(key) == "log" else "linear"
            if kind == "log":
                scaling[key] = ("log", np.log10(data).min(), np.log10(data).max(), -1.0, 1.0)
            else:
                scaling[key] = ("linear", data.min(), data.max(), 0.0, 1.0)

    with open(model_path, "rb") as f:
        classifier = pickle.load(f)
    classifier.to(device)
    classifier.eval()
    return classifier, scaling, feature_keys


def predict_detection_prob(classifier, scaling, feature_keys, catalog, device, batch_size=4096):
    """Detection probability of each event of a catalogue dict, and the mask of events outside the training range."""
    n = len(catalog["z_gal"])
    X = np.zeros((n, len(feature_keys)), dtype=np.float32)
    outside = np.zeros(n, dtype=bool)
    for col, train_key in enumerate(feature_keys):
        raw = np.asarray(catalog[KEY_MAP.get(train_key, train_key)], dtype=float)
        kind, x_min, x_max, t_min, t_max = scaling[train_key]
        X[:, col] = rescale_between(np.log10(raw) if kind == "log" else raw, x_min, x_max, t_min, t_max)
        outside |= (X[:, col] < t_min) | (X[:, col] > t_max)

    X = torch.tensor(X)
    probs = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            probs.append(torch.sigmoid(classifier(X[i:i + batch_size].to(device))).reshape(-1).cpu().numpy())
    return (np.concatenate(probs).astype(np.float32) if probs else np.array([], dtype=np.float32)), outside
