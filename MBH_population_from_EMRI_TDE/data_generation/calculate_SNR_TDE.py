#!/usr/bin/env python
"""Detection probability of each generated TDE for a survey (ZTF/LSST) using the trained detectability classifier."""

import os, sys, pickle, argparse
import numpy as np
import h5py
import torch

sys.path.insert(0, os.path.abspath('../poplar'))
sys.path.insert(0, os.path.abspath('./'))
from utils import ASTROPHYSICAL_DATA_DIR, TDE_DATA_FILE_NAME, TRAIN_DIR_TDE, MERGER_MODEL_CHOICES

# ─────────────────────────────────────────────────────────────
# Per-TDE-model config -- MUST match training_TDE_detectability/
# prep_TDE_params_detectability.py exactly (same MODELS list, same
# SHARED_KEYS order, same EXTRA_PARAM_SCALING), since this is the
# classifier each event's row gets run through below.
# ─────────────────────────────────────────────────────────────
MODELS = [
    "cooling_envelope",
    "stream_stream_tde",
    "tde_analytical",
    "tde_fallback",
]

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

# Map: training key -> catalog key (catalog produced by data_generation/gen_TDE_params.py)
KEY_MAP = {
    'alpha':   'alpha',
    'beta':    'beta',
    'dec_rad': 'dec_rad',
    'eta':     'eta',
    'lgMBH':   'lgMBH_mass',
    'ra_rad':  'ra_rad',
    't0_mjd_transient': 't0_mjd_transient',
    'z_gal':   'z_gal',
    # extra params share the same name in both the training catalog and
    # the population catalog (see gen_TDE_params.py)
}

# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--UNIVERSE_MODEL", type=int, required=True, help="Universe model index")
parser.add_argument("--GALAXIES",      type=int, required=True)
parser.add_argument("--SURVEY",        type=str, default="ZTF", choices=["ZTF", "LSST"])
parser.add_argument("--TDE_model",     type=str, required=True, choices=MODELS,
                     help="redback TDE model whose catalog/classifier to use -- must match "
                          "the --TDE_model this universe's gen_TDE_params.py run used.")
parser.add_argument("--Z_GAL_MIN",    type=float, default=0.0)
parser.add_argument("--Z_GAL_MAX",    type=float, default=10.0)
parser.add_argument("--batch_size",    type=int, default=4096)
parser.add_argument("--mbh_mass_method", type=str, required=True, default="use_sigma", choices=["use_sigma", "use_lgMgal"])
parser.add_argument("--nucleation_method", type=str, required=True, default="hannah", choices=["hannah", "neumayer"])
parser.add_argument("--occupation_method", type=str, required=True, default="uniform", choices=["uniform", "varying"])
parser.add_argument("--merger_model", type=str, required=True, choices=MERGER_MODEL_CHOICES,
                    help="Galaxy major-merger-rate model (see astrophysical_setup/cosmology.py MERGER_MODELS)")
parser.add_argument("--DETECTION_THRESHOLD", type=float, default=0.5,
                     help="Detection-probability threshold applied to the trained "
                          "classifier's output. Lower this (e.g. 0.2-0.3) to recover "
                          "more events in low-rate scenarios -- this does NOT require "
                          "retraining, it only moves the cut on already-computed "
                          "probabilities.")
parser.add_argument("--num_neurons", type=int, required=True,
                     help="Hidden-layer width of the trained (TDE_model, SURVEY) classifier "
                          "to load -- pick the best-performing architecture (see "
                          "EXECUTABLES/config.sh's TDE_ARCH_NEURONS_<model>_<survey> table).")
parser.add_argument("--layers", type=int, required=True,
                     help="Number of hidden layers of the trained classifier to load -- see "
                          "EXECUTABLES/config.sh's TDE_ARCH_LAYERS_<model>_<survey> table.")
args = parser.parse_args()

DETECTION_THRESHOLD = args.DETECTION_THRESHOLD
MODEL = args.TDE_model

MODEL_PATH      = (f"{TRAIN_DIR_TDE}/1E6_events/"
                    f"TDE_DETECT_BCE_no_rescaler_{MODEL}_{args.num_neurons}_{args.layers}_10_{args.SURVEY}/"
                    f"best_final_model/model/model.pth")
TRAIN_H5_PATH   = f"{TRAIN_DIR_TDE}/1E6_events/TDE_params_1E6_{args.SURVEY}_{MODEL}.h5"

DATA_PATH = f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}/{TDE_DATA_FILE_NAME}_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}_{args.merger_model}_{MODEL}_{args.UNIVERSE_MODEL}.h5"
OUT_PATH  = f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}/TDE_SNRs_{args.SURVEY}_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}_{args.merger_model}_{MODEL}_{args.UNIVERSE_MODEL}.h5"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Trained classifier checkpoint missing: {MODEL_PATH}\n"
        f"Train it via training_TDE_detectability/train_TDE_detectability.sub "
        f"({MODEL}, neurons={args.num_neurons}, layers={args.layers}, SURVEY={args.SURVEY}).")

extra_cols = EXTRA_PARAM_SCALING[MODEL]
feature_keys = SHARED_KEYS + [k for k, _ in extra_cols]

SCALING = {}
with h5py.File(TRAIN_H5_PATH, "r") as f:
    for key in SHARED_KEYS:
        data = f[key][:]
        if key in ("eta", "alpha"):
            SCALING[key] = ("log", np.log10(data).min(), np.log10(data).max(), -1.0, 1.0)
        else:
            SCALING[key] = ("linear", data.min(), data.max(), 0.0, 1.0)
    for key, kind in extra_cols:
        data = f[key][:]
        if kind == "log":
            SCALING[key] = ("log", np.log10(data).min(), np.log10(data).max(), -1.0, 1.0)
        else:
            SCALING[key] = ("linear", data.min(), data.max(), 0.0, 1.0)

print(SCALING)


def rescale_between(x, x_min, x_max, t_min, t_max):
    """Exact replica of the training script rescale_between()."""
    if x_max == x_min:
        return np.full_like(x, t_min, dtype=float)
    return (x - x_min) / (x_max - x_min) * (t_max - t_min) + t_min

# ─────────────────────────────────────────────────────────────
# Load catalog
# ─────────────────────────────────────────────────────────────
print(f"Loading catalog: {DATA_PATH}")
with h5py.File(DATA_PATH, "r") as f:
    catalog = {k: f[k][:] for k in f.keys()}
n_events = len(catalog["z_gal"])
print(f"  {n_events} events")

# ── Survey observing windows (MJD) ────────────────────────────
SURVEY_WINDOWS = {
    "ZTF":  (58200, 61396),   # 23 Mar 2018 → Dec 2026
    "LSST": (60827, 64423),   # Jun 2025   → Jun 2035
}

# ── 1. Redshift cut ───────────────────────────────────────────
z = catalog["z_gal"]
z_mask = (z >= args.Z_GAL_MIN) & (z <= args.Z_GAL_MAX)

# ── 2. Survey time window cut ─────────────────────────────────
t0_min, t0_max = SURVEY_WINDOWS[args.SURVEY]
t0 = catalog['t0_mjd_transient']
t0_mask = (t0 >= t0_min) & (t0 <= t0_max)

# ── 3. Combined mask ──────────────────────────────────────────
combined_mask = z_mask & t0_mask
catalog = {k: v[combined_mask] for k, v in catalog.items()}
n_events = len(catalog["z_gal"])

print(f"  After Z_CUTOFF [{args.Z_GAL_MIN}, {args.Z_GAL_MAX}]: "
      f"{z_mask.sum()} events")
print(f"  After t0_CUTOFF [{t0_min}, {t0_max}] MJD ({args.SURVEY}): "
      f"{t0_mask.sum()} events")
print(f"  After combined cut: {n_events} events "
      f"({combined_mask.sum() / len(z) * 100:.1f}% of original)")

if n_events == 0:
    print("[WARN] 0 events survive the redshift/time-window cut, BEFORE the classifier even runs.")
    with h5py.File(OUT_PATH, "w") as f:
        f.attrs["survey"] = args.SURVEY
        f.attrs["model"] = MODEL
        f.attrs["n_events_total"] = 0
        f.attrs["n_events_passed"] = 0
        f.attrs["detection_threshold"] = DETECTION_THRESHOLD
        f.create_dataset("detection_prob", data=np.array([], dtype=np.float32))
    exit()

# ─────────────────────────────────────────────────────────────
# Build scaled input matrix
# ─────────────────────────────────────────────────────────────
X = np.zeros((n_events, len(feature_keys)), dtype=np.float32)

for col, train_key in enumerate(feature_keys):
    cat_key = KEY_MAP.get(train_key, train_key)
    raw = catalog[cat_key].astype(float)

    kind, x_min, x_max, t_min, t_max = SCALING[train_key]
    vals = np.log10(raw) if kind == "log" else raw
    X[:, col] = rescale_between(vals, x_min, x_max, t_min, t_max)

    n_out = ((X[:, col] < t_min) | (X[:, col] > t_max)).sum()
    if n_out > 0:
        print(f"  [WARN] {train_key}: {n_out} events ({n_out/n_events:.1%}) "
              f"outside training range — model extrapolates")

print(f"Input matrix: {X.shape}  min={X.min():.3f}  max={X.max():.3f}")

# ─────────────────────────────────────────────────────────────
# Load model (pickle — not torch.load)
# ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on {device}")

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
model.to(device)
model.eval()

# ─────────────────────────────────────────────────────────────
# Batched inference
# ─────────────────────────────────────────────────────────────
X_tensor   = torch.tensor(X, dtype=torch.float32)
probs_list = []

with torch.no_grad():
    for i in range(0, n_events, args.batch_size):
        batch  = X_tensor[i : i + args.batch_size].to(device)
        logits = model(batch)
        probs  = torch.sigmoid(logits).reshape(-1).cpu().numpy()
        probs_list.append(probs)

detection_prob = np.concatenate(probs_list).astype(np.float32) if probs_list else np.array([], dtype=np.float32)

if n_events > 0:
    print(f"\nDetected (p>0.5): {(detection_prob > 0.5).sum()} / {n_events} "
          f"({(detection_prob > 0.5).mean():.2%})")
    print(f"Prob range      : [{detection_prob.min():.4f}, {detection_prob.max():.4f}]")
else:
    print("\nNo events to classify.")

# ─────────────────────────────────────────────────────────────
# Select events passing the detection-probability threshold
# ─────────────────────────────────────────────────────────────
events_idx = np.where(detection_prob > DETECTION_THRESHOLD)[0]
print(f"{len(events_idx)}/{n_events} events pass detection threshold "
      f"p > {DETECTION_THRESHOLD}")

if len(events_idx) == 0 and n_events > 0:
    print("WARNING: no events pass the detection threshold -- writing an "
          "empty file (downstream scripts should handle n_events=0, or "
          "check this print statement before running them).")
    print(f"  (For reference: the highest probability among these {n_events} "
          f"events was {detection_prob.max():.4f}, threshold was "
          f"{DETECTION_THRESHOLD}. If that max is not wildly below threshold, "
          f"--DETECTION_THRESHOLD may just need lowering for this scenario.)")

# ─────────────────────────────────────────────────────────────
# Save -- filtered to detection-passing events only
# ─────────────────────────────────────────────────────────────
with h5py.File(OUT_PATH, "w") as f:
    f.attrs["survey"]         = args.SURVEY
    f.attrs["model"]          = MODEL
    f.attrs["n_events_total"] = n_events
    f.attrs["n_events_passed"] = len(events_idx)
    f.attrs["detection_threshold"] = DETECTION_THRESHOLD
    f.attrs["param_order"]    = str(feature_keys)
    f.create_dataset("detection_prob", data=detection_prob[events_idx], compression="gzip")
    for k in ["z_gal", "lgMBH_mass", "distance_Gpc", "eta", "alpha", "beta", "star_mass",
              "initial_MBHspin", "ra_rad", "dec_rad", "t0_mjd_transient"]:
        if k in catalog:
            f.create_dataset(k, data=catalog[k][events_idx], compression="gzip")

print(f"Saved {len(events_idx)}/{n_events} detection-passing TDE rows to {OUT_PATH}")


# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# if n_events > 0:
#     finite = np.isfinite(detection_prob)
#     plt.figure(figsize=(7, 6))
#     sc = plt.scatter(catalog['lgMBH_mass'][finite], detection_prob[finite],
#                       c=catalog['z_gal'][finite], cmap='viridis', s=8)
#     plt.colorbar(sc, label='z')
#     plt.axhline(DETECTION_THRESHOLD, color='red', ls='--', lw=1,
#                 label=f'threshold={DETECTION_THRESHOLD:.2f}')
#     plt.xlabel(r'$\log_{10}(M_{\rm MBH}/M_\odot)$')
#     plt.ylabel('Detection probability')
#     plt.title(f'{MODEL} ({args.SURVEY})')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}/TDE_detprob_MBHmass_{args.SURVEY}_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}_{args.merger_model}_{MODEL}_{args.UNIVERSE_MODEL}.png")
#     plt.close()

# # plot cumulative plot for detections with t0_mjd_transient
# t = catalog["t0_mjd_transient"] if n_events > 0 else np.array([])
# mask = detection_prob > DETECTION_THRESHOLD if n_events > 0 else np.array([], dtype=bool)

# t_detected = t[mask]

# if len(t_detected) > 0:
#     year = 2000 + (t_detected - 51544.5) / 365.25

#     year_min = int(np.floor(year.min()))
#     year_max = int(np.ceil(year.max()))
#     bins = np.arange(year_min, year_max + 1)

#     counts, edges = np.histogram(year, bins=bins)
#     cumulative_counts = np.cumsum(counts)

#     plt.figure(figsize=(8, 6))
#     plt.step(edges[:-1], cumulative_counts, where='post')
#     plt.xlabel("Year")
#     plt.ylabel("Cumulative detections")
#     plt.title(f"{args.SURVEY} cumulative detections ({MODEL})")
#     plt.grid()

#     plt.savefig(f"{ASTROPHYSICAL_DATA_DIR}/cumulative_detections_{args.SURVEY}_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}_{args.merger_model}_{MODEL}_{args.UNIVERSE_MODEL}.png")
# else:
#     print("[INFO] No detected TDEs -- skipping cumulative-detections plot.")
