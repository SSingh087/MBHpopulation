#!/usr/bin/env python

import os, sys, pickle, argparse
import numpy as np
import h5py
import torch

sys.path.insert(0, os.path.abspath('../poplar'))
sys.path.insert(0, os.path.abspath('./'))
from utils import ASTROPHYSICAL_DATA_DIR, TDE_DATA_FILE_NAME, TRAIN_DIR_TDE

# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--UNIVERSE_MODEL", type=int, required=True, help="Universe model index")
parser.add_argument("--GALAXIES",      type=int, required=True)
parser.add_argument("--SURVEY",        type=str, default="ZTF", choices=["ZTF", "LSST"])
parser.add_argument("--Z_GAL_MIN",    type=float, default=0.0)
parser.add_argument("--Z_GAL_MAX",    type=float, default=10.0)
parser.add_argument("--batch_size",    type=int, default=4096)
parser.add_argument("--mbh_mass_method", type=str, required=True, default="use_sigma", choices=["use_sigma", "use_lgMgal"])
parser.add_argument("--nucleation_method", type=str, required=True, default="hannah", choices=["hannah", "neumayer"])
parser.add_argument("--occupation_method", type=str, required=True, default="uniform", choices=["uniform", "varying"])
parser.add_argument("--DETECTION_THRESHOLD", type=float, default=0.5,
                     help="Detection-probability threshold applied to the trained "
                          "classifier's output. Lower this (e.g. 0.2-0.3) to recover "
                          "more events in low-rate scenarios -- this does NOT require "
                          "retraining, it only moves the cut on already-computed "
                          "probabilities.")
args = parser.parse_args()

DETECTION_THRESHOLD = args.DETECTION_THRESHOLD

MODEL_PATH      = f"{TRAIN_DIR_TDE}/1E6_events/TDE_DETECT_BCE_no_rescaler_512_4_10_{args.SURVEY}/best_final_model/model/model.pth"
TRAIN_H5_PATH   = f"{TRAIN_DIR_TDE}/1E6_events/TDE_params_1E6_{args.SURVEY}.h5"
TRAIN_NPY_PATH  = f"{TRAIN_DIR_TDE}/1E6_events/TDE_params_scaled_1E6_{args.SURVEY}.npy"

DATA_PATH = f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}/{TDE_DATA_FILE_NAME}_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}_{args.UNIVERSE_MODEL}.h5"
OUT_PATH  = f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}/TDE_SNRs_{args.SURVEY}_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}_{args.UNIVERSE_MODEL}.h5"


PARAM_ORDER = ['alpha', 'beta', 'dec_rad', 'eta', 'lgMBH', 'ra_rad', 't0_mjd_transient', 'z_gal']


SCALING = {}
with h5py.File(TRAIN_H5_PATH, "r") as f:
    for k in PARAM_ORDER:
        data = f[k][:]
        SCALING[k] = ('linear', data.min(), data.max(), 0.0, 1.0)

print(SCALING)

# Map: training key → catalog key
KEY_MAP = {
    'alpha':   'alpha',
    'beta':    'beta',
    'dec_rad': 'dec_rad',
    'eta':     'eta',
    'lgMBH':   'lgMBH_mass',
    'ra_rad':  'ra_rad',
    't0_mjd_transient': 't0_mjd_transient',
    'z_gal':   'z_gal',
}

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
    "ZTF":  (58178, 61396),   # 1 Mar 2018 → Dec 2026
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
    exit()
    
# ─────────────────────────────────────────────────────────────
# Build scaled input matrix
# ─────────────────────────────────────────────────────────────
X = np.zeros((n_events, len(PARAM_ORDER)), dtype=np.float32)

for col, train_key in enumerate(PARAM_ORDER):
    cat_key = KEY_MAP[train_key]
    raw = catalog[cat_key].astype(float)

    scale_type, x_min, x_max, t_min, t_max = SCALING[train_key]
    vals = np.log10(raw) if scale_type == 'log10' else raw

    if scale_type == 'log10':
        x_min, x_max = np.log10(x_min), np.log10(x_max)

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
        probs  = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
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
    f.attrs["n_events_total"] = n_events
    f.attrs["n_events_passed"] = len(events_idx)
    f.attrs["detection_threshold"] = DETECTION_THRESHOLD
    f.attrs["param_order"]    = str(PARAM_ORDER)
    f.create_dataset("detection_prob", data=detection_prob[events_idx], compression="gzip")
    for k in ["z_gal", "lgMBH_mass", "distance_Gpc", "eta", "alpha", "beta", "star_mass",
              "initial_MBHspin", "ra_rad", "dec_rad", "t0_mjd_transient"]:
        if k in catalog:
            f.create_dataset(k, data=catalog[k][events_idx], compression="gzip")

print(f"Saved {len(events_idx)}/{n_events} detection-passing TDE rows to {OUT_PATH}")


import matplotlib.pyplot as plt

if n_events > 0:
    finite = np.isfinite(detection_prob)
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(catalog['lgMBH_mass'][finite], detection_prob[finite],
                      c=catalog['z_gal'][finite], cmap='viridis', s=8)
    plt.colorbar(sc, label='z')
    plt.axhline(DETECTION_THRESHOLD, color='red', ls='--', lw=1,
                label=f'threshold={DETECTION_THRESHOLD:.2f}')
    plt.xlabel(r'$\log_{10}(M_{\rm MBH}/M_\odot)$')
    plt.ylabel('Detection probability')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}/TDE_detprob_MBHmass_{args.SURVEY}_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}_{args.UNIVERSE_MODEL}.png")
    plt.close()

# plot cumulative plot for detections with t0_mjd_transient
t = catalog["t0_mjd_transient"] if n_events > 0 else np.array([])
mask = detection_prob > DETECTION_THRESHOLD if n_events > 0 else np.array([], dtype=bool)

t_detected = t[mask]

if len(t_detected) > 0:
    year = 2000 + (t_detected - 51544.5) / 365.25

    year_min = int(np.floor(year.min()))
    year_max = int(np.ceil(year.max()))
    bins = np.arange(year_min, year_max + 1)

    counts, edges = np.histogram(year, bins=bins)
    cumulative_counts = np.cumsum(counts)

    plt.figure(figsize=(8, 6))
    plt.step(edges[:-1], cumulative_counts, where='mid')
    plt.xlabel("Year")
    plt.ylabel("Cumulative detections")
    plt.title(f"{args.SURVEY} cumulative detections")
    plt.grid()

    plt.savefig(f"{ASTROPHYSICAL_DATA_DIR}/cumulative_detections_{args.SURVEY}_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}_{args.UNIVERSE_MODEL}.png")
else:
    print("[INFO] No detected TDEs -- skipping cumulative-detections plot.")