#!/usr/bin/env python
import argparse, os
import h5py
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES",        type=int,   required=True)
parser.add_argument("--SURVEY",          type=str,   default="ZTF", choices=["ZTF", "LSST"])
parser.add_argument("--is_clustering",   type=int,   default=0)
parser.add_argument("--N_SAMPLES",       type=int,   default=2000)
parser.add_argument("--DET_THRESHOLD",   type=float, default=0.5)
parser.add_argument("--SIGMA_MBH6",      type=float, default=0.1)
parser.add_argument("--SIGMA_MSTAR",     type=float, default=0.1)
parser.add_argument("--SIGMA_REDSHIFT",  type=float, default=0.01)
parser.add_argument("--SIGMA_ETA",       type=float, default=0.01)
parser.add_argument("--SIGMA_ALPHA",     type=float, default=0.05)
parser.add_argument("--SIGMA_BETA",      type=float, default=0.05)
parser.add_argument("--SIGMA_T0",        type=float, default=1.0)
parser.add_argument("--PLOT_FISHER",     action="store_true")
parser.add_argument("--PLOT_CORNER",     action="store_true")
parser.add_argument("--PLOT_COVARIANCE", action="store_true")
parser.add_argument("--PLOT_HISTOGRAMS", action="store_true")
args = parser.parse_args()

BASE   = "./data"
suffix = "clustering" if args.is_clustering else "no_clustering"
SURVEY = args.SURVEY.upper()

DETPROB_PATH = f"{BASE}/astrophysical_data/{args.GALAXIES}/TDE_detection_prob_{SURVEY}_{suffix}.h5"
OUT_DIR      = f"{BASE}/fisher_data/{args.GALAXIES}"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_H5 = os.path.join(OUT_DIR, f"true_data_TDE_{SURVEY}_{suffix}.h5")
OUT_PT = os.path.join(OUT_DIR, f"true_data_TDE_{SURVEY}_{suffix}.pt")

SIGMAS = {
    "mbh_6"            : args.SIGMA_MBH6,
    "stellar_mass"     : args.SIGMA_MSTAR,
    "redshift"         : args.SIGMA_REDSHIFT,
    "eta"              : args.SIGMA_ETA,
    "alpha"            : args.SIGMA_ALPHA,
    "beta"             : args.SIGMA_BETA,
    "t0_mjd_transient" : args.SIGMA_T0,
}

# ── Load detection prob file and keep p > threshold ───────────────────────
print(f"\n[INFO] Loading: {DETPROB_PATH}")
with h5py.File(DETPROB_PATH, "r") as hf:
    det_prob = np.asarray(hf["detection_prob"][()], dtype=np.float32)
    catalog  = {k: np.asarray(hf[k][()], dtype=np.float32)
                for k in hf.keys() if k != "detection_prob"}

mask    = det_prob > args.DET_THRESHOLD
catalog = {k: v[mask] for k, v in catalog.items()}
n       = mask.sum()
print(f"  p > {args.DET_THRESHOLD}: {n} / {len(det_prob)} events")

# ── True parameter values (key rename + mbh_6 offset) ────────────────────
true_vals = {
    "mbh_6"            : catalog["lgMBH_mass"] - 6.0,
    "stellar_mass"     : catalog["z_gal"] * 0 + 1.0,
    "redshift"         : catalog["z_gal"],
    "eta"              : catalog["eta"],
    "alpha"            : catalog["alpha"],
    "beta"             : catalog["beta"],
    "t0_mjd_transient" : catalog["t0_mjd_transient"],
}

# ── Gaussians ────────────────────────────────────────────────────────────
posteriors = {
    k: np.random.normal(v[:, None], SIGMAS[k], (n, args.N_SAMPLES)).astype(np.float32)
    for k, v in true_vals.items()
}

# ── Save ─────────────────────────────────────────────────────────────────
with h5py.File(OUT_H5, "w") as hf:
    hf.attrs.update({"survey": SURVEY, "n_events": int(n), "n_samples": args.N_SAMPLES})
    for k, v in true_vals.items():
        hf.create_dataset(k, data=v, compression="gzip")
    for k, v in posteriors.items():
        hf.create_dataset(f"{k}_posterior", data=v, compression="gzip")
print(f"[SAVED] {OUT_H5}")

torch.save({k: torch.from_numpy(v) for k, v in posteriors.items()}, OUT_PT)
print(f"[SAVED] {OUT_PT}")

print(f"\n[DONE]  {n} events, {args.N_SAMPLES} posterior samples each")
for k, v in posteriors.items():
    print(f"  {k:22s}  {v.shape}")