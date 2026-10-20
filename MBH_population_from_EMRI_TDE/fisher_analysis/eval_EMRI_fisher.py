"""Fisher analysis of a slice [START_IDX, END_IDX) of the detected EMRI catalogue; writes one batch file with posterior samples."""
import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.abspath('./'))
sys.path.insert(0, os.path.abspath('./fisher_analysis'))
sys.path.insert(0, os.path.abspath('./astrophysical_setup'))
from utils import INFERENCE_DATA_DIR
from cosmology import CosmologyModel
from fish_EMRI import (PARAMS, build_response, compute_fisher, covariance,
                       sample_truncated_gaussian, plunge_eccentricity)

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES", type=int, required=True)
parser.add_argument("--SCENARIO", type=str, required=True,
                    help="{mbh_mass_method}_{nucleation_method}_{occupation_method}")
parser.add_argument("--OBSERVING_WINDOW", type=float, required=True, help="LISA observing window [yr]")
parser.add_argument("--PSD_PATH", type=str, required=True, help="LISA PSD (.npy, shape (2, n_freq)) from training_EMRI_SNR/calculate_PSD.py")
parser.add_argument("--START_IDX", type=int, required=True)
parser.add_argument("--END_IDX", type=int, required=True)
parser.add_argument("--BATCH_ID", type=int, required=True)
parser.add_argument("--N_SAMPLES", type=int, default=500, help="posterior samples per event")
parser.add_argument("--DT", type=float, default=10.0)
parser.add_argument("--SEED", type=int, default=0)
args = parser.parse_args()

loc = f"{INFERENCE_DATA_DIR}/{args.GALAXIES}"
os.makedirs(f"{loc}/fisher_batches", exist_ok=True)

with h5py.File(f"{loc}/detected_catalog_EMRI_{args.SCENARIO}.h5", "r") as f:
    n_total = f["z_gal"].shape[0]
    sl = slice(args.START_IDX, min(args.END_IDX, n_total))
    catalog = {k: f[k][sl] for k in list(PARAMS) + ["sBH_mass", "z_gal", "T_SIGNAL_duration_years"] if k in f}
catalog["CO_mass"] = catalog.pop("sBH_mass")

import cupy as cp
from few.trajectory.inspiral import EMRIInspiral

psd = cp.asarray(np.load(args.PSD_PATH))
response = build_response(args.OBSERVING_WINDOW, args.DT)
traj = EMRIInspiral(func="pn5")
rng = np.random.default_rng(args.SEED + args.BATCH_ID)

cosmo = CosmologyModel()
_z_grid = np.linspace(1e-4, 10.0, 4000)
_dl_grid = cosmo.luminosity_distance(_z_grid, unit=__import__("astropy.units", fromlist=["Gpc"]).Gpc)


def redshift_from_distance(dl_gpc):
    """Redshift for luminosity distances [Gpc] by interpolating D_L(z)."""
    return np.interp(dl_gpc, _dl_grid, _z_grid)


n_events = len(catalog["z_gal"])
out = {k: [] for k in ["lgMBH_mass", "MBHspin", "z_gal", "e_pl"]}
errors, ok_idx = [], []

for i in range(n_events):
    theta = {p: float(catalog[p][i]) for p in PARAMS}
    theta["CO_mass"] = float(catalog["CO_mass"][i])
    try:
        F = compute_fisher(response, theta, psd, args.DT)
        C = covariance(F)
        truth = np.array([theta[p] for p in PARAMS])
        # Observed value = truth + noise draw; the posterior is centred on it (not on the truth)
        centre = rng.multivariate_normal(truth, C)
        s = sample_truncated_gaussian(centre, C, args.N_SAMPLES, rng)
        idx = {p: PARAMS.index(p) for p in PARAMS}
        out["lgMBH_mass"].append(s[:, idx["lgMBH_mass"]])
        out["MBHspin"].append(s[:, idx["MBHspin"]])
        out["z_gal"].append(redshift_from_distance(s[:, idx["distance_Gpc"]]))
        out["e_pl"].append(plunge_eccentricity(traj, s, theta["CO_mass"], catalog["T_SIGNAL_duration_years"][i], args.DT))
        errors.append(np.sqrt(np.diag(C)))
        ok_idx.append(args.START_IDX + i)
    except Exception as exc:
        print(f"[BATCH {args.BATCH_ID}] event {args.START_IDX + i} failed: {type(exc).__name__}: {exc}")

out_path = f"{loc}/fisher_batches/EMRI_{args.SCENARIO}_{args.BATCH_ID:05d}.h5"
with h5py.File(out_path, "w") as hf:
    hf.attrs["n_events_in_batch"] = n_events
    hf.create_dataset("event_index", data=np.array(ok_idx))
    hf.create_dataset("errors", data=np.array(errors))
    for k, v in out.items():
        hf.create_dataset(k, data=np.array(v), compression="gzip")
print(f"[BATCH {args.BATCH_ID}] wrote {len(ok_idx)}/{n_events} events -> {out_path}")
