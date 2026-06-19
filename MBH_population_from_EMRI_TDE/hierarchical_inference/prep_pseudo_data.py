#!/usr/bin/env python

import os, sys, argparse
sys.path.insert(0, os.path.abspath('../../poplar'))
from utils import get_true_x, make_dist
import h5py
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES", type=int, required=True)
parser.add_argument("--POPULATION",      type=str, default="A", choices=["A", "B", "MIX"])
args = parser.parse_args()

N_POSTERIOR      = 1000
SIGMA_POSTERIOR  = 1e-3
Z_MIN, Z_MAX     = 0.01, 2.0

# ── Population setup ──────────────────────────────────────────────────────
popdist = make_dist(args.POPULATION)
true_x  = get_true_x(args.POPULATION)

loc = f'./data/pseudo_inference_data/{args.GALAXIES}'
os.makedirs(loc, exist_ok=True)

print(f"[INFO] Population : {args.POPULATION}")
print(f"[INFO] Events     : {args.GALAXIES}")

# ── Draw catalogue ────────────────────────────────────────────────────────
catalogue = popdist.draw_samples(x=true_x, size=args.GALAXIES)
cat = {k: np.asarray(v, dtype=np.float32) for k, v in catalogue.items()}

# z_gal is not part of the population model — inject separately
# cat["z_gal"] = np.random.uniform(Z_MIN, Z_MAX, size=args.GALAXIES).astype(np.float32)

print("\n[INFO] Catalogue keys:")
for k, v in cat.items():
    print(f"  {k:20s}  shape={v.shape}  min={v.min():.3f}  max={v.max():.3f}")


# ── Posterior factory ─────────────────────────────────────────────────────
def make_posteriors(data: dict, n_samples: int, sigma: float) -> dict:
    """Gaussian mock posteriors centred on true values. Shape: (events, n_samples)."""
    return {
        k: np.random.normal(v[:, None], sigma, (len(v), n_samples)).astype(np.float32)
        for k, v in data.items()
    }

# ── HDF5 helper ───────────────────────────────────────────────────────────
def save_h5(path: str, datasets: dict, label: str = ""):
    with h5py.File(path, 'w') as hf:
        for key, arr in datasets.items():
            hf.create_dataset(key, data=arr)
    print(f"[SAVED] {label:35s} → {path}")


def save_pt(path: str, datasets: dict, label: str = ""):
    torch.save(
        {k: torch.from_numpy(arr) for k, arr in datasets.items()},
        path
    )
    print(f"[SAVED] {label:35s} → {path}")


# ══════════════════════════════════════════════════════════════════════════
# EMRI  —  lgMBH_mass, MBHspin, CO_mass, z_gal
# ══════════════════════════════════════════════════════════════════════════
emri_keys = ["lgMBH_mass", "MBHspin", "CO_mass", "z_gal", "e_pl"]

missing = [k for k in emri_keys if k not in cat]
if missing:
    raise KeyError(f"EMRI keys missing: {missing}\nAvailable: {list(cat.keys())}")

emri_data = {k: cat[k] for k in emri_keys}
emri_post = make_posteriors(emri_data, N_POSTERIOR, SIGMA_POSTERIOR)

save_h5(
    os.path.join(loc, f"true_data_EMRI_no_clustering_pseudo_data_{args.POPULATION}.h5"),
    {**emri_data, **{f"{k}_posterior": emri_post[k] for k in emri_keys}},
    label="EMRI"
)


# ══════════════════════════════════════════════════════════════════════════
# TDE  —  saved in raw fisher naming (mbh_6, stellar_mass, redshift)
#         so that prepare_data.py can read them without modification
# ══════════════════════════════════════════════════════════════════════════
tde_keys_internal = ["lgMBH_mass", "CO_mass", "z_gal"]

missing_tde = [k for k in tde_keys_internal if k not in cat]
if missing_tde:
    raise KeyError(f"TDE keys missing: {missing_tde}\nAvailable: {list(cat.keys())}")

def internal_to_raw(data: dict) -> dict:
    """Convert internal key names → raw fisher key names expected by prepare_data.py."""
    return {
        "mbh_6"       : data["lgMBH_mass"] - 6.0,
        "stellar_mass": data["CO_mass"],
        "redshift"    : data["z_gal"],
    }

tde_internal = {k: cat[k] for k in tde_keys_internal}
tde_raw      = internal_to_raw(tde_internal)
tde_post_raw      = make_posteriors(tde_raw,      N_POSTERIOR, SIGMA_POSTERIOR * 1E2)  # for h5 (raw keys)
tde_post_internal = make_posteriors(tde_internal, N_POSTERIOR, SIGMA_POSTERIOR * 1E2)  # for .pt (internal keys)

for survey in ["ZTF", "LSST"]:
    save_h5(
        os.path.join(loc, f"true_data_TDE_{survey}_no_clustering_pseudo_data_{args.POPULATION}.h5"),
        {**tde_raw, **{f"{k}_posterior": tde_post_raw[k] for k in tde_raw}},
        label=f"TDE {survey}"
    )
    save_pt(
        os.path.join(loc, f"true_data_TDE_{survey}_no_clustering_pseudo_data_{args.POPULATION}.pt"),
        tde_post_internal,   # ← internal keys: lgMBH_mass, CO_mass, z_gal
        label=f"TDE {survey} (PT)"
    )

save_pt(
    os.path.join(loc, f"true_data_EMRI_no_clustering_pseudo_data_{args.POPULATION}.pt"),
    emri_post,  # internal keys: lgMBH_mass, MBHspin, CO_mass, z_gal, e_pl
    label="EMRI (PT)"
)

# ── Summary ───────────────────────────────────────────────────────────────
print("\n[SUMMARY]")
print(f"  Population  : {args.POPULATION}")
print(f"  EMRI events : {args.GALAXIES}")
print(f"  TDE events  : {args.GALAXIES}  (ZTF and LSST identical, no selection yet)")
print(f"  Post. samps : {N_POSTERIOR} per event  (σ = {SIGMA_POSTERIOR * 1E2:.1e} for TDE, σ = {SIGMA_POSTERIOR:.1e} for EMRI)")
print(f"  Saved to    : {loc}")
print(f"\n[SHAPES]")
for k, v in emri_post.items():
    print(f"  EMRI  {k:20s}  {v.shape}  → (n_events={v.shape[0]}, n_samples={v.shape[1]})")
for k, v in tde_post_internal.items():
    print(f"  TDE   {k:20s}  {v.shape}  → (n_events={v.shape[0]}, n_samples={v.shape[1]})")
print("\n[DONE]")