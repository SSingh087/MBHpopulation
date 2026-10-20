#!/usr/bin/env python
"""
generate_emri_fisher_data.py

Reads the FLAT EMRI parameter file produced by generate_EMRI_params.py
(all datasets at HDF5 root level, one row per event), wraps Gaussian mock
posteriors around every true value, and writes the flat HDF5 + PT files
expected by prepare_data_with_posteriors.py.

Input  (flat root-level datasets from doc-5 pipeline):
    astrophysical_data/{GALAXIES}/all_galaxies_EMRI_events_{suffix}.h5
        lgMBH_mass, MBHspin, z_gal, eccentricities_at_plunge,
        e0, p0, qS, phiS, Y0, qK, phiK, Phi_phi0, Phi_theta0,
        Phi_r0, T_SIGNAL_duration_years, distance_Gpc, sBH_mass

Output (flat, ready for inference):
    fisher_data/{GALAXIES}/true_data_EMRI_{suffix}.h5
        true values  +  {key}_posterior arrays  (n_events × N_SAMPLES)
    fisher_data/{GALAXIES}/true_data_EMRI_{suffix}.pt
        posterior tensors only, internal keys

Usage:
    python generate_emri_fisher_data.py \
        --GALAXIES 100000 \
        --is_clustering 0 \
        --OBSERVING_WINDOW 4.0 \
        --N_SAMPLES 2000 \
        --SIGMA 1e-3 \
        --CO_MASS 10.0
"""

import argparse
import os
import h5py
import numpy as np
import torch

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES",         type=int,   required=True,
                    help="Catalog size label (used for directory paths)")
parser.add_argument("--OBSERVING_WINDOW", type=float, required=True,
                    help="LISA observing window in years (stored as metadata)")
parser.add_argument("--is_clustering",    type=int,   default=0,
                    help="1 = clustered catalog, 0 = no clustering")
parser.add_argument("--N_SAMPLES",        type=int,   default=2000,
                    help="Posterior samples per event")
parser.add_argument("--SIGMA",            type=float, default=1e-3,
                    help="Gaussian σ for EMRI posteriors (parameter units)")
parser.add_argument("--CO_MASS",          type=float, default=10.0,
                    help="Fixed compact-object mass [Msun] injected for all events")
# plot flags — reserved, no-op here
parser.add_argument("--PLOT_FISHER",     action="store_true")
parser.add_argument("--PLOT_CORNER",     action="store_true")
parser.add_argument("--PLOT_COVARIANCE", action="store_true")
parser.add_argument("--PLOT_HISTOGRAMS", action="store_true")
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────
BASE    = "./data"
suffix  = "clustering" if args.is_clustering else "no_clustering"

# Input: flat file produced by generate_EMRI_params.py (doc 5)
CATALOG_PATH = (f"{BASE}/astrophysical_data/{args.GALAXIES}/"
                f"all_galaxies_EMRI_events_{suffix}.h5")

# Output: fisher_data dir consumed by prepare_data_with_posteriors.py
OUT_DIR = f"{BASE}/fisher_data/{args.GALAXIES}"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_H5 = os.path.join(OUT_DIR, f"true_data_EMRI_{suffix}.h5")
OUT_PT = os.path.join(OUT_DIR, f"true_data_EMRI_{suffix}.pt")

# ── Key definitions ───────────────────────────────────────────────────────
# Keys to READ from the flat catalog (names as saved by doc 5 / generate_EMRI_params.py)
# eccentricities_at_plunge is the doc-5 name; we rename it to e_pl on load.
CATALOG_KEYS = ["lgMBH_mass", "MBHspin", "z_gal", "eccentricities_at_plunge"]

# Internal key names expected by prepare_data_with_posteriors.py
# CO_mass is injected below — it is not stored in the catalog.
OUTPUT_KEYS = ["lgMBH_mass", "MBHspin", "CO_mass", "z_gal", "e_pl"]

# ── Helpers ───────────────────────────────────────────────────────────────
def make_posteriors(data: dict, n_samples: int, sigma: float) -> dict:
    """Gaussian posteriors centred on true values. Shape: (n_events, n_samples)."""
    return {
        k: np.random.normal(v[:, None], sigma, (len(v), n_samples)).astype(np.float32)
        for k, v in data.items()
    }


def save_h5(path: str, datasets: dict, label: str = "", **attrs):
    """Save the datasets to an HDF5 file."""
    with h5py.File(path, "w") as hf:
        for key, arr in datasets.items():
            hf.create_dataset(key, data=arr, compression="gzip")
        for attr_key, attr_val in attrs.items():
            hf.attrs[attr_key] = attr_val
    print(f"[SAVED] {label:50s} → {path}")


def save_pt(path: str, datasets: dict, label: str = ""):
    """Save the datasets as torch tensors."""
    torch.save(
        {k: torch.from_numpy(np.asarray(arr, dtype=np.float32))
         for k, arr in datasets.items()},
        path,
    )
    print(f"[SAVED] {label:50s} → {path}")


# ══════════════════════════════════════════════════════════════════════════
# 1.  Load FLAT catalog
#     The file has all datasets at the HDF5 root (no sub-groups).
#     Iterating hf gives dataset names, not galaxy IDs.
# ══════════════════════════════════════════════════════════════════════════
print(f"\n[INFO] Loading flat EMRI catalog:\n       {CATALOG_PATH}")
with h5py.File(CATALOG_PATH, "r") as hf:
    available = list(hf.keys())
    print(f"  Datasets in file : {available}")

    missing = [k for k in CATALOG_KEYS if k not in hf]
    if missing:
        raise KeyError(
            f"Expected keys {missing} not found.\n"
            f"Available: {available}"
        )

    raw: dict[str, np.ndarray] = {
        k: np.asarray(hf[k][()], dtype=np.float32) for k in CATALOG_KEYS
    }

n_events = len(raw["lgMBH_mass"])
print(f"  Events loaded    : {n_events}")
for k, v in raw.items():
    print(f"  {k:30s}  min={v.min():.4f}  max={v.max():.4f}")


# ── Rename eccentricities_at_plunge → e_pl ────────────────────────────────
emri_data: dict[str, np.ndarray] = {
    "lgMBH_mass" : raw["lgMBH_mass"],
    "MBHspin"    : raw["MBHspin"],
    "z_gal"      : raw["z_gal"],
    "e_pl"       : raw["eccentricities_at_plunge"],   # ← rename here
}

# ── Inject CO_mass (fixed physical value, absent from catalog) ────────────
emri_data["CO_mass"] = np.full(n_events, args.CO_MASS, dtype=np.float32)
print(f"\n  CO_mass injected : {args.CO_MASS} Msun  (fixed for all {n_events} events)")


# ══════════════════════════════════════════════════════════════════════════
# 2.  Build Gaussian posteriors around every true value
# ══════════════════════════════════════════════════════════════════════════
print(f"\n[INFO] Building posteriors  "
      f"(N_SAMPLES={args.N_SAMPLES}, σ={args.SIGMA:.1e}) ...")

emri_post = make_posteriors(emri_data, args.N_SAMPLES, args.SIGMA)


# ══════════════════════════════════════════════════════════════════════════
# 3.  Save
# ══════════════════════════════════════════════════════════════════════════
# HDF5: true values + posterior arrays side by side
save_h5(
    OUT_H5,
    {**emri_data,
     **{f"{k}_posterior": emri_post[k] for k in OUTPUT_KEYS}},
    label="EMRI (H5 — true + posteriors)",
    n_events=int(n_events),
    n_samples=args.N_SAMPLES,
    sigma=args.SIGMA,
    observing_window_years=args.OBSERVING_WINDOW,
    is_clustering=int(args.is_clustering),
)

# PT: posterior tensors only with internal key names
save_pt(
    OUT_PT,
    emri_post,
    label="EMRI (PT — posteriors, internal keys)",
)


# ══════════════════════════════════════════════════════════════════════════
# 4.  Summary
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[SUMMARY]")
print(f"  Input catalog    : {CATALOG_PATH}")
print(f"  Suffix           : {suffix}")
print(f"  Events           : {n_events}")
print(f"  Observing window : {args.OBSERVING_WINDOW} yr  (metadata only)")
print(f"  N_SAMPLES        : {args.N_SAMPLES}")
print(f"  σ (all params)   : {args.SIGMA:.1e}")
print(f"  CO_mass          : {args.CO_MASS} Msun  (fixed)")
print(f"  Output dir       : {OUT_DIR}")
print("\n[POSTERIOR SHAPES]")
for k, v in emri_post.items():
    print(f"  {k:20s}  {v.shape}  →  (n_events, n_samples)")
print("=" * 60)
print("[DONE]")