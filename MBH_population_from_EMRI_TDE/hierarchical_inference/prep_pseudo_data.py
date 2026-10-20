#!/usr/bin/env python
"""Build the pseudo-observation data sets for inference: add measurement noise and save posterior samples per source and survey."""

import os, sys, argparse
sys.path.insert(0, os.path.abspath('../../poplar'))
from utils import get_true_x, make_dist
import h5py
import numpy as np
import torch
from scipy.stats import truncnorm

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES", type=int, required=True)
parser.add_argument("--POPULATION", type=str, default="A", choices=["A", "B", "MIX"])
parser.add_argument("--INCLUDE_LSST", type=int, default=0,
                     help="1 = also load/process TDE LSST data (requires "
                          "true_detected_data_TDE_LSST.h5 to exist). "
                          "0 = ZTF only.")
parser.add_argument("--mbh_mass_method", type=str, required=True, default="use_sigma", choices=["use_sigma", "use_lgMgal"])
parser.add_argument("--nucleation_method", type=str, required=True, default="hannah", choices=["hannah", "neumayer"])
parser.add_argument("--occupation_method", type=str, required=True, default="uniform", choices=["uniform", "varying"])
parser.add_argument("--SEED", type=int, default=42,
                     help="seed for the observation noise (obs = true + sigma*eps)")
parser.add_argument("--SIGMA_MAX_MASS", type=float, default=None,
                     help="cap on the per-event TDE lgMBH sigma [dex] (default: no cap)")
parser.add_argument("--SIGMA_MAX_Z", type=float, default=None,
                     help="cap on the per-event TDE redshift sigma (default: no cap)")
parser.add_argument("--SIGMA_FILE", type=str, default=None,
                     help="json from fisher_analysis/fit_TDE_sigma.py: Fisher-calibrated sigma(z) for the "
                          "TDE surveys. If given it replaces the linear sigma_ref*(1+z/z_ref) model.")
args = parser.parse_args()

N_POSTERIOR = 2000

# ── Population setup ──────────────────────────────────────────────────────
popdist = make_dist(args.POPULATION)
# FIX: get_true_x now requires the scenario (mbh_mass_method,
# nucleation_method) to pick the right per-scenario truth values -- this
# used to call get_true_x(args.POPULATION) with only one arg, which
# raises a TypeError against the updated utils.get_true_x signature.
true_x  = get_true_x(args.POPULATION, args.mbh_mass_method, args.nucleation_method, args.occupation_method)

BASE = "./data"
INFERENCE_DIR = f"{BASE}/inference_data/{args.GALAXIES}"

# Count number of detections from the files
with h5py.File(f"{INFERENCE_DIR}/true_detected_data_EMRI_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}.h5", "r") as hf:
    N_EMRI = len(hf["lgMBH_mass"])

with h5py.File(f"{INFERENCE_DIR}/true_detected_data_TDE_ZTF_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}.h5", "r") as hf:
    N_TDE_ZTF = len(hf["lgMBH_mass"])

N_TDE_LSST = None
if args.INCLUDE_LSST:
    with h5py.File(f"{INFERENCE_DIR}/true_detected_data_TDE_LSST_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}.h5", "r") as hf:
        N_TDE_LSST = len(hf["lgMBH_mass"])

print(f"[INFO] Population : {args.POPULATION}")
print(f"[INFO] Scenario   : {args.mbh_mass_method} / {args.nucleation_method} / {args.occupation_method}")
print(f"[INFO] EMRI events   : {N_EMRI}")
print(f"[INFO] TDE (ZTF) events : {N_TDE_ZTF}")
if args.INCLUDE_LSST:
    print(f"[INFO] TDE (LSST) events: {N_TDE_LSST}")


loc = f'./data/pseudo_inference_data/{args.GALAXIES}'
os.makedirs(loc, exist_ok=True)

# ── Draw catalogue ────────────────────────────────────────────────────────
catalogue_EMRI = popdist.draw_samples(x=true_x, size=N_EMRI)
cat_EMRI = {k: np.asarray(v, dtype=np.float32) for k, v in catalogue_EMRI.items()}

catalogue_TDE_ZTF = popdist.draw_samples(x=true_x, size=N_TDE_ZTF)
cat_TDE_ZTF = {k: np.asarray(v, dtype=np.float32) for k, v in catalogue_TDE_ZTF.items()}

cat_TDE_LSST = None
if args.INCLUDE_LSST:
    catalogue_TDE_LSST = popdist.draw_samples(x=true_x, size=N_TDE_LSST)
    cat_TDE_LSST = {k: np.asarray(v, dtype=np.float32) for k, v in catalogue_TDE_LSST.items()}


print("\n[INFO] Catalogue keys EMRI:")
for k, v in cat_EMRI.items():
    print(f"  {k:20s}  shape={v.shape}  min={v.min():.3f}  max={v.max():.3f}")

print("\n[INFO] Catalogue keys TDE (ZTF):")
for k, v in cat_TDE_ZTF.items():
    print(f"  {k:20s}  shape={v.shape}  min={v.min():.3f}  max={v.max():.3f}")

if args.INCLUDE_LSST:
    print("\n[INFO] Catalogue keys TDE (LSST):")
    for k, v in cat_TDE_LSST.items():
        print(f"  {k:20s}  shape={v.shape}  min={v.min():.3f}  max={v.max():.3f}")


# ══════════════════════════════════════════════════════════════════════════
# Per-parameter measurement uncertainties, redshift-dependent.
# ══════════════════════════════════════════════════════════════════════════
Z_REF_EMRI     = 1.0   # EMRIs (LISA) detectable to much higher z at fixed precision
Z_REF_TDE_ZTF  = 0.1   # TDEs (ZTF) degrade fast with distance -- shorter reference scale
Z_REF_TDE_LSST = 0.2   # PLACEHOLDER: LSST is deeper than ZTF, so probably a larger
                        # reference z than ZTF's -- not derived from real LSST
                        # sensitivity, tune once real LSST data/model exists.


def z_scaled_sigma(sigma_ref: float, z: np.ndarray, z_ref: float) -> np.ndarray:
    """Per-event sigma that grows linearly with redshift, anchored at z_ref.
    Same functional form used for every parameter of a given source --
    only sigma_ref (and z_ref, per source population) differs."""
    return sigma_ref * (1.0 + z / z_ref)


import json
SIGMA_TABLE = json.load(open(args.SIGMA_FILE)) if args.SIGMA_FILE else None
SIGMA_CAP = {"lgMBH_mass": args.SIGMA_MAX_MASS, "z_gal": args.SIGMA_MAX_Z}


def tde_sigma(survey: str, param: str, sigma_ref: float, z: np.ndarray, z_ref: float) -> np.ndarray:
    """TDE per-event sigma. Fisher-calibrated table (np.interp, clamped to the calibrated z range)
    if --SIGMA_FILE was given, otherwise the linear surrogate; then the optional cap."""
    if SIGMA_TABLE is not None and param in SIGMA_TABLE.get(survey, {}):
        t = SIGMA_TABLE[survey][param]
        sig = np.interp(z, t["z"], t["sigma"])
    else:
        sig = z_scaled_sigma(sigma_ref, z, z_ref)
    cap = SIGMA_CAP[param]
    return sig if cap is None else np.minimum(sig, cap)


z_emri = cat_EMRI["z_gal"]
z_tde_ztf = cat_TDE_ZTF["z_gal"]

SIGMA_EMRI = {
    "lgMBH_mass": z_scaled_sigma(1e-2, z_emri, Z_REF_EMRI),
    "MBHspin":    z_scaled_sigma(1e-2, z_emri, Z_REF_EMRI),
    "z_gal":      z_scaled_sigma(1e-2, z_emri, Z_REF_EMRI),
    "e_pl":       z_scaled_sigma(5e-2, z_emri, Z_REF_EMRI),
}

SIGMA_TDE_ZTF = {
    "lgMBH_mass": tde_sigma("ZTF", "lgMBH_mass", 3e-1, z_tde_ztf, Z_REF_TDE_ZTF),
    "z_gal":      tde_sigma("ZTF", "z_gal",      5e-3, z_tde_ztf, Z_REF_TDE_ZTF),
    # no MBHspin, no e_pl -- TDEs don't measure these
}

SIGMA_TDE_LSST = None
if args.INCLUDE_LSST:
    z_tde_lsst = cat_TDE_LSST["z_gal"]
    SIGMA_TDE_LSST = {
        "lgMBH_mass": tde_sigma("LSST", "lgMBH_mass", 3e-1, z_tde_lsst, Z_REF_TDE_LSST),
        "z_gal":      tde_sigma("LSST", "z_gal",      5e-3, z_tde_lsst, Z_REF_TDE_LSST),
    }

# ══════════════════════════════════════════════════════════════════════════
# Physical bounds for every parameter -- same units as `true_x` / catalogue.
# ══════════════════════════════════════════════════════════════════════════
PARAM_BOUNDS = {
    "lgMBH_mass": (4.0, 7.0),
    "MBHspin":    (0.01, 0.99),
    "z_gal":      (0.01, 8.0),
    "e_pl":       (0.01, 0.99),
}


def internal_to_raw_bounds(bounds: dict) -> dict:
    """Map internal-key bounds -> raw fisher-key bounds (mbh_6 = lgMBH_mass - 6),
    same values just relabeled/shifted where needed."""
    lo, hi = bounds["lgMBH_mass"]
    return {
        "mbh_6":        (lo - 6.0, hi - 6.0),
        "redshift":     bounds["z_gal"],
    }


BOUNDS_TDE_RAW = internal_to_raw_bounds(PARAM_BOUNDS)

SIGMA_TDE_ZTF_RAW = {
    "mbh_6":        SIGMA_TDE_ZTF["lgMBH_mass"],
    "redshift":     SIGMA_TDE_ZTF["z_gal"],
}

SIGMA_TDE_LSST_RAW = None
if args.INCLUDE_LSST:
    SIGMA_TDE_LSST_RAW = {
        "mbh_6":        SIGMA_TDE_LSST["lgMBH_mass"],
        "redshift":     SIGMA_TDE_LSST["z_gal"],
    }


# ── Observation noise ─────────────────────────────────────────────────────
# A real posterior is centred on the NOISY measurement, obs = true + sigma*eps, not on the
# truth. Centring on the truth means the data carry no noise while the hierarchical
# likelihood still smooths the model by sigma, which biases the hyperparameters (it drove
# lam_schechter to the prior edge for TDEs; see test_lambda_bias.py). Draw eps ONCE per
# event so the raw (.h5) and internal (.pt) posteriors share the same centres.
rng_obs = np.random.default_rng(args.SEED)


def add_obs_noise(data: dict, sigma_dict: dict) -> dict:
    """obs = true + sigma * N(0,1), one independent draw per event and parameter."""
    out = {}
    for k, v in data.items():
        v = np.atleast_1d(np.asarray(v, dtype=np.float64))
        sigma = np.broadcast_to(np.asarray(sigma_dict[k], dtype=np.float64), v.shape)
        out[k] = v + sigma * rng_obs.standard_normal(v.shape)
    return out


# ── Posterior factory ─────────────────────────────────────────────────────
def make_posteriors(data: dict, n_samples: int, sigma_dict: dict, bounds_dict: dict) -> dict:
    """Gaussian mock posteriors centred on the values in `data` (pass the NOISY
    observations from add_obs_noise, not the truth), truncated to [lo, hi].
    Shape: (events, n_samples).

    NOTE: this used to be np.random.normal(...) followed by
    np.clip(samples, lo, hi). Clipping AFTER sampling doesn't truncate the
    distribution -- it piles every sample that would have landed outside
    the box onto the boundary value exactly, manufacturing a delta-spike
    at the edge that has no physical meaning. For an event whose sigma is
    wide relative to the box (e.g. TDE mass sigma ~0.3 dex against a
    3-dex-wide box), that spike can carry >10% of that event's posterior
    mass. With hundreds of such events, the hyper-population fit "sees" a
    large, statistically loud excess of density sitting exactly at the
    wall and can only explain it by dragging its own shape parameter
    (xc/lam_schechter/lambda_M) to the same wall -- i.e. exactly the
    boundary-railing behaviour this replaced. truncnorm draws from the
    correctly truncated Gaussian instead, so probability mass that would
    fall outside the box is redistributed within it, not stacked on the edge.
    """
    missing_sigma  = [k for k in data if k not in sigma_dict]
    missing_bounds = [k for k in data if k not in bounds_dict]
    if missing_sigma:
        raise KeyError(f"No sigma specified for: {missing_sigma}")
    if missing_bounds:
        raise KeyError(f"No bounds specified for: {missing_bounds}")

    out = {}
    for k, v in data.items():
        lo, hi = bounds_dict[k]
        v = np.atleast_1d(np.asarray(v, dtype=np.float64))
        sigma = np.broadcast_to(np.asarray(sigma_dict[k], dtype=np.float64), v.shape)
        a = (lo - v) / sigma
        b = (hi - v) / sigma
        samples = truncnorm.rvs(
            a[:, None], b[:, None], loc=v[:, None], scale=sigma[:, None],
            size=(len(v), n_samples),
        )
        out[k] = samples.astype(np.float32)
    return out


# ── HDF5 / torch helpers ──────────────────────────────────────────────────
def save_h5(path: str, datasets: dict, label: str = ""):
    """Save the datasets to an HDF5 file."""
    with h5py.File(path, 'w') as hf:
        for key, arr in datasets.items():
            hf.create_dataset(key, data=arr)
    print(f"[SAVED] {label:35s} → {path}")


def save_pt(path: str, datasets: dict, label: str = ""):
    """Save the datasets as torch tensors."""
    torch.save(
        {k: torch.from_numpy(arr) for k, arr in datasets.items()},
        path
    )
    print(f"[SAVED] {label:35s} → {path}")


# ══════════════════════════════════════════════════════════════════════════
# EMRI  —  lgMBH_mass, MBHspin, z_gal, e_pl
# ══════════════════════════════════════════════════════════════════════════
emri_keys = ["lgMBH_mass", "MBHspin", "z_gal", "e_pl"]

missing = [k for k in emri_keys if k not in cat_EMRI]
if missing:
    raise KeyError(f"EMRI keys missing: {missing}\nAvailable: {list(cat_EMRI.keys())}")

emri_data = {k: cat_EMRI[k] for k in emri_keys}
emri_obs = add_obs_noise(emri_data, SIGMA_EMRI)
emri_post = make_posteriors(emri_obs, N_POSTERIOR, SIGMA_EMRI, PARAM_BOUNDS)

save_h5(
    os.path.join(loc, f"true_data_EMRI_pseudo_data_{args.POPULATION}_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}.h5"),
    {**emri_data, **{f"{k}_obs": emri_obs[k] for k in emri_keys},
     **{f"{k}_posterior": emri_post[k] for k in emri_keys}},
    label="EMRI"
)


# ══════════════════════════════════════════════════════════════════════════
# TDE  —  saved in raw fisher naming (mbh_6, stellar_mass, redshift)
#         so that prepare_data.py can read them without modification.
#         ZTF always runs; LSST only if --INCLUDE_LSST 1 and the file exists.
# ══════════════════════════════════════════════════════════════════════════
tde_keys_internal = ["lgMBH_mass", "z_gal"]

def internal_to_raw(data: dict) -> dict:
    """Convert internal key names → raw fisher key names expected by prepare_data.py."""
    return {
        "mbh_6"       : data["lgMBH_mass"] - 6.0,
        "redshift"    : data["z_gal"],
    }


def process_tde_survey(survey: str, cat: dict, sigma_internal: dict, sigma_raw: dict):
    """Add noise and build the posteriors for one TDE survey."""
    missing_tde = [k for k in tde_keys_internal if k not in cat]
    if missing_tde:
        raise KeyError(f"TDE {survey} keys missing: {missing_tde}\nAvailable: {list(cat.keys())}")

    internal = {k: cat[k] for k in tde_keys_internal}
    raw      = internal_to_raw(internal)
    obs_internal  = add_obs_noise(internal, sigma_internal)
    obs_raw       = internal_to_raw(obs_internal)          # same noisy centres as the .pt file
    post_raw      = make_posteriors(obs_raw,      N_POSTERIOR, sigma_raw,      BOUNDS_TDE_RAW)
    post_internal = make_posteriors(obs_internal, N_POSTERIOR, sigma_internal, PARAM_BOUNDS)

    save_h5(
        os.path.join(loc, f"true_data_TDE_{survey}_pseudo_data_{args.POPULATION}_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}.h5"),
        {**raw, **{f"{k}_obs": obs_raw[k] for k in raw},
         **{f"{k}_posterior": post_raw[k] for k in raw}},
        label=f"TDE {survey}"
    )
    save_pt(
        os.path.join(loc, f"true_data_TDE_{survey}_pseudo_data_{args.POPULATION}_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}.pt"),
        post_internal,
        label=f"TDE {survey} (PT)"
    )
    return post_internal


tde_post_internal_ztf = process_tde_survey("ZTF", cat_TDE_ZTF, SIGMA_TDE_ZTF, SIGMA_TDE_ZTF_RAW)

tde_post_internal_lsst = None
if args.INCLUDE_LSST:
    tde_post_internal_lsst = process_tde_survey("LSST", cat_TDE_LSST, SIGMA_TDE_LSST, SIGMA_TDE_LSST_RAW)

save_pt(
    os.path.join(loc, f"true_data_EMRI_pseudo_data_{args.POPULATION}_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}.pt"),
    emri_post,  # internal keys: lgMBH_mass, MBHspin, z_gal, e_pl
    label="EMRI (PT)"
)

# ── Summary ───────────────────────────────────────────────────────────────
print("\n[SUMMARY]")
print(f"  Post. samps      : {N_POSTERIOR} per event")
print(f"  Z_REF_EMRI       : {Z_REF_EMRI}")
print(f"  Z_REF_TDE_ZTF    : {Z_REF_TDE_ZTF}")
if args.INCLUDE_LSST:
    print(f"  Z_REF_TDE_LSST   : {Z_REF_TDE_LSST}  (placeholder, not physically derived)")
print(f"  Bounds           : {PARAM_BOUNDS}")
print(f"  Saved to         : {loc}")
# print(f"\n[SHAPES]")
# print("\n[INFO] EMRI shapes:")
# for k, v in emri_post.items():
#     print(f"  EMRI  {k:20s}  {v.shape}  → (n_events={v.shape[0]}, n_samples={v.shape[1] if v.ndim > 1 else 1})")
# print("\n[INFO] TDE (ZTF) shapes:")
# for k, v in tde_post_internal_ztf.items():
#     print(f"  TDE   {k:20s}  {v.shape}  → (n_events={v.shape[0]}, n_samples={v.shape[1] if v.ndim > 1 else 1})")
# if args.INCLUDE_LSST:
#     print("\n[INFO] TDE (LSST) shapes:")
#     for k, v in tde_post_internal_lsst.items():
#         print(f"  TDE   {k:20s}  {v.shape}  → (n_events={v.shape[0]}, n_samples={v.shape[1] if v.ndim > 1 else 1})")
# print("\n[DONE]")