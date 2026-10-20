"""
Shared config/helpers for the data_gen_realistic sweep.

Pipeline (see sweep_thin_TDE_ZTF.py / sweep_thin_TDE_LSST.py / sweep_thin_EMRI.py):

  1. ZTF TDE catalogs (data_generation/calculate_SNR_TDE.py output) are
     wildly over-predicted vs. the real 43-event Yao+23 ZTF sample. For each
     (scenario M1-M8, TDE lightcurve model) pair we compute ONE reduction
     ratio = 43 / mean_original_count_across_universes, then thin every
     universe's detected events with that same keep-probability, applied
     independently per redshift bin (so we get an auditable per-bin removed
     count), via a per-event binomial draw.
  2. That same (scenario, TDE model) ratio is reused, unchanged, to thin the
     corresponding LSST TDE catalogs (no real LSST catalog exists yet, so
     this is the best available proxy for "realistic").
  3. EMRI catalogs have no TDE-lightcurve-model axis, so each scenario's
     EMRI ratio is the mean of that scenario's 4 TDE-model ZTF ratios.
"""
import glob
import hashlib
import os
import re

import h5py
import numpy as np

ASTROPHYSICAL_DATA_DIR = "./data/astrophysical_data"
GALAXIES_DEFAULT = 100000
MERGER_MODEL_CHOICES = ["mundy17", "oleary21", "duan24", "baseline"]   # keep in sync with utils.MERGER_MODEL_CHOICES

TDE_MODELS = [
    "cooling_envelope",
    "stream_stream_tde",
    "tde_analytical",
    "tde_fallback",
]

# (mbh_mass_method, nucleation_method, occupation_method) -> M1..M8, matches
# data_generation/compare_TDE_detection.py's SCENARIO_NAME_MAP.
SCENARIOS = [
    ("use_lgMgal", "hannah",   "uniform"),
    ("use_sigma",  "hannah",   "uniform"),
    ("use_lgMgal", "neumayer", "uniform"),
    ("use_sigma",  "neumayer", "uniform"),
    ("use_lgMgal", "hannah",   "varying"),
    ("use_sigma",  "hannah",   "varying"),
    ("use_lgMgal", "neumayer", "varying"),
    ("use_sigma",  "neumayer", "varying"),
]
SCENARIO_NAME_MAP = {s: f"M{i+1}" for i, s in enumerate(SCENARIOS)}

N_REDSHIFT_BINS_DEFAULT = 20


# Galaxy merger-rate model whose universes these scripts process. Must be set (from --merger_model) before
# any path/tag helper below is used; it is inserted after the occupation method in every file name,
# matching data_generation/*.py and report_universes.py.
MERGER_MODEL = None


def set_merger_model(mm):
    global MERGER_MODEL
    MERGER_MODEL = mm


def scenario_tag(scenario):
    mbh, nuc, occ = scenario
    if MERGER_MODEL is None:
        raise RuntimeError("call common_realistic.set_merger_model(args.merger_model) first")
    return f"{mbh}_{nuc}_{occ}_{MERGER_MODEL}"


def load_real_ztf_count():
    """Number of real, observed ZTF TDEs (Yao+23) -- the thinning target."""
    path = f"{ASTROPHYSICAL_DATA_DIR}/real_tde_detections.npz"
    d = np.load(path, allow_pickle=True)
    return int(len(d["ztf_mjd"]))


def discover_universes(glob_pattern):
    """Return sorted universe indices for files matching glob_pattern, which
    must end in '..._{UNIVERSE_MODEL}.h5'."""
    paths = glob.glob(glob_pattern)
    idx = []
    for p in paths:
        m = re.search(r"_(\d+)\.h5$", os.path.basename(p))
        if m:
            idx.append(int(m.group(1)))
    return sorted(idx)


def tde_snr_path(astro_dir, galaxies, survey, scenario, tde_model, universe, realistic=False):
    tag = scenario_tag(scenario)
    fname = f"TDE_SNRs_{survey}_{tag}_{tde_model}_{universe}.h5"
    if realistic:
        return os.path.join(astro_dir, str(galaxies), "realistic", fname)
    return os.path.join(astro_dir, str(galaxies), fname)


def emri_snr_path(astro_dir, galaxies, scenario, universe, realistic=False):
    tag = scenario_tag(scenario)
    if realistic:
        return os.path.join(astro_dir, str(galaxies), "realistic",
                             f"EMRI_SNRs_data_realistic_{tag}_{universe}.h5")
    return os.path.join(astro_dir, str(galaxies), f"EMRI_SNRs_data_{tag}_{universe}.h5")


def load_h5_catalog(path):
    """Load every dataset + attrs from an h5 catalog file into plain dicts."""
    with h5py.File(path, "r") as f:
        data = {k: f[k][:] for k in f.keys()}
        attrs = dict(f.attrs)
    return data, attrs


def redshift_bin_edges(z, n_bins=N_REDSHIFT_BINS_DEFAULT):
    if len(z) == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    z_max = max(float(np.max(z)), 1e-6)
    return np.linspace(0.0, z_max, n_bins + 1)


def thin_by_redshift_bin(z, keep_prob, bin_edges, rng):
    """
    Per-event binomial thinning with the SAME keep_prob in every redshift
    bin (no shape assumption -- the real catalog has no per-event redshift
    to match a shape against, so uniform thinning is the least-assumption
    choice). Returns (keep_mask, per_bin_removed, per_bin_original).

    keep_prob == 1.0 short-circuits to "keep everything" (used when a
    scenario is already at/below the target -- no reduction applied).
    """
    n = len(z)
    keep_mask = np.ones(n, dtype=bool)
    bin_idx = np.clip(np.digitize(z, bin_edges) - 1, 0, len(bin_edges) - 2) if n > 0 else np.array([], dtype=int)

    n_bins = len(bin_edges) - 1
    per_bin_removed = np.zeros(n_bins, dtype=int)
    per_bin_original = np.zeros(n_bins, dtype=int)

    if n == 0:
        return keep_mask, per_bin_removed, per_bin_original

    for b in range(n_bins):
        in_bin = np.where(bin_idx == b)[0]
        per_bin_original[b] = len(in_bin)
        if len(in_bin) == 0:
            continue
        if keep_prob >= 1.0:
            continue
        draw = rng.random(len(in_bin))
        drop = in_bin[draw >= keep_prob]
        keep_mask[drop] = False
        per_bin_removed[b] = len(drop)

    return keep_mask, per_bin_removed, per_bin_original


def save_thinned_h5(out_path, data, attrs, keep_mask, extra_attrs=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        for k, v in attrs.items():
            f.attrs[k] = v
        if extra_attrs:
            for k, v in extra_attrs.items():
                f.attrs[k] = v
        f.attrs["n_events_passed"] = int(keep_mask.sum())
        for k, v in data.items():
            f.create_dataset(k, data=v[keep_mask], compression="gzip")


def rng_for(*parts):
    """Deterministic RNG seeded from a tuple of strings/ints -- reproducible
    thinning across re-runs of the same (scenario, model, universe, survey).
    Uses a stable hash (hashlib, not builtin hash()) since Python's builtin
    str hash is randomized per-process (PYTHONHASHSEED) and would give a
    different, non-reproducible thinning on every run otherwise."""
    key = "_".join(str(p) for p in parts).encode()
    seed = int(hashlib.sha256(key).hexdigest()[:8], 16)
    return np.random.default_rng(seed)
