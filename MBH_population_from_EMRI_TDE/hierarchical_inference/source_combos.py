#!/usr/bin/env python
"""
Shared source/survey combination list + safe result loader, used by every
plotting script (1D posteriors, PPD, combined PPD+1D).

Adding, removing, or renaming a combination is a ONE-LINE edit to
SOURCE_SURVEY_COMBOS below -- no plotting script needs to change. If a
given combo's result.hdf5 is missing, unreadable, or empty, it is skipped
with a printed warning rather than crashing the whole plot -- one broken
condor job does not block the rest of the figure.
"""
import h5py
import numpy as np
import seaborn as sns

pal = sns.color_palette(palette="colorblind").as_hex()

# (source, survey, label, color)
# survey="None" for EMRI-only runs, matching the directory naming
# convention f"{SOURCE}_pop_{POPULATION}_{SURVEY}".
SOURCE_SURVEY_COMBOS = [
    ("EMRI",     "None", "EMRI",            pal[4]),
    ("TDE",      "ZTF",  "TDE (ZTF)",       pal[2]),
    ("TDE",      "LSST", "TDE (LSST)",      pal[0]),
    ("TDE",      "ALL",  "TDE (ALL)",       pal[6]),
    ("EMRI_TDE", "ZTF",  "EMRI+TDE (ZTF)",  pal[3]),
    ("EMRI_TDE", "LSST", "EMRI+TDE (LSST)", pal[1]),
    ("EMRI_TDE", "ALL",  "EMRI+TDE (ALL)",  pal[5]),
]


def result_path(base_dir: str, source: str, survey: str, population: str) -> str:
    """Path of the inference result for a source/survey/population."""
    return f"{base_dir}/{source}_pop_{population}_{survey}/result.hdf5"


def load_samples_dict(path: str) -> dict:
    """Read every field actually present in this run's posterior_samples
    table, instead of assuming a fixed schema. Raises on missing/malformed
    files -- caller decides whether to skip."""
    with h5py.File(path, "r") as hf:
        ds = hf["posterior_samples"]
        return {name: np.asarray(ds[name]) for name in ds.dtype.names}


def load_available_combos(base_dir: str, population: str,
                           combos=SOURCE_SURVEY_COMBOS) -> list:
    """Try to load every (source, survey) combo's result.hdf5. Any combo
    that fails to load is skipped with a warning -- it does NOT stop the
    other combos from loading, and does NOT require touching this
    function or any plotting script.

    Returns a list of dicts (only the combos that loaded successfully):
        {"source", "survey", "label", "color", "samples"}
    in the same order as `combos`.
    """
    available = []
    for source, survey, label, color in combos:
        path = result_path(base_dir, source, survey, population)
        try:
            samples = load_samples_dict(path)
        except (FileNotFoundError, OSError, KeyError) as exc:
            print(f"[skip] {label} (pop {population}): could not load {path} ({exc})")
            continue
        if not samples:
            print(f"[skip] {label} (pop {population}): {path} has no posterior_samples fields")
            continue
        available.append({
            "source": source, "survey": survey, "label": label,
            "color": color, "samples": samples,
        })
    if not available:
        print(f"[warn] No combos loaded at all for pop {population} under {base_dir}")
    return available