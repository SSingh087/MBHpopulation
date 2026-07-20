#!/usr/bin/env python
import sys, os
sys.path.insert(0, os.path.abspath('./'))

import numpy as np
import torch
import h5py
import argparse
from utils import Plotting, BASE_DIR, ASTROPHYSICAL_DATA_DIR, INFERENCE_DATA_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES",      type=int, required=True)
parser.add_argument("--is_clustering", type=int, default=0)
args = parser.parse_args()

OUT_DIR = f"{INFERENCE_DATA_DIR}/{args.GALAXIES}"
os.makedirs(OUT_DIR, exist_ok=True)

EMRI_SNR_PATH = f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}/EMRI_SNRs_data.h5"
TDE_SNR_PATH_ZTF = f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}/TDE_SNRs_ZTF.h5"
TDE_SNR_PATH_LSST = f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}/TDE_SNRs_LSST.h5"

# ── Key rename maps ───────────────────────────────────────────────────────
# EMRI: SNR file already uses internal names, except sBH_mass -> CO_mass
# and eccentricities_at_plunge -> e_pl.
EMRI_RENAME = {
    "lgMBH_mass"                : ("lgMBH_mass", 0.0),
    "MBHspin"                   : ("MBHspin",    0.0),
    "sBH_mass"                  : ("CO_mass",    0.0),
    "z_gal"                     : ("z_gal",      0.0),
    "eccentricities_at_plunge"  : ("e_pl",       0.0),
}

# TDE: SNR file uses star_mass (CO mass) and its own lgMBH_mass/z_gal directly.
TDE_RENAME = {
    "lgMBH_mass": ("lgMBH_mass", 0.0),
    "star_mass" : ("CO_mass",    0.0),
    "z_gal"     : ("z_gal",      0.0),
}

# ══════════════════════════════════════════════════════════════════════════
# EMRI
# ══════════════════════════════════════════════════════════════════════════
print("\n=======================")
print("EMRI data:")
print("=======================")

emri_data = {}

with h5py.File(EMRI_SNR_PATH, "r") as hf, \
     h5py.File(f"{OUT_DIR}/true_detected_data_EMRI.h5", "w") as hf_out:
    for key in hf.keys():
        arr = hf[key][()]
        print(f"  {key:25s}  shape={arr.shape}")
        if key in EMRI_RENAME:
            new_key, offset = EMRI_RENAME[key]
            emri_data[new_key] = (arr + offset).astype(np.float32)
            hf_out.create_dataset(new_key, data=emri_data[new_key])

torch.save(
    {k: torch.from_numpy(v) for k, v in emri_data.items()},
    f"{OUT_DIR}/true_detected_data_EMRI.pt",
)
print(f"[SAVED] EMRI  → {OUT_DIR}")

# ══════════════════════════════════════════════════════════════════════════
# TDE  (ZTF + LSST)
# ══════════════════════════════════════════════════════════════════════════
tde_by_survey = {}

for survey in ["ZTF", "LSST"]:
    print(f"\n=======================")
    print(f"TDE {survey} data:")
    print(f"=======================")

    tde_path = f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}/TDE_SNRs_{survey}.h5"
    tde_data = {}

    with h5py.File(tde_path, "r") as hf, \
         h5py.File(f"{OUT_DIR}/true_detected_data_TDE_{survey}.h5", "w") as hf_out:
        for key in hf.keys():
            arr = hf[key][()]
            print(f"  {key:25s}  shape={arr.shape}")
            if key in TDE_RENAME:
                new_key, offset = TDE_RENAME[key]
                tde_data[new_key] = (arr + offset).astype(np.float32)
                hf_out.create_dataset(new_key, data=tde_data[new_key])

    tde_by_survey[survey] = tde_data
    torch.save(
        {k: torch.from_numpy(v) for k, v in tde_data.items()},
        f"{OUT_DIR}/true_detected_data_TDE_{survey}.pt",
    )
    print(f"[SAVED] TDE {survey} → {OUT_DIR}")

# ══════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════
kwargs_all = {
    "EMRI"           : emri_data,
    f"TDE (ZTF)": tde_by_survey["ZTF"],
    f"TDE (LSST)": tde_by_survey["LSST"],
}

Plotting.plot_comparison_histogram(key="lgMBH_mass", loc=OUT_DIR, x_label=r"$\log_{10}(M_{\mathrm{BH}}/M_\odot)$", **kwargs_all)
Plotting.plot_comparison_histogram(key="CO_mass", loc=OUT_DIR, x_label=r"$m_{\mathrm{CO}}\ [M_\odot]$", **kwargs_all)
Plotting.plot_comparison_histogram(key="z_gal", loc=OUT_DIR, x_label=r"$z$", **kwargs_all)

# EMRI-only keys
Plotting.plot_comparison_histogram(key="MBHspin", loc=OUT_DIR, x_label=r"$a$", EMRI=emri_data)
Plotting.plot_comparison_histogram(key="e_pl", loc=OUT_DIR, x_label=r"$e_{\mathrm{pl}}$", EMRI=emri_data)

print("\n[DONE]")