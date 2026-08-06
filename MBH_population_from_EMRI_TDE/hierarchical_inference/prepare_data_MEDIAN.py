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
parser.add_argument("--mbh_mass_method", type=str, required=False, default="M_sigma", choices=["use_sigma", "use_lgMgal"])
parser.add_argument("--nucleation_method", type=str, required=False, default="hannah", choices=["hannah", "neumayer"])
args = parser.parse_args()

OUT_DIR = f"{INFERENCE_DATA_DIR}/{args.GALAXIES}"
os.makedirs(OUT_DIR, exist_ok=True)

EMRI_SNR_PATH = f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}/max_detected_EMRI_SNRs_data_{args.mbh_mass_method}_{args.nucleation_method}.h5"
TDE_SNR_PATH_ZTF = f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}/max_detected_TDE_SNRs_ZTF_{args.mbh_mass_method}_{args.nucleation_method}.h5"
TDE_SNR_PATH_LSST = f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}/max_detected_TDE_SNRs_LSST_{args.mbh_mass_method}_{args.nucleation_method}.h5"

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
     h5py.File(f"{OUT_DIR}/true_detected_data_EMRI_{args.mbh_mass_method}_{args.nucleation_method}.h5", "w") as hf_out:
    for key in hf.keys():
        arr = hf[key][()]
        print(f"  {key:25s}  shape={arr.shape}")
        if key in EMRI_RENAME:
            new_key, offset = EMRI_RENAME[key]
            emri_data[new_key] = (arr + offset).astype(np.float32)
            hf_out.create_dataset(new_key, data=emri_data[new_key])

torch.save(
    {k: torch.from_numpy(v) for k, v in emri_data.items()},
    f"{OUT_DIR}/true_detected_data_EMRI_{args.mbh_mass_method}_{args.nucleation_method}.pt",
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

    tde_path = TDE_SNR_PATH_ZTF if survey == "ZTF" else TDE_SNR_PATH_LSST
    tde_data = {}

    with h5py.File(tde_path, "r") as hf, \
         h5py.File(f"{OUT_DIR}/true_detected_data_TDE_{survey}_{args.mbh_mass_method}_{args.nucleation_method}.h5", "w") as hf_out:
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
        f"{OUT_DIR}/true_detected_data_TDE_{survey}_{args.mbh_mass_method}_{args.nucleation_method}.pt",
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

model = f"{args.mbh_mass_method}_{args.nucleation_method}"

Plotting.plot_comparison_histogram(key="lgMBH_mass", loc=OUT_DIR, x_label=r"$\log_{10}(M_{\mathrm{BH}}/M_\odot)$", model=model, **kwargs_all)
Plotting.plot_comparison_histogram(key="CO_mass", loc=OUT_DIR, x_label=r"$m_{\mathrm{CO}}\ [M_\odot]$", model=model, **kwargs_all)
Plotting.plot_comparison_histogram(key="z_gal", loc=OUT_DIR, x_label=r"$z$", model=model, **kwargs_all)

# EMRI-only keys
Plotting.plot_comparison_histogram(key="MBHspin", loc=OUT_DIR, x_label=r"$a$", model=model, EMRI=emri_data)
Plotting.plot_comparison_histogram(key="e_pl", loc=OUT_DIR, x_label=r"$e_{\mathrm{pl}}$", model=model, EMRI=emri_data)

print("\n[DONE]")