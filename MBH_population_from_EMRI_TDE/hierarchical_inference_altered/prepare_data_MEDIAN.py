#!/usr/bin/env python
"""Prepare inference inputs from the median-detection (best) universe of a scenario."""
import sys, os, re
sys.path.insert(0, os.path.abspath('./'))

import numpy as np
import torch
import h5py
import argparse
from utils import Plotting, BASE_DIR, ASTROPHYSICAL_DATA_DIR, INFERENCE_DATA_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES",      type=int, required=True)
parser.add_argument("--is_clustering", type=int, default=0)
parser.add_argument("--mbh_mass_method", type=str, required=True, default="use_sigma", choices=["use_sigma", "use_lgMgal"])
parser.add_argument("--nucleation_method", type=str, required=True, default="hannah", choices=["hannah", "neumayer"])
parser.add_argument("--occupation_method", type=str, required=True, default="uniform", choices=["uniform", "varying"])
parser.add_argument("--best_universe_model", type=str, default="auto",
                     help="Winning UNIVERSE_MODEL to pull max_detected_ files from. Either an "
                          "explicit integer, or 'auto' (default) to resolve it at run time from "
                          "PLOTS/universe_report/best_universe_<tde_model>.txt, as written by "
                          "report_universes.py.")
parser.add_argument("--merger_model", type=str, default=None, choices=["mundy17","oleary21","duan24","baseline"],
                     help="Galaxy merger-rate model tag in the input file names / report folder. Omit for legacy untagged names.")
parser.add_argument("--tde_model", type=str, default="cooling_envelope")
args = parser.parse_args()

OUT_DIR = f"{INFERENCE_DATA_DIR}/{args.GALAXIES}"
os.makedirs(OUT_DIR, exist_ok=True)


MM_DIR = f"{args.merger_model}/" if args.merger_model else ""
MM_TAG = f"{args.merger_model}_" if args.merger_model else ""


def resolve_best_universe(galaxies, tde_model, best_universe_arg):
    """Either the explicit integer passed in, or (if 'auto'/empty) the
    winning UNIVERSE_MODEL for this TDE model, parsed out of
    report_universes.py's best_universe_<tde_model>.txt."""
    if best_universe_arg not in (None, "", "auto"):
        return int(best_universe_arg)

    winner_path = (f"{ASTROPHYSICAL_DATA_DIR}/{galaxies}/PLOTS/universe_report/{MM_DIR}"
                   f"best_universe_{tde_model}.txt")
    if not os.path.exists(winner_path):
        raise FileNotFoundError(
            f"--best_universe_model=auto but {winner_path} doesn't exist -- "
            f"REPORT_UNIVERSES must run (and produce this file) before this script."
        )
    text = open(winner_path).read()
    m = re.search(rf"for TDE model '{re.escape(tde_model)}':\s*(\d+)", text)
    if not m:
        raise RuntimeError(f"Could not parse winning universe for '{tde_model}' out of {winner_path}")
    return int(m.group(1))


best_universe_model = resolve_best_universe(args.GALAXIES, args.tde_model, args.best_universe_model)
print(f"[INFO] best_universe_model = {best_universe_model} "
      f"({'explicit' if args.best_universe_model not in (None, '', 'auto') else 'auto-resolved'} "
      f"for tde_model='{args.tde_model}')")

# Must match report_universes.py's max_detected_ naming exactly: it copies
# each source file with a "max_detected_<tde_model>_" prefix PREPENDED to
# the ORIGINAL basename -- so for the TDE files (whose basename already
# contains the model) the model ends up appearing twice.
EMRI_SNR_PATH = (f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}/max_detected_{args.tde_model}_"
                 f"EMRI_SNRs_data_{args.mbh_mass_method}_{args.nucleation_method}_"
                 f"{args.occupation_method}_{MM_TAG}{best_universe_model}.h5")
TDE_SNR_PATH_ZTF = (f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}/max_detected_{args.tde_model}_"
                    f"TDE_SNRs_ZTF_{args.mbh_mass_method}_{args.nucleation_method}_"
                    f"{args.occupation_method}_{MM_TAG}{args.tde_model}_{best_universe_model}.h5")
TDE_SNR_PATH_LSST = (f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}/max_detected_{args.tde_model}_"
                     f"TDE_SNRs_LSST_{args.mbh_mass_method}_{args.nucleation_method}_"
                     f"{args.occupation_method}_{MM_TAG}{args.tde_model}_{best_universe_model}.h5")

# ── Key rename maps ───────────────────────────────────────────────────────
# EMRI: SNR file already uses internal names, except sBH_mass -> CO_mass
# and eccentricities_at_plunge -> e_pl.
EMRI_RENAME = {
    "lgMBH_mass"                : ("lgMBH_mass", 0.0),
    "MBHspin"                   : ("MBHspin",    0.0),
    # "sBH_mass"                  : ("CO_mass",    0.0),
    "z_gal"                     : ("z_gal",      0.0),
    "eccentricities_at_plunge"  : ("e_pl",       0.0),
}

# TDE: SNR file uses star_mass (CO mass) and its own lgMBH_mass/z_gal directly.
TDE_RENAME = {
    "lgMBH_mass": ("lgMBH_mass", 0.0),
    # "star_mass" : ("CO_mass",    0.0),
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
     h5py.File(f"{OUT_DIR}/true_detected_data_EMRI_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}.h5", "w") as hf_out:
    for key in hf.keys():
        arr = hf[key][()]
        print(f"  {key:25s}  shape={arr.shape}")
        if key in EMRI_RENAME:
            new_key, offset = EMRI_RENAME[key]
            emri_data[new_key] = (arr + offset).astype(np.float32)
            hf_out.create_dataset(new_key, data=emri_data[new_key])

torch.save(
    {k: torch.from_numpy(v) for k, v in emri_data.items()},
    f"{OUT_DIR}/true_detected_data_EMRI_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}.pt",
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
         h5py.File(f"{OUT_DIR}/true_detected_data_TDE_{survey}_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}.h5", "w") as hf_out:
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
        f"{OUT_DIR}/true_detected_data_TDE_{survey}_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}.pt",
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

model = f"{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}"

Plotting.plot_comparison_histogram(key="lgMBH_mass", loc=OUT_DIR, x_label=r"$\log_{10}(M_{\mathrm{BH}}/M_\odot)$", model=model, **kwargs_all)
Plotting.plot_comparison_histogram(key="CO_mass", loc=OUT_DIR, x_label=r"$m_{\mathrm{CO}}\ [M_\odot]$", model=model, **kwargs_all)
Plotting.plot_comparison_histogram(key="z_gal", loc=OUT_DIR, x_label=r"$z$", model=model, **kwargs_all)

# EMRI-only keys
Plotting.plot_comparison_histogram(key="MBHspin", loc=OUT_DIR, x_label=r"$a$", model=model, EMRI=emri_data)
Plotting.plot_comparison_histogram(key="e_pl", loc=OUT_DIR, x_label=r"$e_{\mathrm{pl}}$", model=model, EMRI=emri_data)

print("\n[DONE]")