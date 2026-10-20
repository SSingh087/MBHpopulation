#!/usr/bin/env python
"""Analytic LISA SNR estimate for the generated EMRI catalogue; flags events above --SNR_THRESHOLD."""

import numpy as np
import argparse
import h5py
import sys, os
sys.path.insert(0, os.path.abspath('./'))
sys.path.insert(0, os.path.abspath('./astrophysical_setup'))
from utils import ASTROPHYSICAL_DATA_DIR, EMRI_DATA_FILE_NAME, MERGER_MODEL_CHOICES

G = 6.674e-11          # m^3 kg^-1 s^-2
CLIGHT = 299792458.0    # m/s
MSUN = 1.98892e30       # kg
GPC = 3.0857e25         # m

parser = argparse.ArgumentParser(description="Vectorized rough EMRI SNR estimate.")
parser.add_argument("--UNIVERSE_MODEL", type=int, required=True, help="Universe model index")
parser.add_argument("--GALAXIES", type=int, required=True)
parser.add_argument("--mbh_mass_method", type=str, required=True, default="use_sigma", choices=["use_sigma", "use_lgMgal"])
parser.add_argument("--nucleation_method", type=str, required=True, default="hannah", choices=["hannah", "neumayer"])
parser.add_argument("--occupation_method", type=str, required=True, default="uniform", choices=["uniform", "varying"])
parser.add_argument("--merger_model", type=str, required=True, choices=MERGER_MODEL_CHOICES,
                    help="Galaxy major-merger-rate model (see astrophysical_setup/cosmology.py MERGER_MODELS)")
# FIX: was a hardcoded module-level constant (SNR_THRESHOLD = 20). This
# is a rough ANALYTIC SNR estimate, not an ML model -- lowering this
# threshold is a legitimate, zero-retraining way to recover more
# "detected" EMRIs in low-count scenarios (e.g. lgMgal/neumayer). Kept
# the same default (20) so existing runs are unaffected unless you pass
# a different value.
parser.add_argument("--SNR_THRESHOLD", type=float, default=20.0,
                     help="Detection SNR threshold. Lower this (e.g. 10-15) to recover "
                          "more events in low-rate scenarios -- no retraining needed, "
                          "this is an analytic formula, not a learned model.")
args = parser.parse_args()

SNR_THRESHOLD = args.SNR_THRESHOLD

in_path = (f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}/{EMRI_DATA_FILE_NAME}_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}_{args.merger_model}_{args.UNIVERSE_MODEL}.h5")

with h5py.File(in_path, "r") as f:
    data = {key: np.array(f[key][:]) for key in f.keys()}

M_arr = 10 ** data['lgMBH_mass'] * MSUN         # MBH mass, kg (source frame)
mu_arr = data['sBH_mass'] * MSUN                # CO mass, kg (source frame)
p0_arr = data['p0']                             # dimensionless (units of M)
e0_arr = data['e0']
z_arr = data['z_gal']
dist_m = data['distance_Gpc'] * GPC             # luminosity distance, m
n_events = len(M_arr)

valid = np.isfinite(p0_arr) & np.isfinite(e0_arr) & (dist_m > 0)
print(f"{valid.sum()}/{n_events} events have finite (p0, e0, D_L); "
      f"rest will get SNR = NaN.")
if n_events > 0 and valid.sum() < n_events:
    print(f"NOTE: {n_events - valid.sum()}/{n_events} events have non-finite p0 -- "
          f"these are almost certainly the backward-integration fallback rows "
          f"(see prepare_EMRIs.py's n_fallback warning). They can NEVER pass the "
          f"SNR cut regardless of --SNR_THRESHOLD, since their SNR is NaN.")

# ============================================================
# GW frequency from semi-latus rectum p0, SOURCE FRAME
# (Keplerian estimate: Omega ~ M^{1/2} / (p0*M)^{3/2}, f_gw = 2*f_orb)
# ============================================================
Mtot_geom_time = G * M_arr / CLIGHT ** 3
f_orb_source = 1.0 / (2 * np.pi * Mtot_geom_time * p0_arr ** 1.5)
f_gw_source = 2.0 * f_orb_source

# Redshift to detector frame -- matters a lot given z up to ~8 in this catalog
f_gw = f_gw_source / (1.0 + z_arr)

# ============================================================
# Chirp mass, SOURCE FRAME, then redshifted to detector frame.
# ============================================================
Mchirp_source = mu_arr ** (3.0 / 5.0) * M_arr ** (2.0 / 5.0)
Mchirp_det = Mchirp_source * (1.0 + z_arr)

# ============================================================
# Sky/polarization-averaged characteristic strain, standard formula
# (Moore, Cole & Berry 2015, Eq. 20).
# ============================================================
Mchirp_geom_time = G * Mchirp_det / CLIGHT ** 3   # seconds

h_c = (np.sqrt(2.0 / 3.0) * np.pi ** (-2.0 / 3.0)
       * Mchirp_geom_time ** (5.0 / 6.0)
       * (CLIGHT / dist_m)
       * f_gw ** (-1.0 / 6.0))

# ============================================================
# Analytic LISA PSD (Robson, Cornish & Liu 2019, ~4yr confusion fit)
# ============================================================
def lisa_psd(f):
    """Analytic LISA noise power spectral density at frequency f."""
    f = np.asarray(f)
    L = 2.5e9
    fstar = CLIGHT / (2 * np.pi * L)

    P_oms = (1.5e-11) ** 2 * (1 + (2e-3 / f) ** 4)
    P_acc = (3e-15) ** 2 * (1 + (0.4e-3 / f) ** 2) * (1 + (f / 8e-3) ** 4)

    Pn = (P_oms + 2 * (1 + np.cos(f / fstar) ** 2) * P_acc / (2 * np.pi * f) ** 4) / L ** 2
    Sn_instrument = Pn * (1 + 6 / 10 * (f / fstar) ** 2)

    A = 9e-45
    alpha, beta, kappa, gamma, fk = 0.138, -221.0, 521.0, 1680.0, 0.00113
    Sc = A * f ** (-7. / 3.) * np.exp(-f ** alpha + beta * f * np.sin(kappa * f)) \
         * (1 + np.tanh(gamma * (fk - f)))

    return Sn_instrument + Sc

Sn = lisa_psd(f_gw)

# ============================================================
# Rough SNR: rho ~ h_c / sqrt(f * Sn(f))
# ============================================================
snr_rough = h_c / np.sqrt(f_gw * Sn)
snr_rough = np.where(valid, snr_rough, np.nan)

n_finite = np.sum(np.isfinite(snr_rough))
n_above_thresh = np.sum(snr_rough[np.isfinite(snr_rough)] > SNR_THRESHOLD)
print(f"{n_finite}/{n_events} SNRs computed.")
print(f"{n_above_thresh}/{n_finite} events above rho_t = {SNR_THRESHOLD} ")

# ============================================================
# Select events passing the SNR threshold
# ============================================================
events_idx = np.where(snr_rough > SNR_THRESHOLD)[0]
print(f"{len(events_idx)}/{n_events} events pass SNR threshold rho_t = {SNR_THRESHOLD}")

if len(events_idx) == 0:
    print("WARNING: no events pass the SNR threshold -- writing an empty file "
          "(downstream scripts should handle n_events=0 gracefully, or check "
          "this print statement before running them).")
    if n_finite > 0:
        print(f"  (For reference: the highest SNR among finite events was "
              f"{np.nanmax(snr_rough):.2f}, threshold was {SNR_THRESHOLD}. "
              f"If that max is not wildly below threshold, --SNR_THRESHOLD "
              f"may just need lowering for this scenario.)")

# ============================================================
# Save output -- full parameter set + SNR-derived quantities,
# filtered to SNR-passing events only
# ============================================================
out_path = (f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}/EMRI_SNRs_data_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}_{args.merger_model}_{args.UNIVERSE_MODEL}.h5")

with h5py.File(out_path, "w") as f:
    f.attrs['n_events_total'] = n_events
    f.attrs['n_events_passed'] = len(events_idx)
    f.attrs['rho_threshold'] = SNR_THRESHOLD

    for key, val in data.items():
        f.create_dataset(key, data=val[events_idx])

    f.create_dataset("SNR", data=snr_rough[events_idx])
    f.create_dataset("f_gw_det_Hz", data=f_gw[events_idx])
    f.create_dataset("Mchirp_source_kg", data=Mchirp_source[events_idx])

print(f"Saved {len(events_idx)}/{n_events} SNR-passing EMRI parameter rows to {out_path}")


# import matplotlib.pyplot as plt

# finite = np.isfinite(snr_rough)
# plt.figure(figsize=(7, 6))
# sc = plt.scatter(data['lgMBH_mass'][finite], snr_rough[finite],
#                   c=z_arr[finite], cmap='viridis', s=8)
# plt.colorbar(sc, label='z')
# plt.axhline(SNR_THRESHOLD, color='red', ls='--', lw=1,
#             label=fr'$\rho_t={SNR_THRESHOLD:.0f}$')
# plt.yscale('log')
# plt.xlabel(r'$\log_{10}(M_{\rm MBH}/M_\odot)$')
# plt.ylabel(r'Rough SNR ($\rho$)')
# plt.legend()
# plt.tight_layout()
# plot_path = (f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}/SNR_MBHmass_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}_{args.merger_model}_{args.UNIVERSE_MODEL}.png")
# plt.savefig(plot_path)
# plt.close()
# print(f"Saved diagnostic plot to {plot_path}")