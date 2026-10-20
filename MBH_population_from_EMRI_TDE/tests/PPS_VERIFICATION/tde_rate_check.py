#!/usr/bin/env python
"""
Checks the intrinsic TDE rate per galaxy per year against Marta's
expected range (1e-4 - 1e-5 per galaxy per year), using the REPO'S OWN
rate table (astrophysical_setup/rate.py's ConstantRate -- the same
object sources_num.py calls for Gamma_TDE_i), not a re-derivation.

For a grid of MBH masses (1e4-1e7 Msun, matching ConstantRate's own
calibration grid) and each of the 4 calibrated sigma/sigma_0 ratios,
builds a single-galaxy Galaxy/NSC pair (Mode 2: lgMBH+sigma given
directly, so no stochastic M-sigma scatter or nucleation/occupation
randomness enters -- this isolates the rate table itself) and reads off
constant_rate(kind='TDE'), converted from events/Gyr to events/year.

Produces tests/PPS_VERIFICATION/tde_rate_check.png: rate vs MBH mass, one
curve per sigma/sigma_0 ratio, with the [1e-5, 1e-4] target band shaded.

Usage:
    python tests/PPS_VERIFICATION/tde_rate_check.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('./astrophysical_setup'))
sys.path.insert(0, os.path.abspath('./'))
from config import pc_to_cm, sec_per_year
from galaxy import Galaxy
from nsc import NSC
from rate import ConstantRate

RATE_LO, RATE_HI = 1e-5, 1e-4   # Marta's expected range, per galaxy per year

# ConstantRate's own calibration grid (astrophysical_setup/rate.py:121,124)
MBH_grid = np.logspace(4.0, 7.0, 25)
sigma_ratios = [0.464, 0.681, 1.0, 1.468]


def km_s_to_pc_yr(sigma_km_s):
    """Convert km/s to pc/yr."""
    return sigma_km_s * 1e5 * sec_per_year / pc_to_cm


rows = []
fig, ax = plt.subplots(figsize=(7, 5.5))

for sr in sigma_ratios:
    rates_per_yr = []
    for MBH in MBH_grid:
        lgMBH = np.log10(MBH)
        gal_tmp = Galaxy(lgMBH=np.array([lgMBH]), sigma_pc_yr=np.array([1.0]),
                          z_gal=np.array([0.0]), mbh_mass_method='use_sigma')
        nsc_tmp = NSC(gal_tmp)
        cr_tmp = ConstantRate(nsc=nsc_tmp)
        sigma0_broggi = cr_tmp.broggi_sigma0(MBH)
        sigma_km_s = sr * sigma0_broggi

        gal = Galaxy(lgMBH=np.array([lgMBH]), sigma_pc_yr=np.array([km_s_to_pc_yr(sigma_km_s)]),
                      z_gal=np.array([0.0]), mbh_mass_method='use_sigma')
        nsc = NSC(gal)
        cr = ConstantRate(nsc=nsc)
        rate_TDE_Gyr = cr.constant_rate(kind='TDE')[0]
        rate_per_yr = rate_TDE_Gyr / 1e9
        rates_per_yr.append(rate_per_yr)
        rows.append((MBH, sr, rate_per_yr))

    ax.plot(MBH_grid, rates_per_yr, marker='o', ms=3, label=fr"$\sigma/\sigma_0={sr}$")

ax.axhspan(RATE_LO, RATE_HI, color='grey', alpha=0.25, label=f"target [{RATE_LO:.0e}, {RATE_HI:.0e}] /gal/yr")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$M_{\rm BH}$ [$M_\odot$]", fontsize=14)
ax.set_ylabel("Intrinsic TDE rate", fontsize=14)
ax.legend(fontsize=10)
plt.tight_layout()
out_path = "tests/PPS_VERIFICATION/tde_rate_check.png"
plt.savefig(out_path, dpi=150)
print(f"Saved {out_path}\n")

print(f"{'M_BH':>10s} {'sigma/sigma0':>12s} {'rate/gal/yr':>14s} {'in [1e-5,1e-4]?':>16s}")
n_in_range = 0
for MBH, sr, r in rows:
    ok = RATE_LO <= r <= RATE_HI
    n_in_range += ok
    if MBH in (MBH_grid[0], MBH_grid[len(MBH_grid)//4], MBH_grid[len(MBH_grid)//2], MBH_grid[-1]):
        print(f"{MBH:10.2e} {sr:12.3f} {r:14.3e} {'YES' if ok else 'no':>16s}")

frac_in_range = n_in_range / len(rows)
print(f"\n{n_in_range}/{len(rows)} (MBH, sigma_ratio) grid points fall inside "
      f"[{RATE_LO:.0e}, {RATE_HI:.0e}] /galaxy/yr ({frac_in_range:.0%}).")
print("\nRESULT:", "PASS -- intrinsic rate is within the expected 1e-4-1e-5/gal/yr range "
      "across most of the calibrated grid." if frac_in_range > 0.5 else
      "MISMATCH -- rate falls outside the expected range for most of the grid, worth a closer look.")
