#!/usr/bin/env python
"""
Extends DIAGNOSTICS/mass_function_density_check.py's existing BHMF
order-of-magnitude comparison (reused directly below, not rewritten)
with the two specific ratios Marta asked for:

  1. model/observation number-density ratio in the 1e6-1e7 Msun DECADE
     specifically (mass_function_density_check.py only gives pointwise
     values at 1e6/3e6/1e7, not the decade integral).
  2. integrated total-BH-count ratio over the TDE-producing mass range
     used by astrophysical_setup/rate.py's ConstantRate grid (confirmed
     [1e4, 1e7] Msun -- sources_num.py's MBH_MASK and rate.py's
     _N_T/_t_f tables are both defined on exactly this range), first
     raw (ignoring nucleation, per Marta's request) then multiplied by
     the occupation fraction.

Usage:
    python tests/PPS_VERIFICATION/bhmf_order_of_magnitude.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.abspath('./astrophysical_setup'))
sys.path.insert(0, os.path.abspath('./'))
from cosmology import GalaxyStellarMassFunction, MBHMassFunction
from galaxy import _f_occupation_varying_

# ---- same eyeballed z=0 reference digitization as DIAGNOSTICS/mass_function_density_check.py ----
fig_nuclear_logM = np.array([3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.3])
fig_nuclear_phi  = np.array([4e-4, 4.5e-4, 6e-4, 8e-4, 9.5e-4, 1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3, 8e-4, 4e-4, 2e-4])
fig_wander_logM = np.array([3.5, 4.0, 4.3, 4.7, 5.0, 5.5, 6.0, 6.3, 6.7, 7.0, 8.0, 8.5, 9.0, 9.3])
fig_wander_phi  = np.array([3.8e-3, 5e-3, 7e-3, 6e-3, 4.5e-3, 2.8e-3, 1.7e-3, 1.2e-3, 1.0e-3, 1.0e-3, 1.0e-3, 8e-4, 4e-4, 2e-4])

gsmf = GalaxyStellarMassFunction()
mbhmf = MBHMassFunction(gsmf)

logMBH_grid, log10phi_MBH_raw = mbhmf.get_mbhmf(0.0, n_points_mass=4000)
phi_raw = 10 ** log10phi_MBH_raw[0]

lgMgal_grid, _ = gsmf.get_gsmf(0.0, n_points_mass=4000)
f_occ = _f_occupation_varying_(lgMgal_grid)
phi_occ = phi_raw * f_occ


def integrate(logM_lo, logM_hi, phi_model_raw, phi_model_occ):
    """Integrated number of MBHs between logM_lo and logM_hi."""
    mask = (logMBH_grid >= logM_lo) & (logMBH_grid <= logM_hi)
    N_raw = np.trapz(phi_model_raw[mask], logMBH_grid[mask])
    N_occ = np.trapz(phi_model_occ[mask], logMBH_grid[mask])

    mask_n = (fig_nuclear_logM >= logM_lo) & (fig_nuclear_logM <= logM_hi)
    mask_w = (fig_wander_logM >= logM_lo) & (fig_wander_logM <= logM_hi)
    N_fig_n = np.trapz(fig_nuclear_phi[mask_n], fig_nuclear_logM[mask_n])
    N_fig_w = np.trapz(fig_wander_phi[mask_w], fig_wander_logM[mask_w])
    return N_raw, N_occ, N_fig_n, N_fig_w


print("=" * 78)
print("1. Number-density ratio, 1e6-1e7 Msun DECADE (integrated, z=0)")
print("=" * 78)
N_raw, N_occ, N_fig_n, N_fig_w = integrate(6.0, 7.0, phi_raw, phi_occ)
print(f"  UNIVERSE MODEL : {N_raw:.3e} Mpc^-3")
print(f"  UNIVERSE model x occupation  : {N_occ:.3e} Mpc^-3")
print(f"  reference 'Nuclear'      : {N_fig_n:.3e} Mpc^-3")
print(f"  reference 'Nuclear+wander': {N_fig_w:.3e} Mpc^-3")
print(f"  RAW / Nuclear        = {N_raw/N_fig_n:6.1f}x     RAW / Nuclear+wander        = {N_raw/N_fig_w:6.1f}x")
print(f"  OCC-only / Nuclear   = {N_occ/N_fig_n:6.1f}x     OCC-only / Nuclear+wander   = {N_occ/N_fig_w:6.1f}x")

print()
print("=" * 78)
print("2. Integrated total-BH-count ratio, TDE-producing range [1e4, 1e7] Msun")
print("   (matches astrophysical_setup/rate.py's ConstantRate grid range exactly)")
print("=" * 78)
N_raw2, N_occ2, N_fig_n2, N_fig_w2 = integrate(4.0, 7.0, phi_raw, phi_occ)
print(f"  UNIVERSE MODEL : {N_raw2:.3e} Mpc^-3")
print(f"  UNIVERSE model x occupation  : {N_occ2:.3e} Mpc^-3")
print(f"  reference 'Nuclear'      : {N_fig_n2:.3e} Mpc^-3")
print(f"  reference 'Nuclear+wander': {N_fig_w2:.3e} Mpc^-3")
print(f"  RAW / Nuclear        = {N_raw2/N_fig_n2:6.1f}x     RAW / Nuclear+wander        = {N_raw2/N_fig_w2:6.1f}x")
print(f"  OCC-only / Nuclear   = {N_occ2/N_fig_n2:6.1f}x     OCC-only / Nuclear+wander   = {N_occ2/N_fig_w2:6.1f}x")