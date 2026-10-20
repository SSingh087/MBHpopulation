#!/usr/bin/env python
"""
Checks this repo's derived MBH mass function (MBHMassFunction, built by
change-of-variables from GalaxyStellarMassFunction via the M-Mgal relation,
astrophysical_setup/cosmology.py) against the reference "Nuclear" /
"Nuclear + wander" MBHMF figure (# Mpc^-3 dex^-1 vs M_BH), at z=0,1,2.

Extends tests/PPS_VERIFICATION/bhmf_order_of_magnitude.py (which already
eyeball-digitized this exact reference figure and did the z=0 ratio math)
by (a) adding z=1,2, (b) plotting model vs reference, and (c) keeping the
same raw -> x-occupation ratio breakdown at each redshift.

Reference digitization is the same eyeballed points as
bhmf_order_of_magnitude.py -- order-of-magnitude only, not a rigorous
digitization (no axis gridlines/data available beyond reading the figure).
The reference curve is presumed z~0 (no z-evolution given in the figure);
comparing it against the model at z=1,2 shows how much the model's own
BHMF evolves relative to that fixed z=0 anchor, not a z-matched comparison.

Usage:
    python tests/PPS_VERIFICATION/bhmf_vs_reference_multi_z.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('./astrophysical_setup'))
sys.path.insert(0, os.path.abspath('./'))
from cosmology import GalaxyStellarMassFunction, MBHMassFunction
from galaxy import _f_occupation_varying_

# ---- eyeballed digitization of the reference figure (same as bhmf_order_of_magnitude.py) ----
fig_nuclear_logM = np.array([3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.3])
fig_nuclear_phi  = np.array([4e-4, 4.5e-4, 6e-4, 8e-4, 9.5e-4, 1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3, 8e-4, 4e-4, 2e-4])
fig_wander_logM = np.array([3.5, 4.0, 4.3, 4.7, 5.0, 5.5, 6.0, 6.3, 6.7, 7.0, 8.0, 8.5, 9.0, 9.3])
fig_wander_phi  = np.array([3.8e-3, 5e-3, 7e-3, 6e-3, 4.5e-3, 2.8e-3, 1.7e-3, 1.2e-3, 1.0e-3, 1.0e-3, 1.0e-3, 8e-4, 4e-4, 2e-4])

REDSHIFTS = [0.0, 1.0, 2.0]
TDE_LOGM_LO, TDE_LOGM_HI = 4.0, 7.0   # matches rate.py's ConstantRate grid / sources_num.py's MBH_MASK
DECADE_LOGM_LO, DECADE_LOGM_HI = 6.0, 7.0

gsmf = GalaxyStellarMassFunction()
mbhmf = MBHMassFunction(gsmf)

logMBH_grid, log10phi_MBH_allz = mbhmf.get_mbhmf(np.array(REDSHIFTS), n_points_mass=4000)
lgMgal_grid, _ = gsmf.get_gsmf(np.array(REDSHIFTS), n_points_mass=4000)
f_occ = _f_occupation_varying_(lgMgal_grid)  # same lgMgal grid MBHMassFunction used internally


def integrate(logM_lo, logM_hi, phi_raw_z, phi_occ_z):
    mask = (logMBH_grid >= logM_lo) & (logMBH_grid <= logM_hi)
    N_raw = np.trapz(phi_raw_z[mask], logMBH_grid[mask])
    N_occ = np.trapz(phi_occ_z[mask], logMBH_grid[mask])

    mask_n = (fig_nuclear_logM >= logM_lo) & (fig_nuclear_logM <= logM_hi)
    mask_w = (fig_wander_logM >= logM_lo) & (fig_wander_logM <= logM_hi)
    N_fig_n = np.trapz(fig_nuclear_phi[mask_n], fig_nuclear_logM[mask_n])
    N_fig_w = np.trapz(fig_wander_phi[mask_w], fig_wander_logM[mask_w])
    return N_raw, N_occ, N_fig_n, N_fig_w


# ---- Plot: model (raw AND x occupation) at each z vs reference curves,
#      with the two integration ranges shaded and their integrated values annotated ----
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)

colors = ['tab:green', 'tab:orange', 'tab:purple']

for ax, occ_on in zip(axes, [False, True]):
    ax.plot(fig_nuclear_logM, fig_nuclear_phi, 'r-', lw=2, label="reference 'Nuclear'")
    ax.plot(fig_wander_logM, fig_wander_phi, 'b-.', lw=2, label="reference 'Nuclear+wander'")

    for i, z in enumerate(REDSHIFTS):
        phi_raw_z = 10 ** log10phi_MBH_allz[i]
        phi_z = phi_raw_z * f_occ if occ_on else phi_raw_z
        label = f"model (x occupation), z={z:.0f}" if occ_on else f"model (raw, no occ/nuc), z={z:.0f}"
        ax.plot(logMBH_grid, phi_z, color=colors[i], lw=1.5, label=label)

    # shade the two integration ranges (z=0 model curve only, to keep it legible)
    phi_z0 = 10 ** log10phi_MBH_allz[0]
    if occ_on:
        phi_z0 = phi_z0 * f_occ
    for lo, hi, hatch, shade_label in [(DECADE_LOGM_LO, DECADE_LOGM_HI, '//', '1e6-1e7 decade'),
                                         (TDE_LOGM_LO, TDE_LOGM_HI, '\\\\', 'TDE range 1e4-1e7')]:
        mask = (logMBH_grid >= lo) & (logMBH_grid <= hi)
        ax.fill_between(logMBH_grid[mask], 1e-6, phi_z0[mask], alpha=0.10, hatch=hatch,
                         color='grey', label=f'{shade_label}')

    ax.set_yscale("log")
    ax.set_xlim(3.5, 9.5)
    ax.set_ylim(1e-5, 1e-1)
    ax.set_xlabel(r"$\log_{10}(M_{\rm BH}/M_\odot)$", fontsize=20)
    ax.set_title("model x occupation fraction" if occ_on else "model, raw (no occ/nuc)", fontsize=20)
    ax.legend(fontsize=10, loc='upper right')
    ax.tick_params(axis='both', which='major', labelsize=14)

axes[0].set_ylabel(r"# Mpc$^{-3}$ dex$^{-1}$", fontsize=20)
plt.tight_layout()
out1 = "tests/PPS_VERIFICATION/bhmf_vs_reference_multi_z.png"
plt.savefig(out1, dpi=150)
print(f"Saved {out1}")
plt.close(fig)

# ---- Second figure: bar chart of the integrated N values themselves (the actual numbers quoted) ----
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.5))
range_defs = [("1e6-1e7 Msun decade", DECADE_LOGM_LO, DECADE_LOGM_HI),
              ("TDE range 1e4-1e7 Msun", TDE_LOGM_LO, TDE_LOGM_HI)]

for ax, (title, lo, hi) in zip(axes2, range_defs):
    x = np.arange(len(REDSHIFTS))
    width = 0.18
    N_raw_list, N_occ_list = [], []
    for i, z in enumerate(REDSHIFTS):
        phi_raw_z = 10 ** log10phi_MBH_allz[i]
        phi_occ_z = phi_raw_z * f_occ
        N_raw, N_occ, N_fig_n, N_fig_w = integrate(lo, hi, phi_raw_z, phi_occ_z)
        N_raw_list.append(N_raw)
        N_occ_list.append(N_occ)

    ax.bar(x - 1.5*width, N_raw_list, width, label='model raw', color='tab:green')
    ax.bar(x - 0.5*width, N_occ_list, width, label='model x occupation', color='tab:purple')
    ax.bar(x + 0.5*width, [N_fig_n]*len(REDSHIFTS), width, label='reference Nuclear', color='tab:red', alpha=0.7)
    ax.bar(x + 1.5*width, [N_fig_w]*len(REDSHIFTS), width, label='reference Nuclear+wander', color='tab:blue', alpha=0.7)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([f"z={z:.0f}" for z in REDSHIFTS])
    ax.set_ylabel(r"integrated N [Mpc$^{-3}$]")
    ax.set_title(title)
    ax.legend(fontsize=20)

plt.tight_layout()
out1b = "tests/PPS_VERIFICATION/bhmf_integrated_N_bars.png"
plt.savefig(out1b, dpi=150)
print(f"Saved {out1b}")
plt.close(fig2)

# ---- Print ratio tables at each z ----
for i, z in enumerate(REDSHIFTS):
    phi_raw_z = 10 ** log10phi_MBH_allz[i]
    phi_occ_z = phi_raw_z * f_occ

    print("=" * 78)
    print(f"z = {z:.0f}")
    print("=" * 78)

    N_raw, N_occ, N_fig_n, N_fig_w = integrate(DECADE_LOGM_LO, DECADE_LOGM_HI, phi_raw_z, phi_occ_z)
    print(f"[1e6-1e7 Msun decade]")
    print(f"  UNIVERSE MODEL : {N_raw:.3e} Mpc^-3")
    print(f"  UNIVERSE model x occupation  : {N_occ:.3e} Mpc^-3")
    print(f"  reference Nuclear : {N_fig_n:.3e} Mpc^-3   ->  raw/ref = {N_raw/N_fig_n:6.1f}x   occ/ref = {N_occ/N_fig_n:6.1f}x")
    print(f"  reference Nuclear+wander : {N_fig_w:.3e} Mpc^-3   ->  raw/ref = {N_raw/N_fig_w:6.1f}x   occ/ref = {N_occ/N_fig_w:6.1f}x")

    N_raw2, N_occ2, N_fig_n2, N_fig_w2 = integrate(TDE_LOGM_LO, TDE_LOGM_HI, phi_raw_z, phi_occ_z)
    print(f"[TDE-producing range 1e4-1e7 Msun]")
    print(f"  UNIVERSE MODEL : {N_raw2:.3e} Mpc^-3")
    print(f"  UNIVERSE model x occupation  : {N_occ2:.3e} Mpc^-3")
    print(f"  reference Nuclear : {N_fig_n2:.3e} Mpc^-3   ->  raw/ref = {N_raw2/N_fig_n2:6.1f}x   occ/ref = {N_occ2/N_fig_n2:6.1f}x")
    print(f"  reference Nuclear+wander: {N_fig_w2:.3e} Mpc^-3   ->  raw/ref = {N_raw2/N_fig_w2:6.1f}x   occ/ref = {N_occ2/N_fig_w2:6.1f}x")
    print()
