#!/usr/bin/env python
"""
Model MBHMF at z=0, 1, 2 (using MBHMassFunction.eval_mbhmf --
astrophysical_setup/cosmology.py:219, not previously called anywhere in
the repo) + reference-figure overlay AT z=0 ONLY (per user decision: no
z=1,2 digitized reference data exists in the repo -- only a leftover
mbhmf_compare_z012.png artifact with no surviving source script, see
DIAGNOSTICS/README.md's own TODO). z=1,2 model curves are shown without
a reference overlay.

Produces one figure with three panels:
  (a) MBHMF curves at z=0,1,2 (model only, reference overlay at z=0)
  (b) model/reference ratio at z=0
  (c) integrated number density in [1e6,1e7] and [1e4,1e7] Msun vs redshift

Usage:
    python tests/PPS_VERIFICATION/mbhmf_z012_and_number_density.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('./astrophysical_setup'))
sys.path.insert(0, os.path.abspath('./'))
from cosmology import GalaxyStellarMassFunction, MBHMassFunction

# same eyeballed z=0-only reference digitization as DIAGNOSTICS/mass_function_density_check.py
fig_nuclear_logM = np.array([3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.3])
fig_nuclear_phi  = np.array([4e-4, 4.5e-4, 6e-4, 8e-4, 9.5e-4, 1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3, 8e-4, 4e-4, 2e-4])
fig_wander_logM = np.array([3.5, 4.0, 4.3, 4.7, 5.0, 5.5, 6.0, 6.3, 6.7, 7.0, 8.0, 8.5, 9.0, 9.3])
fig_wander_phi  = np.array([3.8e-3, 5e-3, 7e-3, 6e-3, 4.5e-3, 2.8e-3, 1.7e-3, 1.2e-3, 1.0e-3, 1.0e-3, 1.0e-3, 8e-4, 4e-4, 2e-4])

gsmf = GalaxyStellarMassFunction()
mbhmf = MBHMassFunction(gsmf)

logMBH_plot = np.linspace(4.0, 9.0, 300)
REDSHIFTS = [0.0, 1.0, 2.0]
COLORS = {0.0: "tab:blue", 1.0: "tab:orange", 2.0: "tab:green"}

phi_by_z = {}
for z in REDSHIFTS:
    z_arr = np.full_like(logMBH_plot, z)
    phi_by_z[z] = mbhmf.eval_mbhmf(z_arr, logMBH_plot, return_log10=False)

# ---- integrated number density in two mass ranges, vs redshift ----
z_dense = np.linspace(0.0, 2.0, 21)
N_1e6_1e7 = []
N_1e4_1e7 = []
for z in z_dense:
    logM_fine = np.linspace(4.0, 7.0, 400)
    phi_fine = mbhmf.eval_mbhmf(np.full_like(logM_fine, z), logM_fine, return_log10=False)
    mask67 = (logM_fine >= 6.0) & (logM_fine <= 7.0)
    N_1e6_1e7.append(np.trapz(phi_fine[mask67], logM_fine[mask67]))
    N_1e4_1e7.append(np.trapz(phi_fine, logM_fine))
N_1e6_1e7 = np.array(N_1e6_1e7)
N_1e4_1e7 = np.array(N_1e4_1e7)

fig, axs = plt.subplots(1, 3, figsize=(16, 5))

# panel a: MBHMF at z=0,1,2 + reference at z=0
ax = axs[0]
for z in REDSHIFTS:
    ax.plot(logMBH_plot, phi_by_z[z], color=COLORS[z], label=f"model, z={z:.0f}")
ax.plot(fig_nuclear_logM, fig_nuclear_phi, "k--", label="reference 'Nuclear' (z=0 only)")
ax.plot(fig_wander_logM, fig_wander_phi, "k:", label="reference 'Nuclear+wander' (z=0 only)")
ax.set_yscale("log")
ax.set_xlabel(r"$\log_{10}(M_{\rm BH}/M_\odot)$")
ax.set_ylabel(r"$\phi_{\rm MBH}$ [Mpc$^{-3}$ dex$^{-1}$]")
ax.set_title("MBHMF: model z=0,1,2 vs reference (z=0 only)")
ax.legend(fontsize=8)

# panel b: model/reference ratio at z=0
ax = axs[1]
ref_n_interp = np.interp(logMBH_plot, fig_nuclear_logM, fig_nuclear_phi, left=np.nan, right=np.nan)
ref_w_interp = np.interp(logMBH_plot, fig_wander_logM, fig_wander_phi, left=np.nan, right=np.nan)
ax.plot(logMBH_plot, phi_by_z[0.0] / ref_n_interp, label="model(z=0) / Nuclear")
ax.plot(logMBH_plot, phi_by_z[0.0] / ref_w_interp, label="model(z=0) / Nuclear+wander")
ax.axhline(1.0, color="grey", ls="--", lw=1)
ax.set_yscale("log")
ax.set_xlabel(r"$\log_{10}(M_{\rm BH}/M_\odot)$")
ax.set_ylabel("model / reference")
ax.set_title("Model/reference ratio (z=0)")
ax.legend(fontsize=8)

# panel c: integrated number density vs redshift
ax = axs[2]
ax.plot(z_dense, N_1e6_1e7, label=r"$10^6$-$10^7\,M_\odot$")
ax.plot(z_dense, N_1e4_1e7, label=r"$10^4$-$10^7\,M_\odot$ (TDE range)")
ax.set_yscale("log")
ax.set_xlabel("z")
ax.set_ylabel(r"$\int \phi\, d\log M$ [Mpc$^{-3}$]")
ax.set_title("Integrated MBH number density vs redshift")
ax.legend(fontsize=8)

plt.tight_layout()
out_path = "tests/PPS_VERIFICATION/mbhmf_z012_and_number_density.png"
plt.savefig(out_path, dpi=150)
print(f"Saved {out_path}")

print("\nIntegrated number density [1e6,1e7] Msun vs z:")
for z, n in zip(z_dense, N_1e6_1e7):
    print(f"  z={z:.2f}: {n:.3e} Mpc^-3")
print("\nIntegrated number density [1e4,1e7] Msun (TDE range) vs z:")
for z, n in zip(z_dense, N_1e4_1e7):
    print(f"  z={z:.2f}: {n:.3e} Mpc^-3")
