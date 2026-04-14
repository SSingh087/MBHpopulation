import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

os.makedirs("./plots", exist_ok=True)

latex_map = {
    # --- EMRI parameters ---
    "MBHspin": r"$a_\ast$",
    "Phi_phi0": r"$\Phi_{\phi,0}$",
    "Phi_r0": r"$\Phi_{r,0}$",
    "Phi_theta0": r"$\Phi_{\theta,0}$",
    "T_SIGNAL_duration_years": r"$T_{\rm signal}\;[\mathrm{yr}]$",
    "Y0": r"$Y_0$",
    "distance_Gpc": r"$D_L\;[\mathrm{Gpc}]$",
    "e0": r"$e_0$",
    "lgMBH_mass": r"$\log_{10}(M_{\bullet}/M_\odot)$",
    "phiK": r"$\phi_K$",
    "phiS": r"$\phi_S$",
    "qK": r"$q_K$",
    "qS": r"$q_S$",
    "sBH_mass": r"$m_\star\;[M_\odot]$",

    # --- TDE parameters ---
    "alpha": r"$\alpha$",
    "beta": r"$\beta$",
    "eta": r"$\eta$",
    "mbh_6": r"$M_{\bullet}/10^6\,M_\odot$",
    "redshift": r"$z$",
    "stellar_mass": r"$M_\star$",
}

theta0_EMRI = {}
theta0_TDE = {}

print("Inspecting EMRI data:")
with h5py.File('../fisher_analysis/true_data_EMRI.h5', 'r') as hf:
    for key in hf.keys():
        print(f"Dataset '{key}' shape = {hf[key].shape}, dtype = {hf[key].dtype}")
        theta0_EMRI[key] = hf[key][()]

print("\nInspecting TDE data:")
with h5py.File('../fisher_analysis/true_data_TDE_ZTF.h5', 'r') as hf:
    for key in hf.keys():
        print(f"Dataset '{key}' shape = {hf[key].shape}, dtype = {hf[key].dtype}")
        theta0_TDE[key] = hf[key][()]

z_emri = theta0_EMRI["distance_Gpc"].mean(axis=1)

for key, values in theta0_EMRI.items():

    if key == "distance_Gpc":   # skip plotting z vs z
        continue

    x = values.mean(axis=1)     # event-level mean of parameter samples
    y = z_emri                  # distance as proxy for redshift

    plt.figure(figsize=(8,6))
    counts, xedges, yedges, im = plt.hist2d(
        x, y, bins=40, cmap='magma'
    )
    plt.colorbar(im, label=r"Counts")

    plt.xlabel(latex_map.get(key, key), fontsize=16)
    plt.ylabel(r"$D_L\;[\mathrm{Gpc}]$", fontsize=16)
    plt.title(f"EMRI: {latex_map.get(key, key)} vs $D_L$", fontsize=18)

    plt.tight_layout()
    plt.savefig(f'./plots/EMRI_2D_{key}.png')
    plt.close()


z_tde = theta0_TDE["redshift"].mean(axis=1)

for key, values in theta0_TDE.items():

    if key == "redshift":
        continue

    x = values.mean(axis=1)
    y = z_tde

    plt.figure(figsize=(8,6))
    counts, xedges, yedges, im = plt.hist2d(
        x, y, bins=40, cmap='viridis'
    )
    plt.colorbar(im, label=r"Counts")

    plt.xlabel(latex_map.get(key, key), fontsize=16)
    plt.ylabel(r"$z$", fontsize=16)
    plt.title(f"TDE: {latex_map.get(key, key)} vs $z$", fontsize=18)

    plt.tight_layout()
    plt.savefig(f'./plots/TDE_2D_{key}.png')
    plt.close()