#!/usr/bin/env python
"""
Plot the ZTF/LSST completeness-gradient sky footprints (built once by
build_footprint_maps.py). These maps depend only on survey pointing
strategy/geometry, not on any (mbh_mass_method, nucleation_method,
occupation_method, TDE_model, UNIVERSE_MODEL) combo, so there is a single
footprint plot for the paper rather than one per model/universe.

Usage:
    python plot_ZTF_LSST_footprint.py
"""

import sys, os
sys.path.insert(0, os.path.abspath('./'))
from utils import ASTROPHYSICAL_DATA_DIR

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style, savefig_all, fig_size, TEXT_W

PALOMAR_LAT_DEG = 33.35
CERRO_PACHON_LAT_DEG = -30.24

# ────────────────────────────────────────────────────────────────────
# 1. Load the cached coverage maps (combo-independent).
# ────────────────────────────────────────────────────────────────────
COVERAGE_NPZ = os.path.join(ASTROPHYSICAL_DATA_DIR, 'coverage_maps_nohealpy.npz')
if not os.path.exists(COVERAGE_NPZ):
    raise RuntimeError(
        f"{COVERAGE_NPZ} not found. Run `python data_generation/build_footprint_maps.py` "
        f"once first -- it builds the ZTF/Rubin coverage maps.")

cov = np.load(COVERAGE_NPZ)
ztf_grad, rubin_grad = cov['ztf_grad'], cov['rubin_grad']
ra_centers_deg, dec_centers_deg = cov['ra_centers_deg'], cov['dec_centers_deg']

RA_2d, DEC_2d = np.meshgrid(ra_centers_deg, dec_centers_deg)  # (N_DEC, N_RA)
LON, LAT = -np.deg2rad(RA_2d), np.deg2rad(DEC_2d)             # mollweide convention, mirrored


def plot_panel(ax, grad_map, obs_lat_deg, obs_label, title):
    """Plot one panel."""
    pcm = ax.pcolormesh(LON, LAT, grad_map, cmap='viridis', vmin=0, vmax=1, shading='auto')
    ax.grid(True, alpha=0.3)

    lon_obs, lat_obs = -np.deg2rad(0.0), np.deg2rad(obs_lat_deg)
    ax.scatter(lon_obs, lat_obs, marker='*', s=60, facecolor='#D55E00',
               edgecolor='white', linewidth=0.8, zorder=10,
               label=f'{obs_label} zenith Dec')  # described in the caption

    # ax.set_title(title, pad=25)
    ax.set_xticks(np.deg2rad([-120, -60, 0, 60, 120]))
    return pcm


setup_style()

FOOTPRINT_DIR = os.path.join(ASTROPHYSICAL_DATA_DIR, "footprints")
os.makedirs(FOOTPRINT_DIR, exist_ok=True)

fig1 = plt.figure(figsize=fig_size(TEXT_W / 2, 0.64))
ax1 = fig1.add_subplot(1, 1, 1, projection='mollweide')
pcm1 = plot_panel(ax1, ztf_grad, PALOMAR_LAT_DEG, 'Palomar', 'ZTF era footprint \\& completeness gradient')
cb1 = fig1.colorbar(pcm1, ax=ax1, orientation='horizontal', pad=0.08, shrink=0.8)
cb1.set_label('Pointing-density (normalized completeness)')
savefig_all(fig1, "ztf_footprint_gradient")
plt.close(fig1)

fig2 = plt.figure(figsize=fig_size(TEXT_W / 2, 0.64))
ax2 = fig2.add_subplot(1, 1, 1, projection='mollweide')
pcm2 = plot_panel(ax2, rubin_grad, CERRO_PACHON_LAT_DEG, 'Cerro Pachon', 'LSST/Rubin era footprint \\& completeness gradient')
cb2 = fig2.colorbar(pcm2, ax=ax2, orientation='horizontal', pad=0.08, shrink=0.8)
cb2.set_label('Pointing-density (normalized completeness)')
savefig_all(fig2, "lsst_footprint_gradient")
plt.close(fig2)
