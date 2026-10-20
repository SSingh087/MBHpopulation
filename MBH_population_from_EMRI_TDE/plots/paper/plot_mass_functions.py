"""Paper figure: comparison of the MBH mass functions."""
import sys, os
sys.path.insert(0, os.path.abspath('./'))
sys.path.insert(0, os.path.abspath('./astrophysical_setup'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_style import setup_style, savefig_all, fig_size, COL_W
from cosmology import GalaxyStellarMassFunction, MBHMassFunction
from galaxy import Galaxy
from config import MBH_A_sigma, MBH_B_sigma

setup_style()

z_grid = np.linspace(0.01, 10.0, 5)

GSMF = GalaxyStellarMassFunction()
MBHMF = MBHMassFunction(GSMF)

lgMgal_grid, phi_gsmf_grid = GSMF.get_gsmf(z_gal=z_grid, n_points_mass=1000)

lgMBH_grid, phi_mbhmf_grid = MBHMF.get_mbhmf(z_gal=z_grid, n_points_mass=1000)

cmap = plt.get_cmap('plasma')
colors = cmap(np.linspace(0, 1, len(z_grid)))

fig, ax1 = plt.subplots(figsize=fig_size(COL_W, 0.78))

for i, z in enumerate(z_grid):
    ax1.plot(lgMgal_grid, phi_gsmf_grid[i], color=colors[i], linestyle='-')

ax2 = ax1.twiny()
for i, z in enumerate(z_grid):
    ax2.plot(lgMBH_grid, phi_mbhmf_grid[i], color=colors[i], linestyle='--')
ax2.set_xlim(*Galaxy.lgMBH_from_Mgal(np.array([6.5, 12.0]), A=MBH_A_sigma, B=MBH_B_sigma))
ax2.set_xlabel(r'$\log_{10}(M_\bullet/M_{\odot})$')

ax1.set_xlabel(r'$\log_{10}(M_{\mathrm{gal},\star}/M_{\odot})$')
ax1.set_ylabel(r'$\log_{10}(\phi/\mathrm{cMpc}^{-3} \mathrm{dex}^{-1})$')
ax1.set_xlim(6.5, 12)
ax1.set_ylim(-5, 1)
ax2.xaxis.labelpad = 4

z_legend = [Line2D([0], [0], color=colors[i], lw=1.2, label=f"$z={z_grid[i]:.2f}$") for i in range(len(z_grid))]
style_legend = [Line2D([0], [0], color='k', lw=1.2, ls='-', label='GSMF'),
                Line2D([0], [0], color='k', lw=1.2, ls='--', label='MBHMF')]
leg_z = ax1.legend(handles=z_legend, loc='upper right', ncol=1, labelspacing=0.25)
ax1.add_artist(leg_z)
ax1.legend(handles=style_legend, loc='lower left')

plt.tight_layout()
savefig_all(fig, 'MF_comparison')
plt.close(fig)
