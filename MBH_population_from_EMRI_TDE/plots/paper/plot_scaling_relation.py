"""Paper figure: MBH-galaxy scaling relations."""
import sys, os
sys.path.insert(0, os.path.abspath('./'))
sys.path.insert(0, os.path.abspath('./astrophysical_setup'))

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style, savefig_all, fig_size, COL_W
from galaxy import Galaxy

setup_style()

Mstar = np.logspace(6, 12, 1000)  # Msun
lgMgal = np.log10(Mstar)

z_obs = 0.5

gal_sigma = Galaxy(z_gal=z_obs, lgMgal=lgMgal, mbh_mass_method='use_sigma')
gal_lgMgal = Galaxy(z_gal=z_obs, lgMgal=lgMgal, mbh_mass_method='use_lgMgal')

sigma_kms = gal_sigma.sigma(unit='km/s')
lgsigma = np.log10(sigma_kms)

def fit_and_plot(ax, x, y, color, label, marker):
    """Scatter the data and overlay a least-squares linear fit."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    ax.scatter(x, y, color=color, s=9, alpha=0.7, label=label, marker=marker)
    # m, b = np.polyfit(x, y, 1)
    # x_fit = np.linspace(x.min(), x.max(), 200)
    # ax.plot(x_fit, m * x_fit + b, color=color, linewidth=2.0, linestyle='-',
            # label=fr'fit: slope$={m:.2f}$, intercept$={b:.2f}$')


# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 15))

# # --- Panel 1: Size-mass / virial relation (Mgal - sigma) ---
# fit_and_plot(ax1, gal_sigma.lgMgal, lgsigma, 'steelblue', r'$\sigma$--$M_{\star}$ (virial relation)')
# ax1.set_xlabel(r'$\log_{10}(M_{\star}/M_\odot)$')
# ax1.set_ylabel(r'$\log_{10}(\sigma\,[\mathrm{km\,s^{-1}}])$')
# ax1.legend(frameon=False, fontsize=20)
# # ax1.grid(alpha=0.3)

# # --- Panel 2: M-sigma relation ---
# fit_and_plot(ax2, lgsigma, gal_sigma.lgMBH_mass, 'steelblue', r'$M_{\rm BH}$--$\sigma$ relation')
# ax2.set_xlabel(r'$\log_{10}(\sigma\,[\mathrm{km\,s^{-1}}])$')
# ax2.set_ylabel(r'$\log_{10}(M_{\rm BH}/M_\odot)$')
# ax2.legend(frameon=False, fontsize=20)
# # ax2.grid(alpha=0.3)

# # --- Panel 3: lgMgal-lgMBH relation, both relations overlaid ---
# fit_and_plot(ax3, gal_sigma.lgMgal, gal_sigma.lgMBH_mass, 'steelblue', r'via $M_{\rm BH}$--$\sigma$')
# fit_and_plot(ax3, gal_lgMgal.lgMgal, gal_lgMgal.lgMBH_mass, 'lightcoral', r'direct $M_{\rm BH}$--$M_\star$')
# ax3.set_xlabel(r'$\log_{10}(M_{\star}/M_\odot)$')
# ax3.set_ylabel(r'$\log_{10}(M_{\rm BH}/M_\odot)$')
# ax3.legend(frameon=False, fontsize=20)
# # ax3.grid(alpha=0.3)

# plt.tight_layout()
# savefig_all(fig, 'scaling_relations')
# plt.close(fig)

fig, ax = plt.subplots(figsize=fig_size(COL_W, 0.72))
fit_and_plot(ax, gal_sigma.lgMgal, gal_sigma.lgMBH_mass, 'steelblue', r'via $M_\bullet$--$\sigma$', 'o')
fit_and_plot(ax, gal_lgMgal.lgMgal, gal_lgMgal.lgMBH_mass, 'lightcoral', r'direct $M_\bullet$--$M_{\rm{gal},\star}$', '*')
ax.set_xlabel(r'$\log_{10}(M_{\rm{gal},\star}/M_\odot)$')
ax.set_ylabel(r'$\log_{10}(M_\bullet/M_\odot)$')
ax.legend(frameon=False, loc='upper left', markerscale=1.3)

plt.tight_layout()
savefig_all(fig, 'scaling_relations')
plt.close(fig)