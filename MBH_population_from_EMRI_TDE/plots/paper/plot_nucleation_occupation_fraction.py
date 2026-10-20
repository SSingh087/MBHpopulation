"""Paper figure: nucleation and occupation fractions versus stellar mass."""
import sys, os
sys.path.insert(0, os.path.abspath('./'))
sys.path.insert(0, os.path.abspath('./astrophysical_setup'))

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style, savefig_all, fig_size, COL_W
from galaxy import _f_NSC_Hannah_, _f_NSC_Neumayer_, _f_occupation_varying_

setup_style()

lgMgal_grid = np.linspace(4, 12, 500)

OCC_UNIFORM_VALUE = 1.0
f_nuc_hannah = _f_NSC_Hannah_(lgMgal_grid)
f_nuc_neumayer = _f_NSC_Neumayer_(lgMgal_grid)
f_occ_uniform = np.full_like(lgMgal_grid, OCC_UNIFORM_VALUE)
f_occ_varying = _f_occupation_varying_(lgMgal_grid)

fig, ax = plt.subplots(figsize=fig_size(COL_W, 0.95))

ax.plot(lgMgal_grid, f_nuc_hannah, label='Nucleation (Hannah+24)', linewidth=1.5, color='#0072B2')
ax.plot(lgMgal_grid, f_nuc_neumayer, label='Nucleation (Neumayer+20)', linewidth=1.5, color='#D55E00')
ax.plot(lgMgal_grid, f_occ_uniform, label=f'Occupation (uniform, $f_{{\\rm occ}}={OCC_UNIFORM_VALUE:g}$)', linewidth=1.5, color='#009E73', linestyle='--')
ax.plot(lgMgal_grid, f_occ_varying, label='Occupation (varying, Zou+25)', linewidth=1.5, color='#882255', linestyle='--')

#  check points off arXiv:2407.10911 Fig. 4
x_hannah = np.array([6.5, 7.5, 8.5, 9.0, 9.3, 9.7, 10.0, 10.3, 10.7, 11.0, 11.3])
y_hannah = np.array([1.00, 1.00, 1.00, 1.00, 0.95, 0.90, 0.85, 0.65, 0.35, 0.18, 0.05])
x_neumayer = np.array([6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.3])
y_neumayer = np.array([0.17, 0.30, 0.42, 0.55, 0.68, 0.80, 0.72, 0.58, 0.42, 0.15, 0.03])

ax.scatter(x_hannah, y_hannah, marker='o', s=14, facecolors='none',
           edgecolors='#0072B2', linewidths=0.9, zorder=5)
        #    label='data points (Hannah+24')
ax.scatter(x_neumayer, y_neumayer, marker='s', s=14, facecolors='none',
           edgecolors='#D55E00', linewidths=0.9, zorder=5)
        #    label='Neumayer+20 ( off Fig. 4, arXiv:2407.10911)')

ax.set_xlabel(r'$\log_{10}(M_{\rm gal}/M_\odot)$')
ax.set_ylabel(r'Nucleation fraction, $f_{\rm nucl.}$')
ax.set_xlim(4, 12)
ax.set_ylim(-0.05, 1.5)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.legend(loc='upper left', labelspacing=0.3)

ax_right = ax.twinx()
ax_right.set_ylabel(r'Occupation fraction, $f_{\rm occ.}$')
ax_right.set_ylim(-0.05, 1.5)
ax_right.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

plt.tight_layout()
savefig_all(fig, 'nucleation_occupation_fraction')
plt.close(fig)
