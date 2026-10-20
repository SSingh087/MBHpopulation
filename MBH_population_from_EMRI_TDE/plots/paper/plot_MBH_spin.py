"""Paper figure: initial MBH spin distribution."""
import sys, os
sys.path.insert(0, os.path.abspath('./'))

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style, savefig_all, fig_size, COL_W

setup_style()

def initial_MBHspin(lgMBH_mass):

    alpha0_low_mass = 2.5
    lambda_alpha_low_mass = 0.1
    beta_low_mass = 5.0

    alpha0_high_mass = 9.5
    lambda_alpha_high_mass = 0.1
    beta_high_mass = 5.0

    spins = np.zeros(len(lgMBH_mass))
    spins[lgMBH_mass < 6] = np.random.beta(alpha0_low_mass + lambda_alpha_low_mass * (lgMBH_mass[lgMBH_mass < 6] - 6), beta_low_mass)
    spins[lgMBH_mass >= 6] = np.random.beta(alpha0_high_mass + lambda_alpha_high_mass * (lgMBH_mass[lgMBH_mass >= 6] - 6), beta_high_mass)

    return spins

lgMBH_mass_grid = np.linspace(4.0, 10.0, 100000)
spin_samples = np.array(initial_MBHspin(lgMBH_mass_grid))

# Separate into low and high mass components
low_mass_mask = lgMBH_mass_grid < 6
high_mass_mask = lgMBH_mass_grid >= 6

spin_low_mass = spin_samples[low_mass_mask]
spin_high_mass = spin_samples[high_mass_mask]


fig, ax = plt.subplots(figsize=fig_size(COL_W, 0.78))

ax.hist(spin_samples, bins=100, histtype='step', linewidth=1.5, density=True,
        color='black', label=f'Average spin={spin_samples.mean():.3f}')
ax.hist(spin_low_mass, bins=50, histtype='step', linewidth=1.1, density=True,
        color='steelblue', alpha=0.6, label=r'$M_{{\rm MBH}} < 10^6 M_\odot$ contribution')
ax.hist(spin_high_mass, bins=50, histtype='step', linewidth=1.1, density=True,
        color='coral', alpha=0.6, label=r'$M_{{\rm MBH}} > 10^6 M_\odot$ contribution')
ax.vlines(x=0.3, ymin=0, ymax=ax.get_ylim()[1], color='gray', linestyle='--', linewidth=1.1, alpha=0.7)
ax.vlines(x=0.7, ymin=0, ymax=ax.get_ylim()[1], color='gray', linestyle=':', linewidth=1.1, alpha=0.7)
ax.set_xlabel(r'$a_{\rm BH}$ (spin)')
ax.set_yticks([])
ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
ax.legend(loc='upper left', labelspacing=0.3)
ax.grid(alpha=0.3, linewidth=0.4)
ax.set_xlim(0, 1)

plt.tight_layout()
savefig_all(fig, 'MBHSpin_distribution')
plt.close(fig)

