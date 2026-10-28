"""Paper figure: NSC cusp evolution timeline and last-major-merger models."""
import os, sys
sys.path.insert(0, os.path.abspath('./astrophysical_setup'))
sys.path.insert(0, os.path.abspath('./tests/TEST_merger_rates/scripts'))
sys.path.insert(0, os.path.abspath('./'))

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style, savefig_all, fig_size, COL_W
from cosmology import CosmologyModel, LastMajorMerger
from merger_models import make_models

setup_style()

# seed=9 gives one non-censored draw (z_obs=1.0) and one censored draw (z_obs=0.1)
np.random.seed(9)
cosmo = CosmologyModel(H0=70.0, Om0=0.3, Tcmb0=2.725)
lmm = LastMajorMerger(cosmology_model=cosmo, merger_model="oleary21")

kappa = 0.25       # fraction of relaxation time for cusp regrowth
t_rlx = 1.5        # Gyr, relaxation time
z_obs_array = np.array([1.0, 0.1])

z = np.linspace(0.01, lmm.z_max, 500)
t_gyr = cosmo.age_Gyr(z)
_, h_z = lmm.hazard_function(z)
_, S = lmm.survival_function(z)
z_LMM, t_LMM, t_obs = lmm.sample_LMM_times(z_obs_array, size=1)
T_c = [np.maximum(0, t_gyr - (t_v + kappa * t_rlx)) for t_v in t_LMM]

# Comparison merger models (survival-function overlay)
M = make_models(cosmo)
OVERLAY = {
    "mundy17":      ("#555555", '--', 1.6, r"Mundy+17"),
    "duan24_plexp": ("#7b3fa0", '--', 1.6, r"Duan+25"),
}

fig, ax1 = plt.subplots(figsize=fig_size(COL_W, 0.9))

ax1.fill_between(t_gyr, 0, h_z/np.max(h_z), color='orangered', alpha=0.2,
                 )
ax1.plot(t_gyr, h_z/np.max(h_z), color='orangered', linewidth=1.4)

ax1.plot(t_gyr, S, color='green', linewidth=1.6, linestyle='--',
         label=r"O'Leary+21")
for key, (c, ls, lw, lbl) in OVERLAY.items():
    g = M[key].hazard_integrand(z)
    Lam = np.concatenate([[0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(z))])
    ax1.plot(cosmo.age_Gyr(z), np.exp(-Lam), color=c, linewidth=lw, linestyle=ls,
             label=lbl)

ax1.set_xlabel(r'Cosmic Time $t$ (Gyr)')
ax1.set_ylabel(r'Normalized Probability')
ax1.set_ylim(0, 1.35)
ax1.set_yticks(np.arange(0, 1.01, 0.2))

ax2 = ax1.twinx()
colors = ['royalblue', 'navy']
for i, T in enumerate(T_c):
    ax2.plot(t_gyr, T, color=colors[i], linewidth=1.6,
             label=fr'$z_{{\rm obs}}={z_obs_array[i]}$')
ax2.set_ylabel(r'Cusp age $T_c$ (Gyr)')
ax2.set_yticks(np.arange(0, 13, 2))
ax2.set_ylim(0, max(T.max() for T in T_c)*1.1*1.35/1.05)

for i, t_v in enumerate(t_LMM):
    ax2.axvspan(t_v, t_v + kappa*t_rlx, color='lightblue', alpha=0.35,
)
    ax2.axvline(t_v, color='gray', linestyle='--', linewidth=0.9, alpha=0.7)

# t_LMM of each sampled merger as ticks on a secondary top axis
ax_top = ax1.secondary_xaxis('top')
ax_top.set_xticks(list(t_LMM))
ax_top.set_xticklabels([f'{t:.1f}' for t in t_LMM])
ax_top.set_xlabel(r'$t_{\rm LMM}$ (Gyr)')

h1, b1 = ax1.get_legend_handles_labels()
h2, b2 = ax2.get_legend_handles_labels()
leg1 = ax2.legend(h1, b1, loc='upper left', bbox_to_anchor=(0.04, 0.99), ncol=1,
                  handlelength=1.8, labelspacing=0.25, frameon=True, facecolor='white',
                  framealpha=1.0, edgecolor='none', borderpad=0.2)
ax2.add_artist(leg1)
ax2.legend(h2, b2, loc='upper left', bbox_to_anchor=(0.50, 0.99), ncol=1,
           handlelength=1.8, labelspacing=0.25, frameon=True, facecolor='white',
           framealpha=1.0, edgecolor='none', borderpad=0.2)

# direct labels
ax1.text(13.1, 0.03, r'Merger rate $\varphi_{\rm MM}$', color='orangered', fontsize=8, ha='right')
ax1.text(5.95, 0.66, 'Cusp\nregrowth', fontsize=8, color='steelblue', va='center')

ax1.set_xlim(t_gyr.min(), t_gyr.max())
plt.tight_layout()
savefig_all(fig, 'NSC_cusp_evolution_timeline')
plt.close(fig)
