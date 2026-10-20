"""Paper figure: Dehnen density and mass profiles of the NSC for representative MBH masses."""
import os, sys
sys.path.insert(0, os.path.abspath('./astrophysical_setup'))
sys.path.insert(0, os.path.abspath('./'))

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style, savefig_all, fig_size, COL_W

setup_style()

from config import pc_to_cm, G_pc3_per_Msun_yr2, c_pc_per_year, Msun, Rsun
from galaxy import Galaxy
from nsc import NSC, CompactObject
from density import DehnenProfile

rng = np.random.default_rng(seed=42)

N_OBS = 2

z_gal = np.ones(N_OBS)
ra = rng.uniform(0.0, 360.0, N_OBS)
dec = rng.uniform(-90.0, 90.0, N_OBS)
# lgMgal = rng.uniform(6.5, 12, N_OBS)

lgMgal = rng.uniform(9.0, 9.2, N_OBS) # these value ranges are in for Luca's paper 
# MBH masses fixed to the two representative values quoted in the paper caption (10^6.6 and 10^4.5 Msun)
lgMBH_fixed = np.array([6.6, 4.5])

nucleation_indices = Galaxy.check_nucleation(lgMgal, rng=rng)
print(f"Nucleated: {nucleation_indices.sum()}/{N_OBS} = {100*nucleation_indices.sum()/N_OBS:.1f}%")
gal = Galaxy(lgMgal=lgMgal, lgMBH=lgMBH_fixed, z_gal=z_gal, ra_deg=ra, dec_deg=dec, rng=rng, nucleation_occurs=nucleation_indices)

lgMBH_mass = gal.lgMBH_mass


nsc = NSC(galaxy=gal)

r_inf = nsc.r_influence(kvir=1.0, unit='pc')
print("Influence radius (pc):", r_inf)

CO_objs = CompactObject(nsc=nsc, masses={'sBH': 10.0, 'star': 1.0}, total_mass_in_MBH_mass_units={'sBH': 0.2, 'star': 20.0}, types_CO=['sBH', 'star'], types_masses='same_mass', type_CO_limits=None)

# Create r_grid with shape (N_OBS, Nr) to match number of galaxies
r_grid = np.logspace(-6, 3, 100)
r_grid = np.tile(r_grid, (nucleation_indices.sum(), 1))  # Shape (N_OBS, Nr)

profile_obj = DehnenProfile(nsc, CO_objs, gamma=1.5)

dehnen_at_r_influence = profile_obj.dehnen_number_density_at_radius(r_inf, kind='TDE')
dehnen_at_r_influence_emri = profile_obj.dehnen_number_density_at_radius(r_inf, kind='EMRI')


print("Dehnen number density at influence radius (TDE, 1/pc^3):", dehnen_at_r_influence)
print("Dehnen number density at influence radius (EMRI, 1/pc^3):", dehnen_at_r_influence_emri)

# ------------------------------------------------------------
# Figure setup
# ------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=fig_size(COL_W, 1.85), sharex=True)

colors = plt.cm.magma(np.linspace(0, 0.8, len(lgMBH_mass.squeeze())))
mass_labels = [fr'$10^{{{m:.1f}}}$' for m in lgMBH_mass.squeeze()]

X_LIM = (1e-4, 1e3)

def plot_panel(ax, y_tde, y_emri, title, x, mass_labels=None, legend_mode='none', show_vlines=False):
    ax.set_prop_cycle(color=colors)

    tde_lines = ax.plot(x, y_tde, linestyle='-', lw=1.0)

    ax.set_prop_cycle(color=colors)
    emri_lines = ax.plot(x, y_emri, linestyle='--', lw=1.0)

    if show_vlines:
        vline_handles = []
        for i, x0 in enumerate(r_inf):
            vline_handles.append(ax.axvline(
                x=x0,
                color=colors[i],
                linestyle='-.', lw=0.8
            ))

    # ------------------------------------------------------------
    # Legend: populations (stars vs sBHs)
    # ------------------------------------------------------------
    if legend_mode == 'type':
        leg = ax.legend(
            [tde_lines[0], emri_lines[0]],
            ['stars', 'sBHs'],
            loc='lower left',
            frameon=False
        )
        ax.add_artist(leg)

    # ------------------------------------------------------------
    # Legend: r_inf line + MBH masses (powers of 10)
    # ------------------------------------------------------------
    if legend_mode == 'full':
        leg_rinf = ax.legend(
            [vline_handles[0]],
            [r'$r_{\rm inf}$'],
            loc='lower left',
            frameon=False
        )
        ax.add_artist(leg_rinf)

        mass_handles = [
            plt.Line2D([0], [0], color=colors[i], lw=1.2)
            for i in range(len(mass_labels))
        ]

        ax.legend(
            mass_handles,
            mass_labels,
            title=r'$M_{\rm BH}/M_{\odot}$',
            loc='upper right',
            frameon=True, facecolor='white', framealpha=1.0, edgecolor='none', labelspacing=0.25
        ).set_zorder(10)

    # formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*X_LIM)


# ------------------------------------------------------------
# Panel 1: Number density
# ------------------------------------------------------------
ax = axes[0]

plot_panel(
    ax,
    profile_obj.dehnen_number_density(r_grid, kind='TDE').T,
    profile_obj.dehnen_number_density(r_grid, kind='EMRI').T,
    "Number Density",
    r_grid.T,
    mass_labels=mass_labels,
    legend_mode='full',
    show_vlines=True
)

ax.set_ylabel(r'Number Density (pc$^{-3}$)')
ax.set_ylim(1e-3, 1e16)


# ------------------------------------------------------------
# Panel 2: Radial number distribution
# ------------------------------------------------------------
ax = axes[1]

plot_panel(
    ax,
    profile_obj.radial_number_distribution(r_grid, kind='TDE').T,
    profile_obj.radial_number_distribution(r_grid, kind='EMRI').T,
    "Radial Number Distribution",
    r_grid.T,
    mass_labels=mass_labels,
    legend_mode='type'
)

ax.set_ylabel(r'Radial Number Distribution (pc$^{-1}$)')
ax.relim()
ax.autoscale_view(scalex=False)


# ------------------------------------------------------------
# Panel 3: Cumulative number
# ------------------------------------------------------------
ax = axes[2]

plot_panel(
    ax,
    profile_obj.cumulative_number(r_grid, kind='TDE').T,
    profile_obj.cumulative_number(r_grid, kind='EMRI').T,
    "Cumulative Number",
    r_grid[:, :-1].T,
    mass_labels=mass_labels,
    legend_mode='none'
)

ax.set_ylabel('Cumulative Number')
ax.set_xlabel('Radius (pc)')
ax.set_ylim(1e-2, 1e10)

plt.tight_layout()
savefig_all(fig, 'density_profiles')
plt.close(fig)

mass_density_at_r_influence = profile_obj.mass_density_at_rinfl(r_inf, unit='Msun/pc^3')
print("Mass density at influence radius (Msun/pc^3):", mass_density_at_r_influence)

star_number_density = profile_obj.dehnen_number_density(r_grid, kind='TDE')
sbh_number_density = profile_obj.dehnen_number_density(r_grid, kind='EMRI')

fig2, (ax_mass, ax_num) = plt.subplots(2, 1, figsize=fig_size(COL_W, 1.45), sharex=True)

# ------------------------------------------------------------
# Top panel: mass density
# ------------------------------------------------------------
for i, m in enumerate(mass_labels):
    ax_mass.plot(r_grid[i], profile_obj.mass_density(r_grid).T[:, i], color=colors[i],
                 label=m)
ax_mass.scatter(r_inf, mass_density_at_r_influence, color='red', marker='x', s=25, zorder=5)

for i, x0 in enumerate(r_inf):
    ax_mass.axvline(x=x0, color=colors[i], linestyle='-.',
                     label=r'$r_{\rm inf}$' if i == 0 else None)

ax_mass.set_yscale('log')
ax_mass.set_ylabel(r'Mass Density (M$_\odot$/pc$^3$)')
ax_mass.legend(title=r'$M_{\rm BH}/M_{\odot}$')

# ------------------------------------------------------------
# Bottom panel: number density of stars and sBHs
# ------------------------------------------------------------
for i, m in enumerate(mass_labels):
    ax_num.plot(r_grid[i], star_number_density.T[:, i], color=colors[i], linestyle='-')
    ax_num.plot(r_grid[i], sbh_number_density.T[:, i], color=colors[i], linestyle='--')

for i, x0 in enumerate(r_inf):
    ax_num.axvline(x=x0, color=colors[i], linestyle='-.')

style_leg = ax_num.legend(
    [plt.Line2D([0], [0], color='black', linestyle='-'),
     plt.Line2D([0], [0], color='black', linestyle='--')],
    ['stars', 'sBHs'],
    loc='lower left', frameon=False
)
ax_num.add_artist(style_leg)

mass_handles = [plt.Line2D([0], [0], color=colors[i], lw=1.2) for i in range(len(mass_labels))]
ax_num.legend(mass_handles, mass_labels, title=r'$M_{\rm BH}/M_{\odot}$',
              loc='upper right', frameon=True, facecolor='white', framealpha=1.0, edgecolor='none').set_zorder(10)
ax_num.add_artist(style_leg)

ax_num.set_xscale('log')
ax_num.set_yscale('log')
ax_num.set_xlabel('Radius (pc)')
ax_num.set_ylabel(r'Number Density (pc$^{-3}$)')

ax_num.set_xlim(1e-4, 1e3)

plt.tight_layout()
savefig_all(fig2, 'mass_density_profile')
plt.close(fig2)