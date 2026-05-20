import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('font', family='serif', serif=['Computer Modern'], size=12)
matplotlib.rc('text', usetex=True)

from cosmology import CosmologyModel, LastMajorMerger

# --- Initialize Cosmology and Merger Models ---
cosmo = CosmologyModel(H0=70.0, Om0=0.3, Tcmb0=2.725)
lmm = LastMajorMerger(cosmology_model=cosmo)

# --- Parameters ---
kappa = 0.25       # fraction of relaxation time for cusp regrowth
t_rlx = 1.5        # Gyr, relaxation time

# Sample observation redshifts
z_obs_array = np.array([0.5, 3.0])

# Get cosmic times from cosmology
z = np.linspace(0.01, lmm.z_max, 500)
t_gyr = cosmo.age_Gyr(z)

# Use new separate hazard and survival functions
z_hazard, h_z = lmm.hazard_function(z)
z_survival, S = lmm.survival_function(z)

# Sample LMM for each observation redshift
z_LMM, t_LMM, t_obs = lmm.sample_LMM_times(z_obs_array, size=1)

# Cusp age after each merger (in Gyr)
T_c = []
for i, t_LMM_val in enumerate(t_LMM):
    T = np.maximum(0, t_gyr - (t_LMM_val + kappa * t_rlx))
    T_c.append(T)

# --- Plotting ---
fig, ax1 = plt.subplots(figsize=(14, 7))

# Layer 1: Hazard function (merger rate, normalized)
# h_z is the merger rate λ(z), normalized for visibility on primary y-axis
ax1.fill_between(t_gyr, 0, h_z/np.max(h_z), color='orangered', alpha=0.2, 
                 label=r'Merger Rate $\varphi_{\rm MM}(z)$ (normalized)')
ax1.plot(t_gyr, h_z/np.max(h_z), color='orangered', linewidth=2.5, linestyle='-')

# Layer 2: Survival function
# S(z) = exp(-Λ(z)) is the true survival function (physically accurate)
# Represents probability that NO merger has occurred by redshift z
ax1.plot(t_gyr, S, color='green', linewidth=2.5, linestyle='--', label=r'Survival Function $S(z) = \exp[-\Xi(z)]$')
ax1.set_xlabel(r'Cosmic Time (Gyr)', fontsize=14)
ax1.set_ylabel(r'Normalized Probability', fontsize=14)
ax1.set_ylim(0, 1.05)
ax1.grid(alpha=0.3, linestyle='--', linewidth=0.8)

# Layer 3: Cusp age on secondary Y-axis
ax2 = ax1.twinx()
colors = ['royalblue', 'navy', 'darkblue']
for i, T in enumerate(T_c):
    ax2.plot(t_gyr, T, color=colors[i], linewidth=3.0, 
             label=fr'Cusp Age Sample ($z_{{\rm obs}}={z_obs_array[i]}$)')

ax2.set_ylabel(r'Cusp Age $T_c$ (Gyr)', fontsize=14)
ax2.set_ylim(0, max([T.max() for T in T_c])*1.1)

# LMM event lines and shaded regions
for i, (t_LMM_val, z_LMM_val) in enumerate(zip(t_LMM, z_LMM)):
    
    t_on = t_LMM_val + kappa * t_rlx

    ax2.axvspan(t_LMM_val, t_on, color='lightblue', alpha=0.35, label='Cusp regrowth phase' if i == 0 else None)


    # if i == 0:
    #     ax2.text((t_LMM_val + t_on)/2, ax2.get_ylim()[1]*0.2,
    #             'Regrowth phase\n($T_c = 0$)', ha='center',
    #             fontsize=11, color='black')

    ax2.axvline(t_LMM_val, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    # ax2.text(t_LMM_val, ax2.get_ylim()[1]*0.95, f'$t_{{\\rm LMM}} = {t_LMM_val:.1f}$ Gyr\n($z_{{\\rm LMM}} = {z_LMM_val:.2f}$)',rotation=0, ha='center', fontsize=10, color='gray', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.text(t_LMM_val, ax2.get_ylim()[1]*0.9,fr'$t_{{\rm LMM}}={t_LMM_val:.1f}$ Gyr', ha='center', fontsize=10)

# Annotate linear growth
ax2.text(8, ax2.get_ylim()[1]*0.6,
         r'$T_c = t - t_{\rm on}$'+'\nLinear growth',
         fontsize=12, color='black',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


# Legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=14, 
           frameon=False)

plt.xlim(t_gyr.min(), t_gyr.max())


plt.tight_layout()
plt.savefig('./data/astrophysical_data/NSC_cusp_evolution_timeline.png', dpi=300, bbox_inches='tight')
plt.close()