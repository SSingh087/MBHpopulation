# python -m cProfile -o output.prof test_galaxies.py
# snakeviz output.prof

import os, sys, argparse
sys.path.insert(0, os.path.abspath('./astrophysical_setup'))
import argparse

import numpy as np
from galaxy import Galaxy
from nsc import NSC, CompactObject, MBH_properties
from density import DehnenProfile
from relaxation import RelaxationModel
from rate import RateModel
from evolution import CuspEvolution
from cosmology import LastMajorMerger, CosmologyModel, GalaxyStellarMassFunction

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='serif', serif=['Computer Modern'], size=15)
matplotlib.rc('text', usetex=True)
import seaborn 

import h5py

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES", type=int, required=True, help="Number of galaxies")
parser.add_argument("--OBSERVING_WINDOW", type=float, required=True, help="Observing window in years")
args = parser.parse_args()

cosmo_model = CosmologyModel()

N_objs = args.GALAXIES
T_obs = args.OBSERVING_WINDOW
z_grid = np.random.uniform(1E-5, 8.0, N_objs)  # this is N_objs redshifts 

GSMF = GalaxyStellarMassFunction()
lgMgal_samples = GSMF.sample_gsmf(z_gal=z_grid, size=N_objs) # for each redshits we get N_obj galaxies 
# hence at this stage we have N_obj x N_obj galaxies

nucleation_indices = Galaxy.check_nucleation(lgMgal_samples, z_grid)
 
gal_obj = Galaxy(lgMgal=lgMgal_samples, z_gal=z_grid, nucleation_occurs=nucleation_indices)

NSC_obj = NSC(gal_obj)

MBH_obj = MBH_properties(nsc=NSC_obj)

CO_objs = CompactObject(nsc=NSC_obj, masses={'sBH': 10.0, 'star': 1.0}, total_mass={'sBH': 20.0, 'star': 100.0}, types_CO=['sBH', 'star'], types_masses='same_mass', type_CO_limits=None)

dehnen_obj = DehnenProfile(nsc=NSC_obj, compact_object=CO_objs)

relax_obj = RelaxationModel(nsc=NSC_obj, compact_object=CO_objs, profile=dehnen_obj)

rate_obj = RateModel(nsc=NSC_obj)

cusp_evolution_object = CuspEvolution(nsc=NSC_obj, compact_object=CO_objs, relaxation=relax_obj, rate_model=rate_obj, LastMajorMerger=LastMajorMerger(cosmo_model))

observed_EMRIs = cusp_evolution_object.number_of_objects_in_time(T_obs=T_obs, kvir=1.0, kind='EMRI', unit='Gyr')
observed_TDEs = cusp_evolution_object.number_of_objects_in_time(T_obs=T_obs, kvir=1.0, kind='TDE', unit='Gyr')

print(np.sum(observed_EMRIs), observed_EMRIs.max(), np.sum(observed_TDEs), observed_TDEs.max())

loc = f'/data/wiay/postgrads/shashwat/EMRI_TDE_data/astrophysical_data/{args.GALAXIES}'
if not os.path.exists(loc):
    os.makedirs(loc)

hf = h5py.File(f'{loc}/data_cusp_evolution.h5', 'w')

# apart from nucleation fraction check we also need to apply check on the S/MBH mass 
# since we are dealing with S/MBHs, the masses should be greater than 10^4

# Filter galaxies based on MBH mass
# these massive black holes are source frame
mbh_mask = (gal_obj.lgMBH_mass >= 4)

hf.create_dataset('lgMgal', data=lgMgal_samples[nucleation_indices][mbh_mask]) # apply both nucleation and MBH mass filters
hf.create_dataset('sigma_km_s', data=gal_obj.sigma_km_s[mbh_mask]) # nucleation_index filter already applied

# we don't need to save nucleation indices since we are already filtering based on the MBH mass

hf.create_dataset('z_gal', data=z_grid[nucleation_indices][mbh_mask]) 
sky_locs = gal_obj.sky_location()[nucleation_indices][mbh_mask]
ra, dec = sky_locs[:, 0], sky_locs[:, 1]

hf.create_dataset('ra_deg', data=np.array(ra))
hf.create_dataset('dec_deg', data=np.array(dec))

hf.create_dataset('lgMBH', data=gal_obj.lgMBH_mass[mbh_mask]) # nucleation_index filter already applied
hf.create_dataset('MBHspin', data=MBH_obj.MBHspin[mbh_mask]) # nucleation_index filter already applied

# since this is the same mass case so we just save the scalar values hence no masking is required
hf.create_dataset('sBH_masses', data=CO_objs.masses['sBH'])
hf.create_dataset('star_masses', data=CO_objs.masses['star'])

hf.create_dataset('observed_EMRIs', data=observed_EMRIs[mbh_mask]) # nucleation_index filter already applied
hf.create_dataset('observed_TDEs', data=observed_TDEs[mbh_mask]) # nucleation_index filter already applied
hf.close()

print("After MBH mass filter")
print(np.sum(observed_EMRIs[mbh_mask]), observed_EMRIs[mbh_mask].max(), np.sum(observed_TDEs[mbh_mask]), observed_TDEs[mbh_mask].max())

# import matplotlib.colors as mcolors

# fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

# # --- Color normalization across full z range ---
# norm = mcolors.Normalize(vmin=np.min(z_grid), vmax=np.max(z_grid))
# colors = z_grid[nucleation_indices][mbh_mask]

# # --- Top: EMRIs ---
# sc = axes[0].scatter(gal_obj.lgMBH_mass[mbh_mask], observed_EMRIs[mbh_mask], c=colors, cmap='plasma', norm=norm,
#                     marker='o', alpha=0.9, s=20)

# axes[0].set_title(f'EMRIs = {np.sum(observed_EMRIs[mbh_mask])}')
# axes[0].set_yscale('log')
# axes[0].set_ylabel(f'Number of events in {T_obs} yrs')

# axes[1].scatter(gal_obj.lgMBH_mass[mbh_mask], observed_TDEs[mbh_mask], c=colors, cmap='plasma', norm=norm,
#                 marker='d', alpha=0.9, s=20)

# axes[1].set_title(f'TDEs = {np.sum(observed_TDEs[mbh_mask])}')
# axes[1].set_yscale('log')
# axes[1].set_xlabel(r'$\log_{10}(M_{\mathrm{MBH}} / M_\odot)$')
# axes[1].set_ylabel(f'Number of events in {T_obs} yrs')

# # --- Shared colorbar ---
# cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
# fig.colorbar(sc, cax=cbar_ax, label='Redshift')

# plt.tight_layout(rect=[0, 0, 0.9, 1])
# plt.savefig(f'{loc}/observed_objects.png', dpi=300)
# plt.close()


# # Convert spherical coordinates (RA, Dec, z) to Cartesian for true 3D shells
# # RA in radians, Dec in radians
# ra_rad = np.radians(ra)
# dec_rad = np.radians(dec)
# z_filtered = z_grid[nucleation_indices][mbh_mask]  # for color coding in polar plots

# # Cartesian coordinates
# x = z_filtered * np.cos(dec_rad) * np.cos(ra_rad)
# y = z_filtered * np.cos(dec_rad) * np.sin(ra_rad)
# z = z_filtered * np.sin(dec_rad)

# # --- Polar plot RA vs z ---
# plt.figure(figsize=(8,6), facecolor='white')
# ax = plt.subplot(projection='polar')
# ax.set_facecolor('#f9f9f9')  # light background
# ax.scatter(ra_rad, z_filtered, c=colors, cmap='plasma', marker='o', s=20, alpha=0.6)
# ax.set_rlabel_position(240)  # radial labels at bottom
# ax.grid(True, color='gray', linestyle='--', alpha=0.3)
# plt.title('Galaxy Distribution: RA vs Redshift', fontsize=14)
# plt.savefig(f'{loc}/RA_vs_redshift.png', dpi=300)
# # plt.show()
# plt.close()

# # --- Polar plot Dec vs z ---
# # Since Dec is not circular, we shift it to 0-360 deg for polar visualization
# dec_shifted = dec + 90  # from [-90,90] -> [0,180]
# plt.figure(figsize=(8,6), facecolor='white')
# ax = plt.subplot(projection='polar')
# ax.set_facecolor('#f9f9f9')
# ax.scatter(np.radians(dec_shifted), z_filtered, c=colors, cmap='plasma', marker='o', s=20, alpha=0.6)
# ax.set_rlabel_position(240)  # radial labels at bottom
# ax.grid(True, color='gray', linestyle='--', alpha=0.3)
# plt.title('Galaxy Distribution: Dec vs Redshift', fontsize=14)
# plt.savefig(f'{loc}/Dec_vs_redshift.png', dpi=300)
# # plt.show()
# plt.close()


# # 3D scatter plot with concentric shells
# fig = plt.figure(figsize=(10,8), facecolor='white')
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(x, y, z, c=z_filtered, s=15, cmap='plasma')
# cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
# cbar.set_label('Redshift', fontsize=12)
# ax.set_xlabel('X [$z$]')
# ax.set_ylabel('Y [$z$]')
# ax.set_zlabel('Z [$z$]')
# ax.set_title('3D Galaxy Distribution (Concentric Shells)', fontsize=14)
# ax.grid(False)
# ax.set_box_aspect([1,1,1])

# plt.savefig(f'{loc}/3D_galaxy_distribution.png', dpi=300)
# # plt.show()


# mask = z >= 0
# x_half = x[mask]
# y_half = y[mask]
# z_half = z[mask]
# z_filtered_half = z_filtered[mask]  # assuming you color by z_filtered

# # 3D scatter plot for half-sphere
# fig = plt.figure(figsize=(10,8), facecolor='white')
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(x_half, y_half, z_half, c=z_filtered_half, s=15, cmap='plasma')
# cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
# cbar.set_label('Redshift', fontsize=12)
# ax.set_xlabel('X [$z$]')
# ax.set_ylabel('Y [$z$]')
# ax.set_zlabel('Z [$z$]')
# ax.set_title('Half 3D Galaxy Distribution', fontsize=14)
# ax.grid(False)
# ax.set_box_aspect([1,1,1])

# plt.savefig(f'{loc}/Half_3D_galaxy_distribution.png', dpi=300)
# # plt.show()