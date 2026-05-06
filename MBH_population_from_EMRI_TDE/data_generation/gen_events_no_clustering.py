# python -m cProfile -o output_no_clustering.prof test_galaxies.py
# snakeviz output_no_clustering.prof

import os, sys, argparse
sys.path.insert(0, os.path.abspath('./astrophysical_setup'))
sys.path.insert(0, os.path.abspath('./'))
import argparse

import numpy as np
from galaxy import Galaxy
from nsc import NSC, CompactObject, MBH_properties
from density import DehnenProfile
from relaxation import RelaxationModel
from rate import RateModel
from evolution import CuspEvolution
from cosmology import LastMajorMerger, CosmologyModel, GalaxyStellarMassFunction
from utils import Plotting
import matplotlib.pyplot as plt
import matplotlib
import h5py

rng = np.random.default_rng(seed=42)
Z_MAX = 10.0

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES", type=int, required=True, help="Number of galaxies")
parser.add_argument("--OBSERVING_WINDOW", type=float, required=True, help="Observing window in years")
parser.add_argument("--Z_MAX", type=float, required=False, default=10.0, help="Maximum redshift")
args = parser.parse_args()

cosmo_model = CosmologyModel()
Z_MAX = args.Z_MAX

N_objs = args.GALAXIES
T_obs = args.OBSERVING_WINDOW


# Rmax = cosmo_model.comoving_distance(Z_MAX)
# Rmin = cosmo_model.comoving_distance(1e-56)
# u = rng.random(N_objs)
# r = (Rmin**3 + (Rmax**3 - Rmin**3) * u)**(1/3)

# # Ensure monotonic inversion behaves well
# r.sort()

# z_grid = cosmo_model.z_from_comoving_distance(r)

from scipy.optimize import brentq

Rmin = cosmo_model.comoving_distance(1e-5)
Rmax = cosmo_model.comoving_distance(Z_MAX)

rng = np.random.default_rng()
u = rng.random(N_objs)
r_samples = (Rmin**3 + (Rmax**3 - Rmin**3) * u)**(1/3)

# Function to invert comoving distance
def z_from_r(r):
    return brentq(lambda z: cosmo_model.comoving_distance(z) - r, 1e-5, Z_MAX)

# Vectorized inversion
z_grid = np.array([z_from_r(r) for r in r_samples])

# z_grid = np.random.uniform(1E-5, Z_MAX, N_objs)  # this is N_objs redshifts 

GSMF = GalaxyStellarMassFunction()
lgMgal_samples = GSMF.sample_gsmf(z_gal=z_grid, size=N_objs)

nucleation_indices = Galaxy.check_nucleation(lgMgal_samples, z_grid)
 
gal_obj = Galaxy(lgMgal=lgMgal_samples, z_gal=z_grid, nucleation_occurs=nucleation_indices)

NSC_obj = NSC(gal_obj)

MBH_obj = MBH_properties(nsc=NSC_obj)

CO_objs = CompactObject(nsc=NSC_obj, masses={'sBH': 10.0, 'star': 1.0}, total_mass={'sBH': 20.0, 'star': 100.0}, types_CO=['sBH', 'star'], types_masses='same_mass', type_CO_limits=None)

dehnen_obj = DehnenProfile(nsc=NSC_obj, compact_object=CO_objs)

relax_obj = RelaxationModel(nsc=NSC_obj, compact_object=CO_objs, profile=dehnen_obj)

rate_obj = RateModel(nsc=NSC_obj)

cusp_evolution_object = CuspEvolution(nsc=NSC_obj, compact_object=CO_objs, relaxation=relax_obj, rate_model=rate_obj, LastMajorMerger=LastMajorMerger(cosmo_model))

generated_EMRIs = cusp_evolution_object.number_of_objects_in_time(T_obs=T_obs, kvir=1.0, kind='EMRI', unit='Gyr')
generated_TDEs = cusp_evolution_object.number_of_objects_in_time(T_obs=T_obs, kvir=1.0, kind='TDE', unit='Gyr')

print(np.sum(generated_EMRIs), generated_EMRIs.max(), np.sum(generated_TDEs), generated_TDEs.max())

loc = f'/data/wiay/postgrads/shashwat/EMRI_TDE_data/astrophysical_data/{args.GALAXIES}'
if not os.path.exists(loc):
    os.makedirs(loc)

hf = h5py.File(f'{loc}/data_cusp_evolution_no_clustering.h5', 'w')

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
hf.create_dataset('initial_MBHspin', data=MBH_obj.initial_MBHspin[mbh_mask]) # nucleation_index filter already applied

# since this is the same mass case so we just save the scalar values hence no masking is required
hf.create_dataset('sBH_masses', data=CO_objs.masses['sBH'])
hf.create_dataset('star_masses', data=CO_objs.masses['star'])

hf.create_dataset('generated_EMRIs', data=generated_EMRIs[mbh_mask]) # nucleation_index filter already applied
hf.create_dataset('generated_TDEs', data=generated_TDEs[mbh_mask]) # nucleation_index filter already applied
hf.close()

print("After MBH mass filter")
print(np.sum(generated_EMRIs[mbh_mask]), generated_EMRIs[mbh_mask].max(), np.sum(generated_TDEs[mbh_mask]), generated_TDEs[mbh_mask].max())

# ========================
# PLOTTING 
# ========================

Plotting.MBHmass_vs_spin(gal_obj.lgMBH_mass[mbh_mask], MBH_obj.initial_MBHspin[mbh_mask], z_grid[nucleation_indices][mbh_mask], loc=f'{loc}/MBHmass_vs_spin_no_clustering.png', cmap='plasma')

import matplotlib.colors as mcolors

fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

# --- Color normalization across full z range ---
norm = mcolors.Normalize(vmin=np.min(z_grid), vmax=np.max(z_grid))
colors = z_grid[nucleation_indices][mbh_mask]

# --- Top: EMRIs ---
sc = axes[0].scatter(gal_obj.lgMBH_mass[mbh_mask], generated_EMRIs[mbh_mask], c=colors, cmap='plasma', norm=norm,
                    marker='o', alpha=0.9, s=20)

axes[0].set_title(f'EMRIs = {np.sum(generated_EMRIs[mbh_mask])}')
axes[0].set_yscale('log')
axes[0].set_ylabel(f'Number of events in {T_obs} yrs')

axes[1].scatter(gal_obj.lgMBH_mass[mbh_mask], generated_TDEs[mbh_mask], c=colors, cmap='plasma', norm=norm,
                marker='d', alpha=0.9, s=20)

axes[1].set_title(f'TDEs = {np.sum(generated_TDEs[mbh_mask])}')
axes[1].set_yscale('log')
axes[1].set_xlabel(r'$\log_{10}(M_{\mathrm{MBH}} / M_\odot)$')
axes[1].set_ylabel(f'Number of events in {T_obs} yrs')

# --- Shared colorbar ---
cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
fig.colorbar(sc, cax=cbar_ax, label='Redshift')

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig(f'{loc}/generated_objects_no_clustering.png', dpi=300)
plt.close()

fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

mass_bins = np.linspace(np.min(gal_obj.lgMBH_mass[mbh_mask]), 
                        np.max(gal_obj.lgMBH_mass[mbh_mask]), 100)
z_bins = np.linspace(np.min(z_grid[nucleation_indices][mbh_mask]), 
                     np.max(z_grid[nucleation_indices][mbh_mask]), 100)

# --- Top: EMRIs ---
H_emri, xedges, yedges = np.histogram2d(
    gal_obj.lgMBH_mass[mbh_mask],
    z_grid[nucleation_indices][mbh_mask],
    bins=[mass_bins, z_bins],
    weights=generated_EMRIs[mbh_mask],
)

# Optionally smooth
# H_emri = gaussian_filter(H_emri, sigma=1.5)

X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
pcm = axes[0].pcolormesh(X, Y, H_emri, cmap='plasma', shading='auto')

axes[0].set_title(f'EMRIs = {np.sum(generated_EMRIs[mbh_mask])}')
axes[0].set_ylabel('Redshift (z)')

# Colorbar for EMRIs
cbar_emri = fig.colorbar(pcm, ax=axes[0], label='Number of EMRI events')

# --- Bottom: TDEs ---
H_tde, _, _ = np.histogram2d(
    gal_obj.lgMBH_mass[mbh_mask],
    z_grid[nucleation_indices][mbh_mask],
    bins=[mass_bins, z_bins],
    weights=generated_TDEs[mbh_mask]
)
# H_tde = gaussian_filter(H_tde, sigma=1.5)

pcm2 = axes[1].pcolormesh(X, Y, H_tde, cmap='plasma', shading='auto')

axes[1].set_title(f'TDEs = {np.sum(generated_TDEs[mbh_mask])}')
axes[1].set_xlabel(r'$\log_{10}(M_{\mathrm{MBH}} / M_\odot)$')
axes[1].set_ylabel('Redshift (z)')

# Colorbar for TDEs
cbar_tde = fig.colorbar(pcm2, ax=axes[1], label='Number of TDE events')

plt.tight_layout()
plt.savefig(f'{loc}/generated_objects_no_clustering_smooth.png', dpi=300)
plt.close()

# Convert spherical coordinates (RA, Dec, z) to Cartesian for true 3D shells
# RA in radians, Dec in radians
ra_rad = np.radians(ra)
dec_rad = np.radians(dec)
z_filtered = z_grid[nucleation_indices][mbh_mask]  # for color coding in polar plots

# Cartesian coordinates
x = z_filtered * np.cos(dec_rad) * np.cos(ra_rad)
y = z_filtered * np.cos(dec_rad) * np.sin(ra_rad)
z = z_filtered * np.sin(dec_rad)

# --- Polar plot RA vs z ---
plt.figure(figsize=(8,6), facecolor='white')
ax = plt.subplot(projection='polar')
ax.set_facecolor('#f9f9f9')  # light background
ax.scatter(ra_rad, z_filtered, c=colors, cmap='plasma', marker='o', s=20, alpha=0.6)
ax.set_rlabel_position(240)  # radial labels at bottom
ax.grid(True, color='gray', linestyle='--', alpha=0.3)
plt.title('Galaxy Distribution: RA vs Redshift', fontsize=14)
plt.savefig(f'{loc}/RA_vs_redshift_no_clustering.png', dpi=300)
# plt.show()
plt.close()

# --- Polar plot Dec vs z ---
# Since Dec is not circular, we shift it to 0-360 deg for polar visualization
dec_shifted = dec + 90  # from [-90,90] -> [0,180]
plt.figure(figsize=(8,6), facecolor='white')
ax = plt.subplot(projection='polar')
ax.set_facecolor('#f9f9f9')
ax.scatter(np.radians(dec_shifted), z_filtered, c=colors, cmap='plasma', marker='o', s=20, alpha=0.6)
ax.set_rlabel_position(240)  # radial labels at bottom
ax.grid(True, color='gray', linestyle='--', alpha=0.3)
plt.title('Galaxy Distribution: Dec vs Redshift', fontsize=14)
plt.savefig(f'{loc}/Dec_vs_redshift_no_clustering.png', dpi=300)
# plt.show()
plt.close()


# 3D scatter plot with concentric shells
fig = plt.figure(figsize=(10,8), facecolor='white')
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z, c=z_filtered, s=15, cmap='plasma')
cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
cbar.set_label('Redshift', fontsize=12)
ax.set_xlabel('X [$z$]')
ax.set_ylabel('Y [$z$]')
ax.set_zlabel('Z [$z$]')
ax.set_title('3D Galaxy Distribution (Concentric Shells)', fontsize=14)
ax.grid(False)
ax.set_box_aspect([1,1,1])

plt.savefig(f'{loc}/3D_galaxy_distribution_no_clustering.png', dpi=300)
# plt.show()


# zmin, zmax_slice = 0.8, 2.0  # THIN slice
# mask = (z_grid[nucleation_indices][mbh_mask] > zmin) & (z_grid[nucleation_indices][mbh_mask] < zmax_slice)

# ra_rad  = np.deg2rad(ra)
# dec_rad = np.deg2rad(dec)
# ra_wrap = (ra_rad + np.pi) % (2*np.pi) - np.pi

# fig = plt.figure(figsize=(8,4.5))
# ax = fig.add_subplot(111, projection="mollweide")
# ax.scatter(-ra_wrap, dec_rad, s=5, alpha=0.4, rasterized=True)  # minus flips RA like sky maps
# ax.grid(True, alpha=0.3)
# ax.set_title(f"${zmin} \leq z \leq {zmax_slice}$")
# plt.tight_layout()
# plt.savefig(f'{loc}/mollweide_sky_map_{zmin}_{zmax_slice}_no_clustering.png', dpi=300)
# plt.close()