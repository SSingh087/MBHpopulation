import sys, os
sys.path.insert(0, os.path.abspath('./'))

import argparse
import h5py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
from utils import Plotting

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES", type=int, required=True, help="Number of galaxies")
parser.add_argument("--OBSERVING_WINDOW", type=float, required=True, help="Observing window in years")
parser.add_argument("--Z_MAX", type=float, required=False, default=10.0, help="Maximum redshift")
args = parser.parse_args()

loc = f'./data/astrophysical_data/{args.GALAXIES}'

with h5py.File(f'{loc}/data_cusp_evolution_no_clustering.h5', 'r') as f:
    lgMgal_no_clustering = f['lgMgal'][:]
    sigma_km_s_no_clustering = f['sigma_km_s'][:]
    z_gal_no_clustering = f['z_gal'][:]
    ra_deg_no_clustering = f['ra_deg'][:]
    dec_deg_no_clustering = f['dec_deg'][:]
    lgMBH_no_clustering = f['lgMBH'][:]
    initial_MBHspin_no_clustering = f['initial_MBHspin'][:]
    # sBH_masses_no_clustering = f['sBH_masses'][:]
    # star_masses_no_clustering = f['star_masses'][:]
    generated_EMRIs_no_clustering = f['generated_EMRIs'][:]
    generated_TDEs_no_clustering = f['generated_TDEs'][:]
    
    # Convert spherical coordinates (RA, Dec, z) to Cartesian for true 3D shells
    # RA in radians, Dec in radians
    ra_rad_no_clustering = np.deg2rad(f['ra_deg'][:])
    dec_rad_no_clustering = np.deg2rad(f['dec_deg'][:])
    
    # Cartesian coordinates
    x_no_clustering = z_gal_no_clustering * np.cos(dec_rad_no_clustering) * np.cos(ra_rad_no_clustering)
    y_no_clustering = z_gal_no_clustering * np.cos(dec_rad_no_clustering) * np.sin(ra_rad_no_clustering)
    z_no_clustering = z_gal_no_clustering * np.sin(dec_rad_no_clustering)

with h5py.File(f'{loc}/data_cusp_evolution_clustering.h5', 'r') as f:
    lgMgal_clustering = f['lgMgal'][:]
    sigma_km_s_clustering = f['sigma_km_s'][:]
    z_gal_clustering = f['z_gal'][:]
    ra_deg_clustering = f['ra_deg'][:]
    dec_deg_clustering = f['dec_deg'][:]
    lgMBH_clustering = f['lgMBH'][:]
    initial_MBHspin_clustering = f['initial_MBHspin'][:]
    # sBH_masses_clustering = f['sBH_masses'][:]
    # star_masses_clustering = f['star_masses'][:]
    generated_EMRIs_clustering = f['generated_EMRIs'][:]
    generated_TDEs_clustering = f['generated_TDEs'][:]
    
    # Convert spherical coordinates (RA, Dec, z) to Cartesian for true 3D shells
    # RA in radians, Dec in radians

    ra_rad_clustering = np.deg2rad(f['ra_deg'][:])
    dec_rad_clustering = np.deg2rad(f['dec_deg'][:])

    # Cartesian coordinates
    x_clustering = z_gal_clustering * np.cos(dec_rad_clustering) * np.cos(ra_rad_clustering)
    y_clustering = z_gal_clustering * np.cos(dec_rad_clustering) * np.sin(ra_rad_clustering)
    z_clustering = z_gal_clustering * np.sin(dec_rad_clustering)


# ====================
# MBH MASS VS SPIN
# ====================

Plotting.MBHmass_vs_spin(lgMBH_no_clustering, initial_MBHspin_no_clustering, z_gal_no_clustering, loc=f'{loc}/MBHmass_vs_spin_no_clustering.png', cmap='plasma')

Plotting.MBHmass_vs_spin(lgMBH_clustering, initial_MBHspin_clustering, z_gal_clustering, loc=f'{loc}/MBHmass_vs_spin_clustering.png', cmap='plasma')

# ====================
# EVENT RATES
# ====================


Plotting.plot_event_rates(z_gal_no_clustering, lgMBH_no_clustering, generated_EMRIs_no_clustering, generated_TDEs_no_clustering, args.OBSERVING_WINDOW, filename=f'{loc}/event_rates_no_clustering.png')

Plotting.plot_event_rates(z_gal_clustering, lgMBH_clustering, generated_EMRIs_clustering, generated_TDEs_clustering, args.OBSERVING_WINDOW, filename=f'{loc}/event_rates_clustering.png')

# ====================
# EVENT RATES HIST 2D
# ====================


Plotting.plot_event_rates_hist_2d(lgMBH_no_clustering, z_gal_no_clustering, generated_EMRIs_no_clustering, generated_TDEs_no_clustering, filename=f'{loc}/event_rates_hist_2d_no_clustering.png')

Plotting.plot_event_rates_hist_2d(lgMBH_clustering, z_gal_clustering, generated_EMRIs_clustering, generated_TDEs_clustering, filename=f'{loc}/event_rates_hist_2d_clustering.png')


# ====================
# POLAR RA vs DEC
# ====================

Plotting.plot_polar_ra_dec(ra_deg_no_clustering, dec_deg_no_clustering, z_gal_no_clustering, title="RA vs Dec (No Clustering)", filename=f'{loc}/polar_ra_dec_no_clustering.png', cmap='plasma')

Plotting.plot_polar_ra_dec(ra_deg_clustering, dec_deg_clustering, z_gal_clustering, title="RA vs Dec (Clustering)", filename=f'{loc}/polar_ra_dec_clustering.png', cmap='plasma')


# ====================
# 3D GALAXY DISTRIBUTION
# ====================

Plotting.plot_3D_galaxy_distribution(x_no_clustering, y_no_clustering, z_no_clustering, z_gal_no_clustering, filename=f'{loc}/3D_galaxy_distribution_no_clustering.png')

Plotting.plot_3D_galaxy_distribution(x_clustering, y_clustering, z_clustering, z_gal_clustering, filename=f'{loc}/3D_galaxy_distribution_clustering.png')


zmin_slice = 0.5
zmax_slice = 1.0

# ====================
# MOLLWEIDE PROJECTION
# ====================

Plotting.mollweide_sky_map(ra_deg_no_clustering, dec_deg_no_clustering, z_gal_no_clustering, zmin_slice=zmin_slice, zmax_slice=zmax_slice, filename=f'{loc}/mollweide_sky_map_no_clustering.png')

Plotting.mollweide_sky_map(ra_deg_clustering, dec_deg_clustering, z_gal_clustering, zmin_slice=zmin_slice, zmax_slice=zmax_slice, filename=f'{loc}/mollweide_sky_map_clustering.png')

# ====================
# ANGULAR DENSITY SLICE
# ====================

Plotting.angular_density_slice(zmin_slice=zmin_slice, zmax_slice=zmax_slice, ra_deg=ra_deg_clustering, dec_deg=dec_deg_clustering, z_gal=z_gal_clustering, filename=f'{loc}/angular_density_slice_clustering.png')


# ==========================
# EVENT RATES HIST 2D WATERFALL
# ==========================

Plotting.plot_event_rates_hist_2d_waterfall(lgMBH_no_clustering, z_gal_no_clustering, generated_EMRIs_no_clustering, generated_TDEs_no_clustering, args.OBSERVING_WINDOW, filename=f'{loc}/event_rates_hist_2d_waterfall_no_clustering.png')

Plotting.plot_event_rates_hist_2d_waterfall(lgMBH_clustering, z_gal_clustering, generated_EMRIs_clustering, generated_TDEs_clustering, args.OBSERVING_WINDOW, filename=f'{loc}/event_rates_hist_2d_waterfall_clustering.png')


# ==========================
# MBH VS WATERFALL
# ==========================

Plotting.plot_MBH_vs_waterfall(lgMBH_no_clustering, z_gal_no_clustering, generated_EMRIs_no_clustering, generated_TDEs_no_clustering, args.OBSERVING_WINDOW, filename=f'{loc}/MBH_vs_waterfall_no_clustering.png')

Plotting.plot_MBH_vs_waterfall(lgMBH_clustering, z_gal_clustering, generated_EMRIs_clustering, generated_TDEs_clustering, args.OBSERVING_WINDOW, filename=f'{loc}/MBH_vs_waterfall_clustering.png')
