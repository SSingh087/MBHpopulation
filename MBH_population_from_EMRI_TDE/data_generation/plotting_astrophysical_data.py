# import sys, os
# sys.path.insert(0, os.path.abspath('./'))

# import argparse
# import h5py

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
# from scipy.ndimage import gaussian_filter
# from utils import Plotting

# parser = argparse.ArgumentParser()
# parser.add_argument("--GALAXIES", type=int, required=True, help="Number of galaxies")
# parser.add_argument("--OBSERVING_WINDOW", type=float, required=True, help="Observing window in years")
# parser.add_argument("--Z_MAX", type=float, required=False, default=10.0, help="Maximum redshift")
# args = parser.parse_args()

# loc = f'./data/astrophysical_data/{args.GALAXIES}'

# with h5py.File(f'{loc}/data_cusp_evolution_no_clustering.h5', 'r') as f:
#     lgMgal_no_clustering = f['lgMgal'][:]
#     sigma_km_s_no_clustering = f['sigma_km_s'][:]
#     z_gal_no_clustering = f['z_gal'][:]
#     ra_deg_no_clustering = f['ra_deg'][:]
#     dec_deg_no_clustering = f['dec_deg'][:]
#     lgMBH_no_clustering = f['lgMBH'][:]
#     initial_MBHspin_no_clustering = f['initial_MBHspin'][:]
#     eccentricities_at_plunge_no_clustering = f['eccentricities_at_plunge'][:]
#     # sBH_masses_no_clustering = f['sBH_masses'][:]
#     # star_masses_no_clustering = f['star_masses'][:]
#     generated_EMRIs_no_clustering = f['generated_EMRIs'][:]
#     generated_TDEs_no_clustering = f['generated_TDEs'][:]
    
#     # Convert spherical coordinates (RA, Dec, z) to Cartesian for true 3D shells
#     # RA in radians, Dec in radians
#     ra_rad_no_clustering = np.deg2rad(f['ra_deg'][:])
#     dec_rad_no_clustering = np.deg2rad(f['dec_deg'][:])
    
#     # Cartesian coordinates
#     x_no_clustering = z_gal_no_clustering * np.cos(dec_rad_no_clustering) * np.cos(ra_rad_no_clustering)
#     y_no_clustering = z_gal_no_clustering * np.cos(dec_rad_no_clustering) * np.sin(ra_rad_no_clustering)
#     z_no_clustering = z_gal_no_clustering * np.sin(dec_rad_no_clustering)

# # with h5py.File(f'{loc}/data_cusp_evolution_clustering.h5', 'r') as f:
# #     lgMgal_clustering = f['lgMgal'][:]
# #     sigma_km_s_clustering = f['sigma_km_s'][:]
# #     z_gal_clustering = f['z_gal'][:]
# #     ra_deg_clustering = f['ra_deg'][:]
# #     dec_deg_clustering = f['dec_deg'][:]
# #     lgMBH_clustering = f['lgMBH'][:]
# #     initial_MBHspin_clustering = f['initial_MBHspin'][:]
# #     # sBH_masses_clustering = f['sBH_masses'][:]
# #     # star_masses_clustering = f['star_masses'][:]
# #     generated_EMRIs_clustering = f['generated_EMRIs'][:]
# #     generated_TDEs_clustering = f['generated_TDEs'][:]
    
# #     # Convert spherical coordinates (RA, Dec, z) to Cartesian for true 3D shells
# #     # RA in radians, Dec in radians

# #     ra_rad_clustering = np.deg2rad(f['ra_deg'][:])
# #     dec_rad_clustering = np.deg2rad(f['dec_deg'][:])

# #     # Cartesian coordinates
# #     x_clustering = z_gal_clustering * np.cos(dec_rad_clustering) * np.cos(ra_rad_clustering)
# #     y_clustering = z_gal_clustering * np.cos(dec_rad_clustering) * np.sin(ra_rad_clustering)
# #     z_clustering = z_gal_clustering * np.sin(dec_rad_clustering)


# # ====================
# # MBH MASS VS SPIN
# # ====================

# Plotting.MBHmass_vs_spin(lgMBH_no_clustering, initial_MBHspin_no_clustering, z_gal_no_clustering, loc=f'{loc}/MBHmass_vs_spin_no_clustering.png', cmap='plasma')
    
# # Plotting.MBHmass_vs_spin(lgMBH_clustering, initial_MBHspin_clustering, z_gal_clustering, loc=f'{loc}/MBHmass_vs_spin_clustering.png', cmap='plasma')

# # ====================
# # MBH MASS VS ECCENTRICITY AT PLUNGE
# # ====================

# Plotting.MBHmass_vs_eccentricity_at_plunge(lgMBH_no_clustering, eccentricities_at_plunge_no_clustering, z_gal_no_clustering, loc=f'{loc}/MBHmass_vs_eccentricity_at_plunge_no_clustering.png', cmap='plasma')

# # Plotting.MBHmass_vs_eccentricity_at_plunge(lgMBH_clustering, eccentricities_at_plunge_clustering, z_gal_clustering, loc=f'{loc}/MBHmass_vs_eccentricity_at_plunge_clustering.png', cmap='plasma')


# # ====================
# # EVENT RATES
# # ====================

# Plotting.plot_event_rates(z_gal_no_clustering, lgMBH_no_clustering, generated_EMRIs_no_clustering, generated_TDEs_no_clustering, args.OBSERVING_WINDOW, filename=f'{loc}/event_rates_no_clustering.png')

# # Plotting.plot_event_rates(z_gal_clustering, lgMBH_clustering, generated_EMRIs_clustering, generated_TDEs_clustering, args.OBSERVING_WINDOW, filename=f'{loc}/event_rates_clustering.png')

# # ====================
# # EVENT RATES HIST 2D
# # ====================


# # Plotting.plot_event_rates_hist_2d(lgMBH_no_clustering, z_gal_no_clustering, generated_EMRIs_no_clustering, generated_TDEs_no_clustering, filename=f'{loc}/event_rates_hist_2d_no_clustering.png')

# # Plotting.plot_event_rates_hist_2d(lgMBH_clustering, z_gal_clustering, generated_EMRIs_clustering, generated_TDEs_clustering, filename=f'{loc}/event_rates_hist_2d_clustering.png')


# # ====================
# # POLAR RA vs DEC
# # ====================

# Plotting.plot_polar_ra_dec(ra_deg_no_clustering, dec_deg_no_clustering, z_gal_no_clustering, title="RA vs Dec (No Clustering)", filename=f'{loc}/polar_ra_dec_no_clustering.png', cmap='plasma')

# # Plotting.plot_polar_ra_dec(ra_deg_clustering, dec_deg_clustering, z_gal_clustering, title="RA vs Dec (Clustering)", filename=f'{loc}/polar_ra_dec_clustering.png', cmap='plasma')


# # ====================
# # 3D GALAXY DISTRIBUTION
# # ====================

# # Plotting.plot_3D_galaxy_distribution(x_no_clustering, y_no_clustering, z_no_clustering, z_gal_no_clustering, filename=f'{loc}/3D_galaxy_distribution_no_clustering.png')

# # Plotting.plot_3D_galaxy_distribution(x_clustering, y_clustering, z_clustering, z_gal_clustering, filename=f'{loc}/3D_galaxy_distribution_clustering.png')


# zmin_slice = 0.5
# zmax_slice = 1.0

# # ====================
# # MOLLWEIDE PROJECTION
# # ====================

# # Plotting.mollweide_sky_map(ra_deg_no_clustering, dec_deg_no_clustering, z_gal_no_clustering, zmin_slice=zmin_slice, zmax_slice=zmax_slice, filename=f'{loc}/mollweide_sky_map_no_clustering.png')

# # Plotting.mollweide_sky_map(ra_deg_clustering, dec_deg_clustering, z_gal_clustering, zmin_slice=zmin_slice, zmax_slice=zmax_slice, filename=f'{loc}/mollweide_sky_map_clustering.png')

# # ====================
# # ANGULAR DENSITY SLICE
# # ====================

# # Plotting.angular_density_slice(zmin_slice=zmin_slice, zmax_slice=zmax_slice, ra_deg=ra_deg_clustering, dec_deg=dec_deg_clustering, z_gal=z_gal_clustering, filename=f'{loc}/angular_density_slice_clustering.png')


# # ==========================
# # EVENT RATES HIST 2D WATERFALL
# # ==========================

# # Plotting.plot_event_rates_hist_2d_waterfall(lgMBH_no_clustering, z_gal_no_clustering, generated_EMRIs_no_clustering, generated_TDEs_no_clustering, args.OBSERVING_WINDOW, filename=f'{loc}/event_rates_hist_2d_waterfall_no_clustering.png')

# # Plotting.plot_event_rates_hist_2d_waterfall(lgMBH_clustering, z_gal_clustering, generated_EMRIs_clustering, generated_TDEs_clustering, args.OBSERVING_WINDOW, filename=f'{loc}/event_rates_hist_2d_waterfall_clustering.png')


# # ==========================
# # MBH VS WATERFALL
# # ==========================

# # Plotting.plot_MBH_vs_waterfall(lgMBH_no_clustering, z_gal_no_clustering, generated_EMRIs_no_clustering, generated_TDEs_no_clustering, args.OBSERVING_WINDOW, filename=f'{loc}/MBH_vs_waterfall_no_clustering.png')

# # Plotting.plot_MBH_vs_waterfall(lgMBH_clustering, z_gal_clustering, generated_EMRIs_clustering, generated_TDEs_clustering, args.OBSERVING_WINDOW, filename=f'{loc}/MBH_vs_waterfall_clustering.png')

# # import matplotlib.pyplot as plt
# # import numpy as np
# # plt.hist(lgMBH_no_clustering, bins=30, alpha=0.5, label='No Clustering', histtype='step')
# # # plt.hist(lgMBH_clustering, bins=30, alpha=0.5, label='Clustering', histtype='step')
# # plt.xlabel('log10(MBH Mass)')
# # plt.ylabel('Number of Galaxies')
# # plt.title('MBH Mass Distribution')
# # plt.legend()
# # plt.savefig(f'{loc}/MBH_mass_histogram.png')


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

# ============================================================
# Load resampled EMRI / TDE catalogs (v2 pipeline schema)
#
# Each row in EMRI_catalog / TDE_catalog is already ONE physical
# event, with its own (lgMBH, z_gal, ...) -- not a per-galaxy count
# table anymore. No shared host_id links a row in one catalog to a
# row in the other (hosts were independently re-rolled per catalog --
# see "EMRI rate sampling feasibility challenge" thread).
# ============================================================

with h5py.File(f'{loc}/data_cusp_evolution_no_clustering.h5', 'r') as f:
    emri = {field: f['EMRI_catalog'][field][:] for field in f['EMRI_catalog'].keys()}
    tde = {field: f['TDE_catalog'][field][:] for field in f['TDE_catalog'].keys()}

GALAXY_FIELDS = ['lgMgal', 'z_gal', 'lgMBH', 'sigma_km_s',
                  'ra_deg', 'dec_deg', 'initial_MBHspin']


def combine_and_unique(emri_dict, tde_dict, fields=GALAXY_FIELDS):
    """
    Stack the shared galaxy/MBH property fields from the EMRI and TDE
    catalogs into one array, deduplicated by exact row match.

    NOTE: EMRI and TDE host properties are independently re-rolled
    draws, not shared host realizations, so this dedup is a safety
    net against exact float collisions / empty catalogs -- it is NOT
    expected to meaningfully collapse "the same galaxy counted twice"
    (that's rare by construction, per the shared-host rarity argument).
    """
    n_e = len(emri_dict.get('lgMgal', []))
    n_t = len(tde_dict.get('lgMgal', []))
    if n_e == 0 and n_t == 0:
        return {field: np.array([]) for field in fields}

    stacked = {field: np.concatenate([emri_dict.get(field, np.array([])),
                                       tde_dict.get(field, np.array([]))])
               for field in fields}

    n_rows = len(stacked[fields[0]])
    dtype = [(field, 'f8') for field in fields]
    struct = np.zeros(n_rows, dtype=dtype)
    for field in fields:
        struct[field] = stacked[field]

    unique_struct = np.unique(struct)
    return {field: unique_struct[field] for field in fields}


galaxy_props = combine_and_unique(emri, tde)
lgMBH_combined = galaxy_props['lgMBH']
spin_combined = galaxy_props['initial_MBHspin']
z_combined = galaxy_props['z_gal']
ra_combined = galaxy_props['ra_deg']
dec_combined = galaxy_props['dec_deg']

# Eccentricities at plunge: EMRI-only, no TDE equivalent
eccentricities_at_plunge = emri.get('eccentricities_at_plunge', np.array([]))
lgMBH_emri = emri['lgMBH']
z_emri = emri['z_gal']
lgMBH_tde = tde['lgMBH']
z_tde = tde['z_gal']

print(f"EMRI catalog size: {len(z_emri)}")
print(f"TDE catalog size: {len(z_tde)}")
print(f"Combined unique host population (property plots): {len(z_combined)}")


# ====================
# MBH MASS VS SPIN  (galaxy property -> combined EMRI+TDE population)
# ====================

Plotting.MBHmass_vs_spin(
    lgMBH_combined, spin_combined, z_combined,
    loc=f'{loc}/MBHmass_vs_spin_resampled.png', cmap='plasma'
)

# ====================
# MBH MASS VS ECCENTRICITY AT PLUNGE  (EMRI-only)
# ====================

Plotting.MBHmass_vs_eccentricity_at_plunge(
    lgMBH_emri, eccentricities_at_plunge, z_emri,
    loc=f'{loc}/MBHmass_vs_eccentricity_at_plunge_resampled.png', cmap='plasma'
)

# ====================
# EVENT RATES  (new signature: separate EMRI/TDE arrays, no more
# generated_EMRIs/generated_TDEs count columns)
# ====================

Plotting.plot_event_rates(
    z_emri, lgMBH_emri, z_tde, lgMBH_tde, args.OBSERVING_WINDOW,
    filename=f'{loc}/event_rates_resampled.png'
)

# ====================
# EVENT RATES HIST 2D  -- REMOVED (redundant with the waterfall version,
# no SNR/detectability overlays, no added value)
# ====================

# ====================
# POLAR RA vs DEC  (galaxy property -> combined EMRI+TDE population)
# ====================

Plotting.plot_polar_ra_dec(
    ra_combined, dec_combined, z_combined,
    title="RA vs Dec (Resampled EMRI+TDE hosts)",
    filename=f'{loc}/polar_ra_dec_resampled.png', cmap='plasma'
)

# ====================
# EVENT RATES HIST 2D WATERFALL  (new signature: separate EMRI/TDE
# (lgMBH, z) arrays; counts are now direct histogram2d output, no
# generated_EMRIs/generated_TDEs weights needed)
# ====================

Plotting.plot_event_rates_hist_2d_waterfall(
    lgMBH_emri, z_emri, lgMBH_tde, z_tde, args.OBSERVING_WINDOW,
    filename=f'{loc}/event_rates_hist_2d_waterfall_resampled.png'
)

# ====================
# MBH VS WATERFALL
#
# NOT YET UPDATED -- plot_MBH_vs_waterfall still expects the old
# shared (lgMBH, z_gal) + generated_EMRIs/generated_TDEs count-column
# signature. Wasn't in your list of three to change, so left as-is for
# now. Say the word if you want this one converted to the same
# separate-catalog pattern as the other two (it's the same fix).
# ====================