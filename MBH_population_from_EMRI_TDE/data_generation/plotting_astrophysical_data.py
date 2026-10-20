"""Summary plots of the combined EMRI+TDE mock catalogue (mass, spin, eccentricity, redshift)."""
import sys, os
sys.path.insert(0, os.path.abspath('./'))
from utils import ASTROPHYSICAL_DATA_DIR, MOCK_GALAXY_CATALOG_DATA

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

loc = f'{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}'

with h5py.File(f'{loc}/{MOCK_GALAXY_CATALOG_DATA}.h5', 'r') as f:
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