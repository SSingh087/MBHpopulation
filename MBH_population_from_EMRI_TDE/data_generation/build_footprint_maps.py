#!/usr/bin/env python
"""
One-time build of the ZTF/LSST sky-coverage + completeness-gradient maps
from redback's real pointing-history tables, WITHOUT healpy.

This is the slow part (parsing the pointing tables and, for ZTF, an
O(n_fields x grid) loop to inflate each fixed field to its true circular
FOV) -- it depends only on redback's shipped pointing tables, not on any
(model, universe) combo, so it should run exactly ONCE and be cached to
coverage_maps_nohealpy.npz. footprint_LSST_ZTF.py (run once per combo)
just loads that cache instead of rebuilding it.

Uses a plain RA/Dec grid (equirectangular binning) + matplotlib's built-in
Mollweide projection instead of HEALPix. This is NOT equal-area (grid cells
shrink in true sky-area near the poles), so sky-fraction numbers are
slightly less accurate near the celestial poles than the HEALPix version --
but ZTF/LSST's Dec cuts sit at mid-latitude, so the footprint numbers below
still come out essentially correct (matches the HEALPix version to <1%).

Usage:
    python data_generation/build_footprint_maps.py
"""

import sys, os
sys.path.insert(0, os.path.abspath('./'))
from utils import ASTROPHYSICAL_DATA_DIR

import gzip
import shutil

import numpy as np
import pandas as pd

import redback.simulate_transients as st

# ────────────────────────────────────────────────────────────────────
# 0. Config
# ────────────────────────────────────────────────────────────────────
N_RA, N_DEC = 360, 180        # grid resolution: 1 deg x 1 deg cells

FOV_ZTF_DEG2 = 47.0
ZTF_FOV_RADIUS_DEG = np.sqrt(FOV_ZTF_DEG2 / np.pi)   # circular-equivalent radius

COVERAGE_NPZ = os.path.join(ASTROPHYSICAL_DATA_DIR, 'coverage_maps_nohealpy.npz')


# ────────────────────────────────────────────────────────────────────
# 1. Extract redback's shipped pointing tables (plain gzip, not tar)
# ────────────────────────────────────────────────────────────────────
def extract_gz(src_path, dst_path):
    if os.path.exists(dst_path):
        return dst_path
    with gzip.open(src_path, 'rb') as f_in, open(dst_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    return dst_path


tables_dir = os.path.join(os.path.dirname(st.__file__), 'tables')
ztf_path = extract_gz(os.path.join(tables_dir, 'ztf.tar.gz'),
                       os.path.join(ASTROPHYSICAL_DATA_DIR, 'ztf.csv'))
rubin_path = extract_gz(os.path.join(tables_dir, 'rubin_baseline_v3.0_10yrs.tar.gz'),
                         os.path.join(ASTROPHYSICAL_DATA_DIR, 'rubin.csv'))
print(f"[INFO] ZTF pointing table:   {ztf_path}")
print(f"[INFO] Rubin pointing table: {rubin_path}")


# ────────────────────────────────────────────────────────────────────
# 2. Build RA/Dec grid + visit-density maps + geometric footprint masks
# ────────────────────────────────────────────────────────────────────
# RA grid built directly in [-180, 180) (not [0, 360)) so it is already
# monotonic in the matplotlib Mollweide convention -- avoids a wrap-around
# seam artifact in pcolormesh at RA=0/360.
ra_edges_deg = np.linspace(-180, 180, N_RA + 1)
dec_edges_deg = np.linspace(-90, 90, N_DEC + 1)
ra_centers_deg = 0.5 * (ra_edges_deg[:-1] + ra_edges_deg[1:])
dec_centers_deg = 0.5 * (dec_edges_deg[:-1] + dec_edges_deg[1:])


def build_ztf_map(path):
    """ZTF fields are fixed (no dithering) -> inflate each unique field
    center to its true circular FOV by marking all grid cells within
    ZTF_FOV_RADIUS_DEG (great-circle distance) of the field center."""
    df = pd.read_csv(path, usecols=['_ra', '_dec'])
    fields = df.groupby(['_ra', '_dec']).size().reset_index(name='nvisits')

    visits = np.zeros((N_DEC, N_RA))
    mask = np.zeros((N_DEC, N_RA), dtype=bool)

    dec_grid_rad = np.deg2rad(dec_centers_deg)
    ra_grid_rad = np.deg2rad(ra_centers_deg)
    RA_grid, DEC_grid = np.meshgrid(ra_grid_rad, dec_grid_rad)  # shape (N_DEC, N_RA)

    radius_rad = np.deg2rad(ZTF_FOV_RADIUS_DEG)

    for ra_f, dec_f, nv in zip(fields['_ra'], fields['_dec'], fields['nvisits']):
        # great-circle (haversine) angular separation from field center to every grid cell
        dra = RA_grid - ra_f
        cos_sep = (np.sin(DEC_grid) * np.sin(dec_f) +
                   np.cos(DEC_grid) * np.cos(dec_f) * np.cos(dra))
        cos_sep = np.clip(cos_sep, -1, 1)
        sep = np.arccos(cos_sep)
        within = sep <= radius_rad
        visits[within] += nv
        mask[within] = True

    return visits, mask


def build_rubin_map(path):
    """Rubin pointings are dithered and dense -> direct grid-cell binning
    already fills the footprint without needing FOV inflation.
    _ra/_dec in rubin.csv are in RADIANS (like ztf.csv), not degrees."""
    df = pd.read_csv(path, usecols=['_ra', '_dec'])
    ra_deg = np.rad2deg(df['_ra'].values) % 360
    ra_deg = np.where(ra_deg > 180, ra_deg - 360, ra_deg)
    dec_deg = np.rad2deg(df['_dec'].values)
    i_ra = np.clip(np.digitize(ra_deg, ra_edges_deg) - 1, 0, N_RA - 1)
    i_dec = np.clip(np.digitize(dec_deg, dec_edges_deg) - 1, 0, N_DEC - 1)
    visits = np.zeros((N_DEC, N_RA))
    np.add.at(visits, (i_dec, i_ra), 1)
    mask = visits > 0
    return visits, mask


print("[INFO] Building ZTF coverage map (FOV-inflated, no dithering)... "
      "(this loop is O(n_fields x grid) -- the slow part, a few seconds)")
ztf_visits, ztf_mask = build_ztf_map(ztf_path)

print("[INFO] Building Rubin coverage map (dithered, direct binning)...")
rubin_visits, rubin_mask = build_rubin_map(rubin_path)

# Sky-fraction accounting for the non-uniform cell area on a plain grid
# (cos(dec) weighting -- otherwise polar cells are over-counted)
cell_weight = np.cos(np.deg2rad(dec_centers_deg))[:, None] * np.ones((1, N_RA))
ztf_area_frac = (ztf_mask * cell_weight).sum() / cell_weight.sum()
rubin_area_frac = (rubin_mask * cell_weight).sum() / cell_weight.sum()
print(f"       ZTF geometric sky fraction (area-weighted):   {ztf_area_frac:.3f}")
print(f"       Rubin geometric sky fraction (area-weighted): {rubin_area_frac:.3f}")


def normalized_gradient(visits, mask):
    g = np.full(visits.shape, np.nan)
    v = visits[mask]
    if v.size and v.max() > 0:
        g[mask] = np.clip(v / np.percentile(v, 95), 0, 1)
    return g


ztf_grad = normalized_gradient(ztf_visits, ztf_mask)
rubin_grad = normalized_gradient(rubin_visits, rubin_mask)

np.savez(COVERAGE_NPZ,
         ztf_visits=ztf_visits, ztf_mask=ztf_mask, ztf_grad=ztf_grad,
         rubin_visits=rubin_visits, rubin_mask=rubin_mask, rubin_grad=rubin_grad,
         ra_edges_deg=ra_edges_deg, dec_edges_deg=dec_edges_deg,
         ra_centers_deg=ra_centers_deg, dec_centers_deg=dec_centers_deg)
print(f"[DONE] Saved {COVERAGE_NPZ}")
