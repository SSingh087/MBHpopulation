#!/usr/bin/env python
"""
Plot the ZTF/LSST completeness-gradient map (built once by
build_footprint_maps.py) overlaid with a REAL simulated TDE population for
one (mbh_mass_method, nucleation_method, occupation_method, TDE_model,
UNIVERSE_MODEL) combo.

Population points are colored by the survey's local pointing-density
gradient at their sky position (continuous, not a hard cut) -- only points
that fall completely outside the survey's footprint (never pointed at)
get the separate "outside footprint" marker, since that boundary is a
real geometric fact, not a density gradient.

Usage:
    python data_generation/footprint_LSST_ZTF.py \
        --UNIVERSE_MODEL 0 --GALAXIES 100000 --TDE_model cooling_envelope \
        --mbh_mass_method use_lgMgal --nucleation_method hannah --occupation_method uniform
"""

import sys, os
sys.path.insert(0, os.path.abspath('./'))
from utils import ASTROPHYSICAL_DATA_DIR, TDE_DATA_FILE_NAME, MERGER_MODEL_CHOICES

import argparse

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# NOTE: importing redback (via bilby), even transitively, silently flips
# matplotlib's rcParams['text.usetex'] to True as a side effect, which
# then breaks plotting on any machine without a full LaTeX installation.
matplotlib.rcParams['text.usetex'] = False

MODELS = [
    "cooling_envelope",
    "stream_stream_tde",
    "tde_analytical",
    "tde_fallback",
]

# ────────────────────────────────────────────────────────────────────
# 0. Config
# ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Plot ZTF/LSST footprint + completeness gradient against a "
                "real, per-(TDE model, universe) simulated TDE catalog.")
parser.add_argument("--UNIVERSE_MODEL", type=int, required=True, help="Universe model / realization index")
parser.add_argument("--GALAXIES", type=int, required=True, help="Monte Carlo sample size used to label the run directory")
parser.add_argument("--TDE_model", type=str, required=True, choices=MODELS,
                     help="redback TDE model this catalog was generated for.")
parser.add_argument("--mbh_mass_method", type=str, required=True, choices=["use_sigma", "use_lgMgal"])
parser.add_argument("--nucleation_method", type=str, required=True, choices=["hannah", "neumayer"])
parser.add_argument("--occupation_method", type=str, required=True, choices=["uniform", "varying"])
parser.add_argument("--merger_model", type=str, required=True, choices=MERGER_MODEL_CHOICES,
                    help="Galaxy major-merger-rate model (see astrophysical_setup/cosmology.py MERGER_MODELS)")
args = parser.parse_args()

TAG = f"{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}_{args.merger_model}_{args.TDE_model}_{args.UNIVERSE_MODEL}"

PALOMAR_LAT_DEG = 33.35
CERRO_PACHON_LAT_DEG = -30.24

# ────────────────────────────────────────────────────────────────────
# 1. Load the cached coverage maps (built once, combo-independent, by
#    build_footprint_maps.py -- NOT rebuilt here).
# ────────────────────────────────────────────────────────────────────
COVERAGE_NPZ = os.path.join(ASTROPHYSICAL_DATA_DIR, 'coverage_maps_nohealpy.npz')
if not os.path.exists(COVERAGE_NPZ):
    raise RuntimeError(
        f"{COVERAGE_NPZ} not found. Run `python data_generation/build_footprint_maps.py` "
        f"once first -- it builds the ZTF/Rubin coverage maps, which are shared across "
        f"every (model, universe) combo and don't need rebuilding per combo.")

cov = np.load(COVERAGE_NPZ)
ztf_mask, ztf_grad = cov['ztf_mask'], cov['ztf_grad']
rubin_mask, rubin_grad = cov['rubin_mask'], cov['rubin_grad']
ra_edges_deg, dec_edges_deg = cov['ra_edges_deg'], cov['dec_edges_deg']
ra_centers_deg, dec_centers_deg = cov['ra_centers_deg'], cov['dec_centers_deg']
N_DEC, N_RA = ztf_mask.shape


def radec_to_grid_indices(ra_rad, dec_rad):
    ra_deg = np.rad2deg(ra_rad) % 360
    ra_deg = np.where(ra_deg > 180, ra_deg - 360, ra_deg)   # wrap to [-180, 180)
    dec_deg = np.rad2deg(dec_rad)
    i_ra = np.clip(np.digitize(ra_deg, ra_edges_deg) - 1, 0, N_RA - 1)
    i_dec = np.clip(np.digitize(dec_deg, dec_edges_deg) - 1, 0, N_DEC - 1)
    return i_ra, i_dec


# ────────────────────────────────────────────────────────────────────
# 2. Load the real, per-(TDE model, universe) simulated TDE population
#    and look up each event's local completeness (pointing-density
#    gradient) at its sky position. Same catalog (ra_rad/dec_rad) is
#    used for both surveys -- only the footprint mask/gradient differs
#    between ZTF and LSST.
# ────────────────────────────────────────────────────────────────────
catalog_path = (f'{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}/{TDE_DATA_FILE_NAME}_{TAG}.h5')
print(f"[INFO] TDE catalog: {catalog_path}")

with h5py.File(catalog_path, 'r') as f:
    ra_rad = np.array(f['ra_rad'][:])
    dec_rad = np.array(f['dec_rad'][:])

n_events = len(ra_rad)
if n_events == 0:
    raise RuntimeError(f"TDE catalog {catalog_path} has no events -- nothing to plot.")
print(f"[INFO] Loaded {n_events} simulated TDE events for {TAG}")


def local_completeness(ra_rad, dec_rad, grad_map):
    """Density-gradient value (0-1) at each event's sky position; NaN
    where the survey never points (hard geometric footprint edge)."""
    i_ra, i_dec = radec_to_grid_indices(ra_rad, dec_rad)
    return grad_map[i_dec, i_ra]


comp_ztf = local_completeness(ra_rad, dec_rad, ztf_grad)
comp_lsst = local_completeness(ra_rad, dec_rad, rubin_grad)

in_footprint_ztf = ~np.isnan(comp_ztf)
in_footprint_lsst = ~np.isnan(comp_lsst)

if in_footprint_ztf.any():
    print(f"[INFO] ZTF era:  {in_footprint_ztf.sum()}/{n_events} within footprint "
          f"({in_footprint_ztf.mean():.1%}), mean local completeness "
          f"{np.nanmean(comp_ztf):.2f}")
else:
    print(f"[INFO] ZTF era:  0/{n_events} within footprint")

if in_footprint_lsst.any():
    print(f"[INFO] LSST era: {in_footprint_lsst.sum()}/{n_events} within footprint "
          f"({in_footprint_lsst.mean():.1%}), mean local completeness "
          f"{np.nanmean(comp_lsst):.2f}")
else:
    print(f"[INFO] LSST era: 0/{n_events} within footprint")


# ────────────────────────────────────────────────────────────────────
# 3. Plot with matplotlib's built-in Mollweide axes projection
#    (matplotlib expects lon in [-pi, pi] with 0 at center, lat in [-pi/2, pi/2])
# ────────────────────────────────────────────────────────────────────
def to_mpl_lonlat(ra_deg, dec_deg):
    """Convert RA (any convention) to matplotlib's -pi..pi Mollweide
    convention, mirrored so RA increases to the left (usual sky-map sense)."""
    ra_deg = np.asarray(ra_deg, dtype=float) % 360
    ra_deg = np.where(ra_deg > 180, ra_deg - 360, ra_deg)   # wrap to [-180, 180)
    lon = -np.deg2rad(ra_deg)                                # mirror
    lat = np.deg2rad(dec_deg)
    return lon, lat


# ra_centers_deg is already in [-180, 180) by construction, and matches
# the grid used for binning above -> just mirror it for the plot, no wrap needed.
RA_2d, DEC_2d = np.meshgrid(ra_centers_deg, dec_centers_deg)  # (N_DEC, N_RA)
LON, LAT = -np.deg2rad(RA_2d), np.deg2rad(DEC_2d)


def plot_panel(ax, grad_map, ra_pop, dec_pop, comp_pop, in_footprint, obs_lat_deg, obs_label, title):
    pcm = ax.pcolormesh(LON, LAT, grad_map, cmap='viridis', vmin=0, vmax=1,
                         shading='auto')
    ax.grid(True, alpha=0.3)

    lon_out, lat_out = to_mpl_lonlat(np.rad2deg(ra_pop[~in_footprint]) % 360,
                                      np.rad2deg(dec_pop[~in_footprint]))
    ax.scatter(lon_out, lat_out, marker='.', s=3, color='black', alpha=0.15,
               label='TDE: outside footprint (never pointed at)')

    lon_in, lat_in = to_mpl_lonlat(np.rad2deg(ra_pop[in_footprint]) % 360,
                                    np.rad2deg(dec_pop[in_footprint]))
    ax.scatter(lon_in, lat_in, marker='x', s=10, c=comp_pop[in_footprint],
               cmap='plasma', vmin=0, vmax=1, alpha=0.7,
               label='TDE: within footprint (colored by local completeness)')

    lon_obs, lat_obs = to_mpl_lonlat(np.array([0.0]), np.array([obs_lat_deg]))
    ax.scatter(lon_obs, lat_obs, marker='*', s=400, facecolor='red',
               edgecolor='white', linewidth=1.5, zorder=10,
               label=f'Observatory zenith Dec ({obs_label})')

    ax.set_title(title, fontsize=11)
    ax.legend(loc='lower left', fontsize=7, framealpha=0.9)
    return pcm


# NOTE: each survey is rendered to its own figure/file (rather than stacked
# subplots + tight_layout) -- this avoids a known matplotlib layout-engine
# vs. Mollweide-projection interaction that can misfire on some installs.

FOOTPRINT_DIR = os.path.join(ASTROPHYSICAL_DATA_DIR, str(args.GALAXIES), "footprints")
os.makedirs(FOOTPRINT_DIR, exist_ok=True)

fig1 = plt.figure(figsize=(11, 6))
ax1 = fig1.add_subplot(1, 1, 1, projection='mollweide')
pcm1 = plot_panel(ax1, ztf_grad, ra_rad, dec_rad, comp_ztf, in_footprint_ztf,
                   PALOMAR_LAT_DEG, 'Palomar',
                   f"ZTF era — footprint, completeness gradient & simulated TDE population ({TAG})")
cb1 = fig1.colorbar(pcm1, ax=ax1, orientation='horizontal', pad=0.08, shrink=0.6)
cb1.set_label('redback pointing-density (normalized completeness)')
OUT_PNG_ZTF = os.path.join(FOOTPRINT_DIR, f"ztf_footprint_gradient_{TAG}.png")
fig1.savefig(OUT_PNG_ZTF, dpi=170, bbox_inches='tight')
plt.close(fig1)
print(f"[DONE] Saved figure -> {OUT_PNG_ZTF}")

fig2 = plt.figure(figsize=(11, 6))
ax2 = fig2.add_subplot(1, 1, 1, projection='mollweide')
pcm2 = plot_panel(ax2, rubin_grad, ra_rad, dec_rad, comp_lsst, in_footprint_lsst,
                   CERRO_PACHON_LAT_DEG, 'Cerro Pachon',
                   f"LSST/Rubin era — footprint, completeness gradient & simulated TDE population ({TAG})")
cb2 = fig2.colorbar(pcm2, ax=ax2, orientation='horizontal', pad=0.08, shrink=0.6)
cb2.set_label('redback pointing-density (normalized completeness)')
OUT_PNG_LSST = os.path.join(FOOTPRINT_DIR, f"lsst_footprint_gradient_{TAG}.png")
fig2.savefig(OUT_PNG_LSST, dpi=170, bbox_inches='tight')
plt.close(fig2)
print(f"[DONE] Saved figure -> {OUT_PNG_LSST}")
