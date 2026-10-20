#!/usr/bin/env python
"""
Marta's simplified ZTF detection-rate recipe (Eddington luminosity,
single 3-month duration, single-band flux, m_band<20.5, half sky,
2-year SURVEY WINDOW -- per user decision, "2 years" = the survey
observing baseline, not the per-event 3-month visibility duration),
placed side-by-side with the freshly re-simulated pipeline's own
detection counts (tabulate_grid_counts.py's output, tde_fallback/ZTF)
for direct comparison.

The flux/magnitude formula is copied from
DIAGNOSTICS/collaborator_detection_test.py:47-57 (apparent_mag,
z_max_for_mass) rather than re-derived -- that script already implements
Marta's exact recipe and already uses survey_years as the "2 years"
(not duration_yr, which the recipe defines separately for per-event
lightcurve shape but which this fixed-luminosity analytic estimate
does not actually need in its N_events formula).

Usage:
    python tests/PPS_VERIFICATION/ztf_analytic_vs_pipeline.py \
        [--pipeline_csv astrophysical_data/20000/verification_grid_counts_9001.csv]
"""
import os, sys, argparse, csv
import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

sys.path.insert(0, os.path.abspath('./astrophysical_setup'))
sys.path.insert(0, os.path.abspath('./'))
from cosmology import GalaxyStellarMassFunction, MBHMassFunction

parser = argparse.ArgumentParser()
parser.add_argument("--pipeline_csv", type=str, default=None,
                     help="verification_grid_counts_*.csv from tabulate_grid_counts.py, "
                          "to print side-by-side with the analytic estimate.")
args = parser.parse_args()

cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)

# ---------------- Marta's detection spec (copied from DIAGNOSTICS/collaborator_detection_test.py) ----------------
L0 = 1e38                  # erg/s per Msun of M_BH (Eddington-like)
f_band = 0.05
dnu_band_Hz = 2e14
m_limit = 20.5
survey_years = 2.0          # "2 years" = survey observing WINDOW (per user decision)
sky_fraction = 0.5
xi_TDE_per_yr = 1e-4         # fiducial, within Marta's requested 1e-4 - 1e-5 range


def apparent_mag(MBH_Msun, z):
    L = L0 * MBH_Msun
    L_band = f_band * L
    D_L_cm = cosmo.luminosity_distance(z).to(u.cm).value
    f_nu = (L_band / dnu_band_Hz) / (4 * np.pi * D_L_cm ** 2)
    return -2.5 * np.log10(f_nu) - 48.60


def z_max_for_mass(MBH_Msun, z_grid):
    mags = np.array([apparent_mag(MBH_Msun, z) for z in z_grid])
    ok = mags < m_limit
    return z_grid[ok].max() if ok.any() else 0.0


gsmf = GalaxyStellarMassFunction()
mbhmf = MBHMassFunction(gsmf)

logMBH_bins = np.linspace(4.0, 8.0, 41)
z_fine = np.linspace(1e-4, 2.0, 4000)

total_events = 0.0
for i in range(len(logMBH_bins) - 1):
    lo, hi = logMBH_bins[i], logMBH_bins[i + 1]
    mid_logM = 0.5 * (lo + hi)
    MBH_mid = 10 ** mid_logM

    logMBH_grid, log10phi_MBH = mbhmf.get_mbhmf(0.0, n_points_mass=4000)
    phi = 10 ** log10phi_MBH[0]
    in_bin = (logMBH_grid >= lo) & (logMBH_grid < hi)
    if not in_bin.any():
        continue
    n_density = np.trapz(phi[in_bin], logMBH_grid[in_bin])

    zmax = z_max_for_mass(MBH_mid, z_fine)
    if zmax <= 0:
        continue
    V_max_Mpc3 = cosmo.comoving_volume(zmax).to(u.Mpc ** 3).value
    N_gal = n_density * V_max_Mpc3

    total_events += xi_TDE_per_yr * survey_years * N_gal * sky_fraction

print("=" * 78)
print("Analytic ZTF detection-rate estimate (Marta's recipe)")
print("=" * 78)
print(f"L = 1e38*(M_BH/Msun) erg/s, L_band=0.05L, dnu_band=2e14 Hz, m_limit={m_limit}, "
      f"half sky, {survey_years:.0f}-year survey window, xi_TDE={xi_TDE_per_yr:.0e}/gal/yr")
print(f"\nTOTAL predicted TDE detections (analytic): {total_events:.1f}")

if args.pipeline_csv and os.path.exists(args.pipeline_csv):
    print("\n" + "=" * 78)
    print(f"Pipeline (fresh re-run) detection counts from {args.pipeline_csv}")
    print("=" * 78)
    with open(args.pipeline_csv) as f:
        for row in csv.DictReader(f):
            if row["channel"] == "TDE" and row["survey"] == "ZTF" and row["model"] == "tde_fallback":
                print(f"  {row['scenario']:6s} ({row['mbh']}/{row['nuc']}/{row['occ']}): "
                      f"n_total={row['n_total']}  n_passed={row['n_passed']}  "
                      f"[vs analytic {total_events:.1f}]")
elif args.pipeline_csv:
    print(f"\n[WARN] --pipeline_csv given but not found: {args.pipeline_csv} "
          f"(run tabulate_grid_counts.py first)")
else:
    print("\n(pass --pipeline_csv <verification_grid_counts_*.csv> to compare against "
          "the freshly re-run pipeline's tde_fallback/ZTF detection counts)")
