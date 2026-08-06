#!/usr/bin/env python

import os, sys
sys.path.insert(0, os.path.abspath('./astrophysical_setup'))
sys.path.insert(0, os.path.abspath('./'))

import numpy as np
import argparse
import h5py

from cosmology import CosmologyModel
from utils import ASTROPHYSICAL_DATA_DIR, MOCK_GALAXY_CATALOG_DATA, EMRI_DATA_FILE_NAME

from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_separatrix

_ODE_FUNC = "pn5"

cosmo_model = CosmologyModel()

# ============================================================
# Argument parsing
# ============================================================
parser = argparse.ArgumentParser(description="Generate EMRI waveform parameters.")
parser.add_argument("--UNIVERSE_MODEL", type=int, required=True, help="Universe model index")
parser.add_argument("--GALAXIES", type=int, required=True, help="Monte Carlo sample size used to label the run directory")
parser.add_argument("--Y0", type=float, nargs=2, required=True, help="cos(inclination) range [min, max]")
parser.add_argument("--T_SIGNAL", type=float, nargs=2, required=True, help="Observation duration range, years [min, max]")
parser.add_argument("--dt", type=float, default=10.0, help="Trajectory integrator time step (s)")
parser.add_argument(
    "--E0_FALLBACK", type=float, nargs=2, default=[0.05, 0.5],
    help="Uniform fallback range for e0 when backward integration fails for an event"
)
parser.add_argument("--mbh_mass_method", type=str, required=False, default="use_sigma", choices=["use_sigma", "use_lgMgal"])
parser.add_argument("--nucleation_method", type=str, required=False, default="hannah", choices=["hannah", "neumayer"])
parser.add_argument("--SEED", type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()

rng = np.random.default_rng(seed=args.SEED)

# ============================================================
# Load resampled EMRI catalog (flat, one row = one event)
# ============================================================
loc = f'{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}'
catalog_path = f'{loc}/{MOCK_GALAXY_CATALOG_DATA}_{args.mbh_mass_method}_{args.nucleation_method}_{args.UNIVERSE_MODEL}.h5'

with h5py.File(catalog_path, 'r') as f:
    emri = {field: np.array(f['EMRI_catalog'][field][:]) for field in f['EMRI_catalog'].keys()}

n_events = len(emri['lgMBH'])
print(f"Loaded {n_events} EMRI events from {catalog_path}")

if n_events == 0:
    raise RuntimeError(f"EMRI_catalog in {catalog_path} is empty -- nothing to generate parameters for.")

z_gal = emri['z_gal']
qS_rad = np.pi / 2 - np.deg2rad(emri['ra_deg'])   # ecliptic polar angle
phiS_rad = np.deg2rad(emri['dec_deg'])             # ecliptic azimuthal angle
lgMBH_mass = emri['lgMBH']
initial_MBHspin = emri['initial_MBHspin']
eccentricities_at_plunge = emri['eccentricities_at_plunge']
sBH_masses = 10.0 * np.ones(n_events)
distances_Gpc = cosmo_model.luminosity_distance(z_gal, unit='Gpc')

# ============================================================
# Sample Y0 (x0 / cos-inclination-type constant of motion) and
# T_SIGNAL FIRST, per event -- both are required INPUTS to the
# backward trajectory integration below, not free parameters that
# can be drawn independently afterward.
# ============================================================
Y0_min, Y0_max = args.Y0
T_SIGNAL_min, T_SIGNAL_max = args.T_SIGNAL
qK_min, qK_max = 0.1, np.pi * 0.99
phiK_min, phiK_max = 0.1, 2 * np.pi * 0.99
Phi_phi0_min, Phi_phi0_max = 0.1, 2 * np.pi * 0.99
Phi_theta0_min, Phi_theta0_max = 0.1, 2 * np.pi * 0.99
Phi_r0_min, Phi_r0_max = 0.1, 2 * np.pi * 0.99

Y0_samples = rng.uniform(Y0_min, Y0_max, size=n_events)
T_SIGNAL_samples = rng.uniform(T_SIGNAL_min, T_SIGNAL_max, size=n_events)
qK_samples = rng.uniform(qK_min, qK_max, size=n_events)
phiK_samples = rng.uniform(phiK_min, phiK_max, size=n_events)
Phi_phi0_samples = rng.uniform(Phi_phi0_min, Phi_phi0_max, size=n_events)
Phi_theta0_samples = rng.uniform(Phi_theta0_min, Phi_theta0_max, size=n_events)
Phi_r0_samples = rng.uniform(Phi_r0_min, Phi_r0_max, size=n_events)


traj = EMRIInspiral(func=_ODE_FUNC)

def compute_e0_p0_backward(M, mu, a, e_plunge, x0, T_years, dt):
    p_plunge = get_separatrix(a, e_plunge, x0)
    t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(
        M, mu, a, p_plunge, e_plunge, x0,
        Phi_phi0=0.0, Phi_theta0=0.0, Phi_r0=0.0,
        dt=dt, T=T_years, integrate_backwards=True,
    )
    return e[-1], p[-1]

e0_fallback_min, e0_fallback_max = args.E0_FALLBACK
e0s = np.empty(n_events)
p0s = np.empty(n_events)
n_fallback = 0
for i in range(n_events):
    try:
        e0s[i], p0s[i] = compute_e0_p0_backward(
            M=10 ** lgMBH_mass[i], mu=sBH_masses[i], a=initial_MBHspin[i],
            e_plunge=eccentricities_at_plunge[i], x0=Y0_samples[i],
            T_years=T_SIGNAL_samples[i], dt=args.dt,
        )
    except Exception as exc:
        e0s[i] = rng.uniform(e0_fallback_min, e0_fallback_max)
        p0s[i] = np.nan
        n_fallback += 1

if n_fallback > 0:
    print(f"WARNING: backward integration failed for {n_fallback}/{n_events} events "
          f"({100 * n_fallback / n_events:.1f}%) -- used uniform fallback e0 in "
          f"[{e0_fallback_min}, {e0_fallback_max}] for those (p0 set to NaN for those rows).")
    print(f"NOTE: rows with p0=NaN are EXCLUDED downstream by the SNR script's "
          f"`valid = np.isfinite(p0_arr) & ...` mask -- a high fallback rate here "
          f"silently shrinks your usable EMRI sample further, on top of whatever "
          f"the SNR threshold cuts. Worth checking this percentage specifically "
          f"for low-count scenarios before assuming the SNR cut is the only culprit.")

params = np.column_stack([
    Y0_samples, qK_samples, phiK_samples,
    Phi_phi0_samples, Phi_theta0_samples, Phi_r0_samples,
    T_SIGNAL_samples,
])
param_names = ["Y0", "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0", "T_SIGNAL_duration_years"]

# ============================================================
# Save flat output: one row per event, no galaxy grouping.
# ============================================================
out_path = (f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}/{EMRI_DATA_FILE_NAME}_{args.mbh_mass_method}_{args.nucleation_method}_{args.UNIVERSE_MODEL}.h5")

with h5py.File(out_path, "w") as f:
    f.attrs['n_events'] = n_events
    f.attrs['n_e0_fallback'] = n_fallback

    f.create_dataset("lgMBH_mass", data=lgMBH_mass)
    f.create_dataset("distance_Gpc", data=distances_Gpc)
    f.create_dataset("z_gal", data=z_gal)
    f.create_dataset("MBHspin", data=initial_MBHspin)
    f.create_dataset("qS", data=qS_rad)
    f.create_dataset("phiS", data=phiS_rad)
    f.create_dataset("sBH_mass", data=sBH_masses)
    f.create_dataset("e0", data=e0s)
    f.create_dataset("p0", data=p0s)
    f.create_dataset("eccentricities_at_plunge", data=eccentricities_at_plunge)

    for j, name in enumerate(param_names):
        f.create_dataset(name, data=params[:, j])

print(f"Saved {n_events} EMRI parameter rows to {out_path}")


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.scatter(e0s, p0s, c=Y0_samples, cmap='viridis', s=10)
plt.colorbar(label='Y0')
plt.xlabel('e0')
plt.ylabel('p0')
plt.subplot(2, 2, 2)
plt.scatter(e0s, T_SIGNAL_samples, c=Y0_samples, cmap='viridis', s=10)
plt.colorbar(label='Y0')
plt.xlabel('e0')
plt.ylabel('T_SIGNAL (years)')
plt.subplot(2, 2, 3)
plt.scatter(p0s, T_SIGNAL_samples, c=Y0_samples, cmap='viridis', s=10)
plt.colorbar(label='Y0')
plt.xlabel('p0')
plt.ylabel('T_SIGNAL (years)')
plt.subplot(2, 2, 4)
plt.scatter(e0s, p0s, c=T_SIGNAL_samples, cmap='plasma', s=10)
plt.colorbar(label='T_SIGNAL (years)')
plt.xlabel('e0')
plt.ylabel('p0')
plt.tight_layout()
plt.savefig(f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}/EMRI_params_scatter_plots_{args.mbh_mass_method}_{args.nucleation_method}_{args.UNIVERSE_MODEL}.png")
plt.close()