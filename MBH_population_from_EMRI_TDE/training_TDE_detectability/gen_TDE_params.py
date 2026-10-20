#!/usr/bin/env python
"""Draw TDE parameters for the detectability training set."""

import numpy as np
import argparse
import os, sys
import h5py
import matplotlib.pyplot as plt

# bpl_cooling_envelope and gaussianrise_cooling_envelope are excluded: both
# crash deterministically (KeyError: 'bands') under redback's
# SimulateOpticalTransient (confirmed still broken in redback 1.18.0).
MODEL_CHOICES = [
    "cooling_envelope",
    "stream_stream_tde",
    "tde_analytical",
    "tde_fallback",
]

# Argument parsing
parser = argparse.ArgumentParser(description="Generate data for training.")
parser.add_argument("--lgMBH", type=float, nargs=2, required=True, help="lgMBH range: min max")
parser.add_argument("--eta", type=float, nargs=2, required=True, help="eta range: min max")
parser.add_argument("--alpha", type=float, nargs=2, required=True, help="alpha range: min max")
parser.add_argument("--beta", type=float, nargs=2, required=True, help="beta range: min max")
parser.add_argument("--events", type=int, required=True,
                     help="number of events to sample for THIS model (i.e. EVENTS_PER_MODEL)")
parser.add_argument("--total_events", type=int, required=True,
                     help="total pipeline EVENTS (from config.sh) -- used only to place "
                          "output in the shared 1E{n}_events directory/filename tag, "
                          "consistent across all 6 models and the rest of the pipeline")
parser.add_argument("--z_gal", type=float, nargs=2, required=True, help="redshift range: min max")
parser.add_argument("--SURVEY", type=str, default="ztf", help="Survey name (e.g., 'ztf' or 'lsst')")
parser.add_argument("--model", type=str, required=True, choices=MODEL_CHOICES,
                     help="redback TDE model these params are generated for")
args = parser.parse_args()

MODEL = args.model
TAG = int(np.log10(args.total_events))

output_dir = f'./data/PRE_TRAIN_DATA_TDE/1E{TAG}_events/'

os.makedirs(output_dir, exist_ok=True)

lgMBH_min, lgMBH_max = args.lgMBH
z_gal_min, z_gal_max = args.z_gal
star_masses = 1.0
eta_min, eta_max = args.eta
alpha_min, alpha_max = args.alpha
beta_min, beta_max = args.beta

ra_rad_min, ra_rad_max = 0.0, 2 * np.pi
dec_rad_min, dec_rad_max = -np.pi / 2, np.pi / 2

if args.SURVEY.lower() == "ztf":
    t0_mjd_transient = np.random.uniform(58200, 61396, size=args.events) # ZTF survey period March 17, 2018 - Dec 31, 2026 8.75 yr
elif args.SURVEY.lower() == "lsst":
    # record: https://survey-strategy.lsst.io/progress/index.html; confirmed by Rubin's
    # official announcement https://rubinobservatory.org/news/action-rubin-lsst-begins).
    t0_mjd_transient = np.random.uniform(61220, 64872, size=args.events)  # 2026-06-29 -> 2036-06-28, 10.00 yr

def loguniform(low=0, high=1, size=None):
    """Log-uniform samples between low and high."""
    return np.exp(np.random.uniform(low, high, size))


print(f"Generating {args.events} events for model '{MODEL}' with parameters:")
print(f"lgMBH: {lgMBH_min} to {lgMBH_max}")
print(f"z_gal: {z_gal_min} to {z_gal_max}")
print(f"RA: {np.rad2deg(ra_rad_min)} to {np.rad2deg(ra_rad_max)} degrees / {ra_rad_min} to {ra_rad_max} radians")
print(f"DEC: {np.rad2deg(dec_rad_min)} to {np.rad2deg(dec_rad_max)} degrees / {dec_rad_min} to {dec_rad_max} radians")
print(f"eta: {eta_min} to {eta_max}")
print(f"alpha: {alpha_min} to {alpha_max}")
print(f"beta: {beta_min} to {beta_max}")

# ── Shared base astrophysical parameters ────────────────────────────────────
# Sampled/stored for EVERY model+event for provenance/consistency, even for
# models whose simulator function does not consume all of them (e.g.
# tde_analytical does not use lgMBH/eta/alpha/beta downstream, but we still
# store them here so the merged injection set has a uniform provenance
# record and so re-running with a different model dispatch is possible
# without re-sampling).
shared_data = {
    "lgMBH":             np.random.uniform(lgMBH_min, lgMBH_max, size=args.events),
    "z_gal":              np.random.uniform(z_gal_min, z_gal_max, size=args.events),
    "ra_rad":             np.random.uniform(ra_rad_min, ra_rad_max, size=args.events),
    # FIX: isotropic placement on the celestial sphere requires sampling
    # uniform in sin(dec), NOT uniform in dec itself. Since dOmega =
    # cos(dec) d(dec) d(ra), a naive np.random.uniform(-pi/2, pi/2) draw
    # over-samples the poles and under-samples the celestial equator
    # relative to a true isotropic population. This matters here
    # specifically because ZTF/LSST detectability is a strong function of
    # Dec near the footprint edges -- a biased proposal distorts exactly
    # the region of parameter space the detectability classifier needs to
    # learn well, and would also bias any downstream rate/volume estimate
    # that treats this sample as representative rather than re-weighting it.
    "dec_rad":            np.arcsin(np.random.uniform(-1.0, 1.0, size=args.events)),
    "eta":                loguniform(np.log(eta_min), np.log(eta_max), size=args.events),
    "alpha":              loguniform(np.log(alpha_min), np.log(alpha_max), size=args.events),
    "beta":               np.random.uniform(beta_min, beta_max, size=args.events),
    "t0_mjd_transient":   t0_mjd_transient,
}

# ── Model-specific extra parameters ─────────────────────────────────────────
# Ranges/distributions and citations per the population spec; see the
# top-level pipeline documentation for the full literature justification.
extra_data = {}

if MODEL == "cooling_envelope":
    pass  # no extra params beyond the shared set

elif MODEL == "stream_stream_tde":
    # reasonable value, same range as the other cooling-envelope-family rise
    # times; no independent stream-stream-specific literature constraint found.
    extra_data["peak_time"] = loguniform(np.log(5.0), np.log(40.0), size=args.events)
    extra_data["sigma_t"] = loguniform(np.log(5.0), np.log(30.0), size=args.events)
    # c1, del_omega, f, h_r, inc_tcool kept FIXED at redback's own default
    # values (DeltaFunction priors). These are internal stream-geometry/
    # efficiency factors from the Bonnerot & Lu stream-stream shock formalism
    # (Bonnerot & Lu 2022, MNRAS 511, 2147), not population-level free
    # parameters -- do not randomize them.
    extra_data["c1"] = np.full(args.events, 1.0)
    extra_data["del_omega"] = np.full(args.events, 2.0)
    extra_data["f"] = np.full(args.events, 0.2)
    extra_data["h_r"] = np.full(args.events, 0.2)
    extra_data["inc_tcool"] = np.full(args.events, 1.0)

elif MODEL == "tde_analytical":
    # Unbound ejecta mass from TDE disruption simulations:
    # Guillochon & Ramirez-Ruiz 2013, ApJ 767, 25.
    extra_data["mej"] = loguniform(np.log(0.1), np.log(5.0), size=args.events)
    # Typical TDE outflow velocities: Roth et al. 2020 review (Space Sci.
    # Rev. 216, 114) / observed TDE line widths.
    extra_data["vej"] = loguniform(np.log(3000.0), np.log(15000.0), size=args.events)
    # Electron-scattering opacity range appropriate for optically-thick
    # H/He-rich TDE ejecta.
    extra_data["kappa"] = np.random.uniform(0.1, 1.0, size=args.events)
    # reasonable value (redback default), no strong independent literature constraint
    extra_data["kappa_gamma"] = loguniform(np.log(0.0001), np.log(10000.0), size=args.events)
    # Observed TDE photospheric blackbody temperatures cluster hot,
    # ~2-4x10^4 K: van Velzen et al. 2021, ApJ 908, 4.
    extra_data["temperature_floor"] = np.random.uniform(10000.0, 50000.0, size=args.events)
    # reasonable value (redback default), no strong independent literature constraint
    extra_data["l0"] = loguniform(np.log(1e51), np.log(1e60), size=args.events)
    # reasonable value (redback default), no strong independent literature constraint
    extra_data["t_0_turn"] = loguniform(np.log(0.0001), np.log(500.0), size=args.events)

elif MODEL == "tde_fallback":
    # Disk viscous timescale: Mummery & Balbus 2020, MNRAS 492, 5655;
    # Mummery et al. 2024 fallback-disk modeling.
    extra_data["tvisc"] = loguniform(np.log(1.0), np.log(30.0), size=args.events)
    # reasonable value (redback default), no strong independent literature
    # constraint -- "beta-star" wind photosphere index; NOT the shared beta.
    extra_data["bb"] = np.random.uniform(0.0, 2.0, size=args.events)
    # reasonable value (redback default), no strong independent literature constraint
    extra_data["rph0"] = loguniform(np.log(1e-4), np.log(1e4), size=args.events)
    # reasonable value (redback default), no strong independent literature constraint
    extra_data["lphoto"] = np.random.uniform(0.0, 2.0, size=args.events)
    # leddlimit kept FIXED at redback's own DeltaFunction default -- an
    # Eddington-limiting switch/value, not a free population parameter.
    extra_data["leddlimit"] = np.full(args.events, 1.0)

filename = f"TDE_params_1E{TAG}_{args.SURVEY}_{MODEL}.h5"
file_path = os.path.join(output_dir, filename)

with h5py.File(file_path, "w") as f:
    for key, arr in shared_data.items():
        f.create_dataset(key, data=arr)
    for key, arr in extra_data.items():
        f.create_dataset(key, data=arr)

print(f"[DONE] Wrote {args.events} events for model '{MODEL}' -> {file_path}")
print(f"Shared columns: {list(shared_data.keys())}")
print(f"Extra columns for '{MODEL}': {list(extra_data.keys())}")

# ── Diagnostic plots ─────────────────────────────────────────────────────────
all_data = {**shared_data, **extra_data}
keys = list(all_data.keys())
n_keys = len(keys)
ncols = 3
nrows = int(np.ceil(n_keys / ncols))
fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
axs = np.atleast_2d(axs)
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for idx, key in enumerate(keys):
    r, c = divmod(idx, ncols)
    ax = axs[r, c]
    ax.hist(all_data[key], bins='auto', color=colors[idx % 10], alpha=0.7)
    ax.set_title(key)
# turn off unused axes
for idx in range(n_keys, nrows * ncols):
    r, c = divmod(idx, ncols)
    axs[r, c].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"TDE_params_1E{TAG}_{args.SURVEY}_{MODEL}.png"))
plt.close()
