#!/usr/bin/env python
"""
Fisher-matrix calibration of the per-event TDE measurement errors for the tde_fallback
model, on a small redshift-stratified subsample of DETECTED events (a few hundred instead
of millions).  fit_TDE_sigma.py then turns the result into sigma(z) for prep_pseudo_data.py.

Why a new script: fish_TDE.py / eval_TDE_fisher.py are written for cooling_envelope
(parameters eta, alpha, beta, stellar_mass and mbh_6 = lgM - 6).  tde_fallback instead takes
    mbh6 = 10**(lgM - 6)  [1e6 Msun], mstar = 1 (fixed), eta, tvisc, bb, rph0, lphoto,
    leddlimit = 1 (fixed), redshift, t0, ra, dec
exactly as built in training_TDE_detectability/calculate_SNR_TDE.py::build_theta.

Free (Fisher) parameters, in the coordinate used for the derivative:
    redshift (linear), mbh6 (log10, so sigma_u IS sigma_lgMBH), eta (log10), tvisc (log10),
    bb (linear), rph0 (log10), lphoto (linear)
The nuisance parameters get a Gaussian prior with sigma = std of that column in the loaded
catalogue (uniform / log-uniform sampling priors), which stabilises degenerate directions.
redshift and mbh6 get no prior.

Noise model: the survey's own photometric errors (e_magnitude from the telescope simulator)
on DETECTED points only.  NOTE this differs from fish_TDE.compute_fisher, which uses
0.05*|dm/dz| as the error; that is not a physical noise model.

STEP SIZES ARE UNTESTED for tde_fallback.  Run once with --N_EVENTS 3 --CHECK_STEPS 1 and look
at sigma_lgM vs sigma_lgM_alt (derivative step x2): they should agree to a few percent.

Usage (from the repo root, like eval_TDE_fisher.py):
    python fisher_analysis/calibrate_sigma_TDE_fallback.py --GALAXIES 100000 --SURVEY LSST \
        --BANDS lsstu lsstg lsstr lssti lsstz lssty --OBSERVING_WINDOW 700 \
        --N_EVENTS 200 --N_CHUNKS 10 --CHUNK 0
"""
import argparse
import logging
import os
import sys

import h5py
import numpy as np
if not hasattr(np, "trapezoid"):      # redback calls np.trapezoid (numpy>=2); this env has numpy 1.x
    np.trapezoid = np.trapz

sys.path.insert(0, os.path.abspath('./'))
from utils import ASTROPHYSICAL_DATA_DIR, FISHER_DATA_DIR

logging.getLogger('bilby').disabled = True
logging.getLogger('redback').disabled = True
from redback.simulate_transients import SimulateGenericTransient, SimulateOpticalTransient

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES", type=int, required=True)
parser.add_argument("--SURVEY", type=str, required=True, choices=["ZTF", "LSST"])
parser.add_argument("--BANDS", nargs="+", required=True)
parser.add_argument("--OBSERVING_WINDOW", type=float, required=True, help="days, as in config.sh")
parser.add_argument("--mbh_mass_method", default="use_lgMgal", choices=["use_sigma", "use_lgMgal"])
parser.add_argument("--nucleation_method", default="hannah", choices=["hannah", "neumayer"])
parser.add_argument("--occupation_method", default="uniform", choices=["uniform", "varying"])
parser.add_argument("--merger_model", default="oleary21")
parser.add_argument("--UNIVERSES", type=int, nargs="+", default=[0, 5, 10, 15, 20],
                    help="universe indices (trailing number in the catalogue file names) to draw events from")
parser.add_argument("--N_EVENTS", type=int, default=200, help="total subsample size over ALL chunks")
parser.add_argument("--N_ZBINS", type=int, default=12, help="log-spaced redshift strata")
parser.add_argument("--N_CHUNKS", type=int, default=1)
parser.add_argument("--CHUNK", type=int, default=0)
parser.add_argument("--SEED", type=int, default=0)
parser.add_argument("--MIN_POINTS", type=int, default=5, help="min detected photometric points per event")
parser.add_argument("--STEP_SCALE", type=float, default=1.0, help="multiplies every finite-difference step")
parser.add_argument("--CHECK_STEPS", type=int, default=0, choices=[0, 1],
                    help="1 = also recompute sigma_lgM with steps x3 (convergence check)")
args = parser.parse_args()

SURVEY = args.SURVEY
SIM_FN, SURVEY_STR = ((SimulateOpticalTransient.simulate_transient_in_ztf, "ztf") if SURVEY == "ZTF"
                      else (SimulateOpticalTransient.simulate_transient_in_rubin, "Rubin_10yr_baseline"))
BANDS = list(args.BANDS)

# ── Fisher parameter definitions ──────────────────────────────────────────────
# name, kind (lin | log), step (rel: fraction of value for lin; abs: in u units)
FREE = [
    ("redshift", "lin", ("rel", 1e-2)),
    ("mbh6",     "log", ("abs", 0.05)),
    ("eta",      "log", ("abs", 0.05)),
    ("tvisc",    "log", ("abs", 0.05)),
    ("bb",       "lin", ("abs", 0.01)),
    ("rph0",     "log", ("abs", 0.05)),
    ("lphoto",   "lin", ("abs", 0.01)),
]
NAMES = [f[0] for f in FREE]
P = len(FREE)
I_MBH = NAMES.index("mbh6")
I_Z = NAMES.index("redshift")
PRIORED = [n for n in NAMES if n not in ("redshift", "mbh6")]   # nuisance parameters

# ── Catalogue of detected events ──────────────────────────────────────────────
tag = f"{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}_{args.merger_model}"
ASTRO = f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}"
EXTRA = ["tvisc", "bb", "rph0", "lphoto"]

events = {k: [] for k in ["lgM", "z", "ra", "dec", "t0", "eta"] + EXTRA}
for u in args.UNIVERSES:
    with h5py.File(f"{ASTRO}/TDE_SNRs_{SURVEY}_{tag}_tde_fallback_{u}.h5", "r") as s, \
         h5py.File(f"{ASTRO}/all_TDE_params_data_{tag}_tde_fallback_{u}.h5", "r") as c:
        # the SNR file holds only detected events and lacks the tde_fallback extras; match rows
        # back to the full catalogue by their exact (z, lgM, t0) values.
        lookup = {(z, m, t): i for i, (z, m, t) in enumerate(zip(c["z_gal"][:], c["lgMBH_mass"][:], c["t0_mjd_transient"][:]))}
        zs, ms, ts = s["z_gal"][:], s["lgMBH_mass"][:], s["t0_mjd_transient"][:]
        idx = np.array([lookup.get((z, m, t), -1) for z, m, t in zip(zs, ms, ts)])
        ok = idx >= 0
        if (~ok).any():
            print(f"[warn] universe {u}: {(~ok).sum()} / {len(ok)} detected rows not matched to the catalogue (dropped)")
        idx = idx[ok]
        events["lgM"].append(ms[ok]); events["z"].append(zs[ok]); events["t0"].append(ts[ok])
        events["ra"].append(s["ra_rad"][:][ok]); events["dec"].append(s["dec_rad"][:][ok])
        events["eta"].append(s["eta"][:][ok])
        for k in EXTRA:
            events[k].append(c[k][:][idx])
events = {k: np.concatenate(v) for k, v in events.items()}
N_ALL = len(events["z"])
print(f"[INFO] {N_ALL} detected {SURVEY} events from universes {args.UNIVERSES}; "
      f"z range {events['z'].min():.3f}-{events['z'].max():.3f}")

# nuisance-parameter prior widths in u coordinates
U_SIGMA_PRIOR = {}
for n in PRIORED:
    col = events[n]
    kind = dict((f[0], f[1]) for f in FREE)[n]
    U_SIGMA_PRIOR[n] = float(np.std(np.log10(col) if kind == "log" else col))
PRIOR_PREC = np.zeros(P)
for n, s in U_SIGMA_PRIOR.items():
    PRIOR_PREC[NAMES.index(n)] = 1.0 / s ** 2

# ── redshift-stratified subsample (identical in every chunk job: same seed) ──
rng = np.random.default_rng(args.SEED)
edges = np.geomspace(events["z"].min(), events["z"].max() * 1.0001, args.N_ZBINS + 1)
bin_of = np.clip(np.digitize(events["z"], edges) - 1, 0, args.N_ZBINS - 1)
quota = int(np.ceil(args.N_EVENTS / args.N_ZBINS))
sel = []
for b in range(args.N_ZBINS):
    members = np.where(bin_of == b)[0]
    if len(members):
        sel.extend(rng.choice(members, size=min(quota, len(members)), replace=False))
sel = np.array(sel)
rng.shuffle(sel)
sel = sel[:args.N_EVENTS][args.CHUNK::args.N_CHUNKS]
print(f"[INFO] chunk {args.CHUNK}/{args.N_CHUNKS}: {len(sel)} events")


# ── model evaluation ──────────────────────────────────────────────────────────
def build_theta(i):
    t0 = float(events["t0"][i])
    return {
        "redshift": float(events["z"][i]), "t0_mjd_transient": t0, "t0": t0,
        "ra": float(events["ra"][i]), "dec": float(events["dec"][i]),
        "mbh6": float(10 ** (events["lgM"][i] - 6)),     # NOT lgM - 6
        "mstar": 1.0, "eta": float(events["eta"][i]),
        "tvisc": float(events["tvisc"][i]), "bb": float(events["bb"][i]),
        "rph0": float(events["rph0"][i]), "lphoto": float(events["lphoto"][i]),
        "leddlimit": 1.0,
    }


def observe(theta):
    """Photometry the survey would take: per band (time, err) of DETECTED points, time-sorted."""
    sim = SIM_FN(model="tde_fallback", survey=SURVEY_STR, parameters=theta, model_kwargs={},
                 end_transient_time=args.OBSERVING_WINDOW, snr_threshold=5.0, add_source_noise=False)
    df = sim.observations[['time (days)', 'magnitude', 'e_magnitude', 'band', 'detected']]
    out = {}
    for b in BANDS:
        g = df[(df["band"] == b) & (df["detected"].astype(bool))]
        g = g[np.isfinite(g["e_magnitude"]) & (g["e_magnitude"] > 0)].sort_values("time (days)")
        if len(g):
            out[b] = (g["time (days)"].to_numpy(float), g["e_magnitude"].to_numpy(float))
    return out


def model_mags(theta, band, times):
    o = SimulateGenericTransient(model="tde_fallback", parameters=theta, times=times,
                                 data_points=len(times),
                                 model_kwargs={'bands': (band,), 'output_format': 'magnitude'},
                                 multiwavelength_transient=True, noise_term=0.0)
    d = o.data
    return d[d["band"] == band].sort_values("time")["true_output"].to_numpy(float)


def shifted(theta, name, kind, du):
    """theta with parameter `name` moved by du in its derivative coordinate."""
    t = dict(theta)
    t[name] = theta[name] + du if kind == "lin" else 10 ** (np.log10(theta[name]) + du)
    return t


def fisher(theta, obs, step_mult=1.0):
    F = np.zeros((P, P))
    for b, (times, err) in obs.items():
        J = np.zeros((P, len(times)))
        for k, (name, kind, (smode, s)) in enumerate(FREE):
            h = step_mult * args.STEP_SCALE * (s * max(abs(theta[name]), 1e-3) if smode == "rel" else s)
            mp = model_mags(shifted(theta, name, kind, +h), b, times)
            mm = model_mags(shifted(theta, name, kind, -h), b, times)
            J[k] = (mp - mm) / (2 * h)
        good = np.all(np.isfinite(J), axis=0)
        J, e = J[:, good], err[good]
        F += (J / e ** 2) @ J.T
    return F


def covariance(F, with_prior=True):
    Ft = F + np.diag(PRIOR_PREC) if with_prior else F
    return np.linalg.pinv(0.5 * (Ft + Ft.T)), np.linalg.cond(Ft)


rows = []
for i in sel:
    try:
        theta = build_theta(i)
        obs = observe(theta)
        npts = sum(len(v[0]) for v in obs.values())
        if npts < args.MIN_POINTS:
            print(f"[skip] event {i}: only {npts} detected points")
            continue
        F = fisher(theta, obs)
        C, cond = covariance(F)
        C0, _ = covariance(F, with_prior=False)
        row = dict(idx=i, z=events["z"][i], lgM=events["lgM"][i], npts=npts, cond=cond,
                   sigma=np.sqrt(np.clip(np.diag(C), 0, None)),
                   sigma_noprior=np.sqrt(np.clip(np.diag(C0), 0, None)),
                   sigma_lgM_alt=np.nan)
        if args.CHECK_STEPS:
            Ca, _ = covariance(fisher(theta, obs, step_mult=2.0))
            row["sigma_lgM_alt"] = float(np.sqrt(max(Ca[I_MBH, I_MBH], 0)))
            Cb, _ = covariance(fisher(theta, obs, step_mult=0.4))
            print(f"    step check sigma_lgM: x0.4={np.sqrt(max(Cb[I_MBH, I_MBH], 0)):.3f} "
                  f"x1={np.sqrt(max(C[I_MBH, I_MBH], 0)):.3f} x2={row['sigma_lgM_alt']:.3f}; "
                  f"median e_mag={np.median(np.concatenate([v[1] for v in obs.values()])):.3f}")
        rows.append(row)
        print(f"[ok] event {i} z={row['z']:.3f} lgM={row['lgM']:.2f} npts={npts} "
              f"sigma_lgM={row['sigma'][I_MBH]:.3f} (no prior {row['sigma_noprior'][I_MBH]:.3f}"
              f"{', step x2 ' + format(row['sigma_lgM_alt'], '.3f') if args.CHECK_STEPS else ''}) "
              f"sigma_z={row['sigma'][I_Z]:.4f} cond={cond:.1e}", flush=True)
    except Exception as e:                      # one bad event must not kill the chunk
        print(f"[fail] event {i}: {type(e).__name__}: {e}", flush=True)

outdir = f"{FISHER_DATA_DIR}/{args.GALAXIES}/sigma_calibration"
os.makedirs(outdir, exist_ok=True)
path = f"{outdir}/{SURVEY}_chunk{args.CHUNK:03d}.h5"
with h5py.File(path, "w") as f:
    f.attrs["param_names"] = NAMES
    f.attrs["survey"] = SURVEY
    f.attrs["observing_window"] = args.OBSERVING_WINDOW
    f.attrs["n_requested"] = len(sel)
    for key in ("idx", "z", "lgM", "npts", "cond", "sigma_lgM_alt"):
        f.create_dataset(key, data=np.array([r[key] for r in rows], dtype=float))
    f.create_dataset("sigma", data=np.array([r["sigma"] for r in rows]).reshape(-1, P))
    f.create_dataset("sigma_noprior", data=np.array([r["sigma_noprior"] for r in rows]).reshape(-1, P))
print(f"wrote {len(rows)}/{len(sel)} events -> {path}")
