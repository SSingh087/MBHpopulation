#!/usr/bin/env python
"""
TDE Fisher bank: Fisher matrices for tde_fallback events drawn from WIDE priors over the
event-level parameters (not from a Lambda-specific catalogue), so that a conditional flow
sigma(theta_event) can be trained once and reused for every universe.

Adapted from EMRI_TDE_population/fisher_analysis/calibrate_sigma_TDE_fallback.py (same model,
same free parameters, same finite-difference scheme, same survey-error noise model).  The only
change is where the events come from: sampled here, read from a catalogue there.

Free (Fisher) parameters (derivative coordinate):
    redshift (lin), mbh6 (log10 -> sigma is sigma_lgMBH), eta (log10), tvisc (log10),
    bb (lin), rph0 (log10), lphoto (lin)
Nuisance parameters (eta, tvisc, bb, rph0, lphoto) get a Gaussian prior whose width is the std
of their sampling prior (computed analytically here, identical for every chunk).

Output: one npz per chunk with the event parameters, the 7x7 data Fisher matrix F (no prior),
the prior precision, the 7x7 posterior covariance (with prior), the number of detected photometric
points and the wall time per event.  Failed / uninformative events are kept with ok=0.

Usage:  python tde_fisher_bank.py --survey LSST --chunk 0 --n_events 100 --out_dir DIR
"""
import argparse, os, sys, time, logging
import numpy as np

if not hasattr(np, "trapezoid"):          # redback calls np.trapezoid (numpy>=2); env has numpy 1.x
    np.trapezoid = np.trapz
logging.getLogger('bilby').disabled = True
logging.getLogger('redback').disabled = True
from redback.simulate_transients import SimulateGenericTransient, SimulateOpticalTransient

p = argparse.ArgumentParser()
p.add_argument("--survey", required=True, choices=["ZTF", "LSST"])
p.add_argument("--chunk", type=int, required=True)
p.add_argument("--n_events", type=int, default=100)
p.add_argument("--out_dir", required=True)
p.add_argument("--master_seed", type=int, default=11)
p.add_argument("--lgM_range", type=float, nargs=2, default=[4.0, 7.0])
p.add_argument("--z_max", type=float, default=None, help="default: 0.5 ZTF, 4.0 LSST (classifier training range)")
p.add_argument("--z_min", type=float, default=0.01)
p.add_argument("--t0_range", type=float, nargs=2, default=None,
               help="MJD; default = the windows the detectability classifiers were trained on: ZTF 58200-61396, LSST 61220-64872")
p.add_argument("--window_days", type=float, default=None, help="default: 500 ZTF, 700 LSST (old config.sh)")
p.add_argument("--p_min", type=float, default=0.01, help="classifier pre-filter: skip draws with P(detect) < p_min (production cut is 0.5)")
p.add_argument("--max_prefilter_tries", type=int, default=500)
p.add_argument("--proposal", choices=["prior", "mixed"], default="prior",
               help="prior: (lgM,z) from the wide prior; mixed: a fraction is drawn from the source catalogue (all 10k universes, equal weight per universe)")
p.add_argument("--catalogue_fraction", type=float, default=0.7, help="share of (lgM,z) draws taken from the catalogue when --proposal mixed")
p.add_argument("--proposal_file", default="/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference/simulations/consolidated/tde_proposal_subsample.npz")
p.add_argument("--min_points", type=int, default=5)
p.add_argument("--max_wall_s", type=float, default=5400)
p.add_argument("--step_scale", type=float, default=1.0)
p.add_argument("--check_steps", type=int, default=0, help="1: also store sigma_lgM with steps x2 and x0.4")
p.add_argument("--scan", type=int, default=0, help="1: step-size scan mode (stores F_kk vs step multiplier, no full Fisher)")
p.add_argument("--scan_mults", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
a = p.parse_args()

SURVEY = a.survey
Z_MAX = a.z_max or (0.5 if SURVEY == "ZTF" else 4.0)
T0 = a.t0_range or ((58200., 61396.) if SURVEY == "ZTF" else (61220., 64872.))
WINDOW = a.window_days or (500. if SURVEY == "ZTF" else 700.)
BANDS = ["ztfg", "ztfr", "ztfi"] if SURVEY == "ZTF" else \
        ["lsstu", "lsstg", "lsstr", "lssti", "lsstz", "lssty"]
SIM_FN, SURVEY_STR = ((SimulateOpticalTransient.simulate_transient_in_ztf, "ztf") if SURVEY == "ZTF"
                      else (SimulateOpticalTransient.simulate_transient_in_rubin, "Rubin_10yr_baseline"))

# name, kind, (step mode, step)
FREE = [("redshift", "lin", ("rel", 1e-2)), ("mbh6", "log", ("abs", 0.05)), ("eta", "log", ("abs", 0.05)),
        ("tvisc", "log", ("abs", 0.05)), ("bb", "lin", ("abs", 0.01)), ("rph0", "log", ("abs", 0.05)),
        ("lphoto", "lin", ("abs", 0.01))]
NAMES = [f[0] for f in FREE]; P = len(FREE); I_MBH = 1; I_Z = 0

# sampling priors of the nuisance parameters (match gen_TDE_params.py / gen_TDE_params.sh)
LOGU = {"eta": (0.001, 0.1), "tvisc": (1.0, 30.0), "rph0": (1e-4, 1e4)}
UNIF = {"bb": (0.0, 2.0), "lphoto": (0.0, 2.0)}
PRIOR_PREC = np.zeros(P)
for n, (lo, hi) in LOGU.items():
    PRIOR_PREC[NAMES.index(n)] = 1.0 / ((np.log10(hi) - np.log10(lo)) ** 2 / 12.0)   # std in log10 units
for n, (lo, hi) in UNIF.items():
    PRIOR_PREC[NAMES.index(n)] = 1.0 / ((hi - lo) ** 2 / 12.0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tde_classifier import TDEClassifier
CLF = TDEClassifier(SURVEY)
LOGU_CLF = {"alpha": (0.01, 0.1)}; UNIF_CLF = {"beta": (1.001, 30.0)}     # classifier inputs only (ignored by tde_fallback itself)

rng = np.random.default_rng([a.master_seed, a.chunk, 0 if SURVEY == "ZTF" else 1])


if a.proposal == "mixed":
    _c = np.load(a.proposal_file); _m = (_c["lgM"] >= a.lgM_range[0]) & (_c["lgM"] <= a.lgM_range[1]) & (_c["z"] >= a.z_min) & (_c["z"] <= Z_MAX)
    CAT_LGM, CAT_Z = _c["lgM"][_m].astype(float), _c["z"][_m].astype(float)
    print(f"[proposal] mixed: {len(CAT_Z)} catalogue (lgM,z) pairs within the survey range; catalogue_fraction={a.catalogue_fraction}", flush=True)


def draw_raw():
    d = dict(lgM=rng.uniform(*a.lgM_range),
             z=float(np.exp(rng.uniform(np.log(a.z_min), np.log(Z_MAX)))),
             ra=rng.uniform(0, 2 * np.pi), dec=float(np.arcsin(rng.uniform(-1, 1))),
             t0=rng.uniform(*T0))
    d["from_catalogue"] = 0.0
    if a.proposal == "mixed" and rng.random() < a.catalogue_fraction:
        j = rng.integers(len(CAT_Z)); d["lgM"], d["z"], d["from_catalogue"] = float(CAT_LGM[j]), float(CAT_Z[j]), 1.0
    for n, (lo, hi) in LOGU.items():
        d[n] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    for n, (lo, hi) in UNIF.items():
        d[n] = rng.uniform(lo, hi)
    for n, (lo, hi) in LOGU_CLF.items():
        d[n] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    for n, (lo, hi) in UNIF_CLF.items():
        d[n] = rng.uniform(lo, hi)
    return d


def p_detect(d):
    return float(CLF.prob({"lgMBH": [d["lgM"]], "z_gal": [d["z"]], "ra_rad": [d["ra"]], "dec_rad": [d["dec"]], "eta": [d["eta"]],
                           "alpha": [d["alpha"]], "beta": [d["beta"]], "t0_mjd_transient": [d["t0"]], "tvisc": [d["tvisc"]],
                           "bb": [d["bb"]], "rph0": [d["rph0"]], "lphoto": [d["lphoto"]]})[0])


def draw():
    """Draw from the wide event prior; skip (cheaply) events the classifier says essentially cannot be detected."""
    for n in range(a.max_prefilter_tries):
        d = draw_raw(); d["p_detect"] = p_detect(d)
        if d["p_detect"] >= a.p_min:
            d["n_prefilter_draws"] = n + 1
            return d
    d["n_prefilter_draws"] = a.max_prefilter_tries
    return d


def build_theta(d):
    return {"redshift": d["z"], "t0_mjd_transient": d["t0"], "t0": d["t0"], "ra": d["ra"], "dec": d["dec"],
            "mbh6": 10 ** (d["lgM"] - 6), "mstar": 1.0, "eta": d["eta"], "tvisc": d["tvisc"], "bb": d["bb"],
            "rph0": d["rph0"], "lphoto": d["lphoto"], "leddlimit": 1.0}


def observe(theta):
    sim = SIM_FN(model="tde_fallback", survey=SURVEY_STR, parameters=theta, model_kwargs={},
                 end_transient_time=WINDOW, snr_threshold=5.0, add_source_noise=False)
    df = sim.observations[['time (days)', 'magnitude', 'e_magnitude', 'band', 'detected']]
    out = {}
    for b in BANDS:
        g = df[(df["band"] == b) & (df["detected"].astype(bool))]
        g = g[np.isfinite(g["e_magnitude"]) & (g["e_magnitude"] > 0)].sort_values("time (days)")
        if len(g):
            out[b] = (g["time (days)"].to_numpy(float), g["e_magnitude"].to_numpy(float))
    return out


from redback.transient_models.tde_models import tde_fallback
_MODEL_KEYS = ("redshift", "mbh6", "mstar", "tvisc", "bb", "eta", "leddlimit", "rph0", "lphoto")


def model_mags(theta, band, times):
    """Deterministic tde_fallback magnitudes at EXACTLY `times`.  NOTE: SimulateGenericTransient must NOT be used here:
    it re-draws `data_points` times at random (with replacement) on every call, so repeated evaluations at identical
    parameters differ by up to ~1.5 mag and finite differences measure that resampling (the old Fisher code had this)."""
    kw = {k: theta[k] for k in _MODEL_KEYS}
    return np.asarray(tde_fallback(np.asarray(times, float), output_format="magnitude",
                                   bands=np.array([band] * len(times)), **kw), float)


def shifted(theta, name, kind, du):
    t = dict(theta)
    t[name] = theta[name] + du if kind == "lin" else 10 ** (np.log10(theta[name]) + du)
    return t


BOUNDS = {"bb": (0.0, 2.0)}          # tde_fallback raises ValueError outside b in [0, 2]


def d_dtheta(theta, name, kind, h, band, times):
    """dm/du in the derivative coordinate; central difference, one-sided if the step would leave the valid range."""
    lo, hi = BOUNDS.get(name, (-np.inf, np.inf)); x = theta[name]
    if kind == "lin" and (x - h < lo or x + h > hi):
        if x + h <= hi:
            return (model_mags(shifted(theta, name, kind, +h), band, times) - model_mags(theta, band, times)) / h
        if x - h >= lo:
            return (model_mags(theta, band, times) - model_mags(shifted(theta, name, kind, -h), band, times)) / h
        return np.full(len(times), np.nan)
    return (model_mags(shifted(theta, name, kind, +h), band, times) - model_mags(shifted(theta, name, kind, -h), band, times)) / (2 * h)


def fisher(theta, obs, mult=1.0):
    F = np.zeros((P, P))
    for b, (times, err) in obs.items():
        J = np.zeros((P, len(times)))
        for k, (name, kind, (smode, s)) in enumerate(FREE):
            h = mult * a.step_scale * (s * max(abs(theta[name]), 1e-3) if smode == "rel" else s)
            J[k] = d_dtheta(theta, name, kind, h, b, times)
        good = np.all(np.isfinite(J), axis=0)
        J, e = J[:, good], err[good]
        F += (J / e ** 2) @ J.T
    return F


def scan_diag(theta, obs):
    """F_kk (data only, no prior) vs step multiplier for every parameter: out[k, m]. A good step lies on a
    plateau of F_kk; quantisation noise shows up at small steps, curvature/nonlinearity at large steps."""
    out = np.full((P, len(a.scan_mults)), np.nan)
    for k, (name, kind, (smode, s0)) in enumerate(FREE):
        for im, m in enumerate(a.scan_mults):
            h = m * (s0 * max(abs(theta[name]), 1e-3) if smode == "rel" else s0)
            tot = 0.0
            for b, (times, err) in obs.items():
                J = d_dtheta(theta, name, kind, h, b, times)
                g = np.isfinite(J); tot += np.sum((J[g] / err[g]) ** 2)
            out[k, im] = tot
    return out


def cov_from(F):
    Ft = F + np.diag(PRIOR_PREC)
    return np.linalg.pinv(0.5 * (Ft + Ft.T)), np.linalg.cond(Ft)


KEYS = ["lgM", "z", "ra", "dec", "t0", "eta", "tvisc", "bb", "rph0", "lphoto", "alpha", "beta", "p_detect", "n_prefilter_draws", "from_catalogue"]
rec = {k: [] for k in KEYS}; Fs, Cs, npts_l, ok_l, wall_l, cond_l, sa_l, scan_l = [], [], [], [], [], [], [], []
t_start = time.time()
for i in range(a.n_events):
    if time.time() - t_start > a.max_wall_s:
        print(f"[stop] wall budget reached after {i} events", flush=True); break
    d = draw(); t1 = time.time(); sc = np.full((P, len(a.scan_mults)), np.nan)
    F = np.full((P, P), np.nan); C = np.full((P, P), np.nan); npts = 0; ok = 0; cond = np.nan; salt = np.full(2, np.nan)
    try:
        th = build_theta(d); obs = observe(th)
        npts = sum(len(v[0]) for v in obs.values())
        if npts >= a.min_points and a.scan:
            sc = scan_diag(th, obs); ok = 1
            print(f"[scan] {i} z={d['z']:.3f} lgM={d['lgM']:.2f} npts={npts} {time.time()-t1:.1f}s", flush=True)
        elif npts >= a.min_points:
            F = fisher(th, obs); C, cond = cov_from(F); ok = 1
            if a.check_steps:
                for j, m in enumerate((2.0, 0.4)):
                    Ca, _ = cov_from(fisher(th, obs, m)); salt[j] = np.sqrt(max(Ca[I_MBH, I_MBH], 0))
            print(f"[ok] {i} z={d['z']:.3f} lgM={d['lgM']:.2f} npts={npts} sig_lgM={np.sqrt(C[I_MBH,I_MBH]):.3f} "
                  f"sig_z={np.sqrt(C[I_Z,I_Z]):.4f} cond={cond:.1e} {time.time()-t1:.1f}s", flush=True)
        else:
            print(f"[skip] {i} only {npts} detected points", flush=True)
    except Exception as e:
        print(f"[fail] {i}: {type(e).__name__}: {e}", flush=True)
    for k in KEYS: rec[k].append(d[k])
    Fs.append(F); Cs.append(C); npts_l.append(npts); ok_l.append(ok); wall_l.append(time.time() - t1)
    cond_l.append(cond); sa_l.append(salt); scan_l.append(sc)

os.makedirs(a.out_dir, exist_ok=True)
path = f"{a.out_dir}/{SURVEY}_chunk_{a.chunk:04d}.npz"
np.savez(path, **{k: np.array(v) for k, v in rec.items()}, F=np.array(Fs), C=np.array(Cs), npts=np.array(npts_l),
         ok=np.array(ok_l), wall_s=np.array(wall_l), cond=np.array(cond_l), sigma_lgM_steps_x2_x04=np.array(sa_l),
         scan_Fkk=np.array(scan_l), scan_mults=np.array(a.scan_mults), prior_precision=PRIOR_PREC, param_names=np.array(NAMES), survey=SURVEY, window_days=WINDOW)
print(f"wrote {len(ok_l)} events ({int(np.sum(ok_l))} ok) -> {path}")
