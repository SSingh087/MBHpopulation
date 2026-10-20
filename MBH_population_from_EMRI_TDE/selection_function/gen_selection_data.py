"""Training set of the selection function: log alpha(Lambda) for hyperparameters Lambda drawn from the prior.

For each Lambda, events are drawn from the population model, completed with the remaining injection
parameters as in data_generation/, passed through the detection network of the channel, and alpha is the
detected fraction. Run one chunk per condor job and concatenate the chunks before training.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath('./'))
sys.path.insert(0, os.path.abspath('./hierarchical_inference'))
sys.path.insert(0, os.path.abspath('./selection_function'))
from common import CHANNELS, channel_names, load_prior_bounds, emri_detection_probability, tde_detection_probability
from cosmology import CosmologyModel
from utils import make_dist, get_nested_from_flat

parser = argparse.ArgumentParser()
parser.add_argument("--POPULATION", type=str, required=True, choices=["A", "B"])
parser.add_argument("--CHANNEL", type=str, required=True, choices=CHANNELS)
parser.add_argument("--N_LAMBDA", type=int, default=2000, help="hyperparameter points in this chunk")
parser.add_argument("--N_EVENTS", type=int, default=20000, help="events drawn per hyperparameter point")
parser.add_argument("--CHUNK", type=int, default=0)
parser.add_argument("--SEED", type=int, default=0)
parser.add_argument("--T_OBS_EMRI", type=float, default=6.0)
parser.add_argument("--MODEL_PATH", type=str, help="detection network (EMRI: SNR network)")
parser.add_argument("--SCALING_PATH", type=str, help="EMRI snr_scaling.json")
parser.add_argument("--SNR_THRESHOLD", type=float, default=20.0)
parser.add_argument("--OUT_DIR", type=str, default="./data/selection_function")
args = parser.parse_args()

rng = np.random.default_rng(args.SEED + 1000 * args.CHUNK)
cosmo = CosmologyModel()
popdist = make_dist(args.POPULATION)
names = channel_names(args.POPULATION, args.CHANNEL)
bounds = load_prior_bounds(args.POPULATION)


def complete_emri(draws):
    """Add the remaining EMRI injection parameters to population draws (as in gen_EMRI_params.py)."""
    from few.trajectory.inspiral import EMRIInspiral
    from few.utils.utility import get_separatrix
    traj = EMRIInspiral(func="pn5")
    n = len(draws["z_gal"])
    ev = {k: draws[k] for k in ["lgMBH_mass", "MBHspin", "z_gal"]}
    ev["Y0"] = rng.uniform(0.1, 0.7, n)
    ev["T_SIGNAL_duration_years"] = rng.uniform(1.0, args.T_OBS_EMRI, n)
    ev["qS"], ev["qK"] = np.arccos(rng.uniform(-1, 1, n)), np.arccos(rng.uniform(-1, 1, n))
    ev["phiS"], ev["phiK"] = rng.uniform(0, 2 * np.pi, (2, n))
    ev["distance_Gpc"] = cosmo.luminosity_distance(ev["z_gal"], unit=__import__("astropy.units", fromlist=["Gpc"]).Gpc)
    ev["e0"], ev["p0"] = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        try:   # backward integration from plunge, as in gen_EMRI_params.py
            p_pl = get_separatrix(ev["MBHspin"][i], draws["e_pl"][i], ev["Y0"][i])
            _, p, e, *_ = traj(10 ** ev["lgMBH_mass"][i], 10.0, ev["MBHspin"][i], p_pl, draws["e_pl"][i], ev["Y0"][i],
                               dt=10.0, T=ev["T_SIGNAL_duration_years"][i], integrate_backwards=True)
            ev["p0"][i], ev["e0"][i] = p[-1], e[-1]
        except Exception:
            pass
    return ev


def detection_probability(draws):
    """Per-event detection probability of the channel for population draws."""
    if args.CHANNEL == "EMRI":
        return emri_detection_probability(complete_emri(draws), args)
    return tde_detection_probability(draws, args.CHANNEL.split("_")[1], args)


lam = np.array([[rng.uniform(*bounds[n]) for n in names] for _ in range(args.N_LAMBDA)])
log_alpha = np.empty(args.N_LAMBDA)
log_alpha_err = np.empty(args.N_LAMBDA)

for k in range(args.N_LAMBDA):
    flat = {n: float(v) for n, v in zip(names, lam[k])}
    # components without a free hyperparameter in this channel are fixed at the prior mid-point
    for n, (lo, hi) in bounds.items():
        flat.setdefault(n, 0.5 * (lo + hi))
    nested, _ = get_nested_from_flat(flat, args.POPULATION, "use_sigma", "hannah", "uniform")
    draws = {key: np.asarray(v.detach().cpu()) for key, v in popdist.draw_samples(nested, size=args.N_EVENTS).items()}
    p = detection_probability(draws)
    alpha = max(p.mean(), 1e-12)
    log_alpha[k] = np.log(alpha)
    log_alpha_err[k] = np.sqrt((1 - alpha) / (alpha * len(p)))   # binomial error on log alpha

os.makedirs(args.OUT_DIR, exist_ok=True)
out = f"{args.OUT_DIR}/data_{args.POPULATION}_{args.CHANNEL}_{args.CHUNK:04d}.npz"
np.savez(out, lam=lam, log_alpha=log_alpha, log_alpha_err=log_alpha_err, names=np.array(names))
print(f"wrote {args.N_LAMBDA} points to {out}")
