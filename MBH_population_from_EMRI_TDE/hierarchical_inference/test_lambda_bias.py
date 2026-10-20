#!/usr/bin/env python
"""
ONE job of the lambda-bias test: a single (population, scenario, survey).
combine_lambda_bias.py merges all jobs into the summary table and plots.

Question: do the TDE mock posteriors (Gaussians centred on the TRUE mass, see
make_posteriors in prep_pseudo_data.py) bias the mass hyperparameters?

Three versions of the same events are fitted (mass likelihood only):
  as_built   posterior samples exactly as stored (centred on the true mass)
  exact      each event collapsed to its true mass (no measurement error)
  scattered  obs = true + sigma*eps, posterior = truncnorm centred on obs

Speed: each event's samples are histogrammed onto NB mass bins, so the likelihood
over the WHOLE hyperparameter grid is one matrix product
        sum_e log( W[e,:] @ F[:,grid] )
instead of a python loop over the grid.

Usage:  python test_lambda_bias.py --POP A --SCEN use_lgMgal_hannah_uniform --SURVEY LSST
Output: lambda_bias_test/parts/{POP}_{SCEN}_{SURVEY}.npz
"""
import argparse
import os

import h5py
import numpy as np
from scipy.stats import truncnorm

from utils import mass_scenarios_A, mass_scenarios_B, SCENARIO_KEY

parser = argparse.ArgumentParser()
parser.add_argument("--POP", required=True, choices=["A", "B"])
parser.add_argument("--SCEN", required=True, help="e.g. use_lgMgal_hannah_uniform")
parser.add_argument("--SURVEY", required=True, choices=["ZTF", "LSST"])
parser.add_argument("--GALAXIES", default="100000")
parser.add_argument("--NSAMP", type=int, default=300, help="posterior samples per event")
parser.add_argument("--NREP", type=int, default=5, help="noise realisations for 'scattered'")
parser.add_argument("--SEED", type=int, default=0)
args = parser.parse_args()

base = f"./data/pseudo_inference_data/{args.GALAXIES}"
outdir = "lambda_bias_test/parts"
os.makedirs(outdir, exist_ok=True)
rng = np.random.default_rng(args.SEED)

LO, HI, NB = 4.0, 7.0, 300
EDGES = np.linspace(LO, HI, NB + 1)
CEN = 0.5 * (EDGES[1:] + EDGES[:-1])
DX = EDGES[1] - EDGES[0]

# hyperparameter grids (priors: A xc [4,7], lam [6.5,22];  B lam [-3,3])
if args.POP == "A":
    XC = np.linspace(4.5, 7.0, 51)
    LAM = np.linspace(6.5, 22.0, 63)
    xx, ll = np.meshgrid(XC, LAM, indexing="ij")
    xx, ll = xx.ravel(), ll.ravel()
    lp = ll[None] * np.log(ll[None] * CEN[:, None] / xx[None]) - ll[None] * CEN[:, None] / xx[None]
    F = np.exp(lp - lp.max(0, keepdims=True))
    F = (F / (F.sum(0, keepdims=True) * DX)).astype(np.float32)          # (NB, ngrid)
    grid_shape = (len(XC), len(LAM))
else:
    LAM = np.linspace(-3.0, 3.0, 301)
    lp = LAM[None] * np.log(CEN[:, None])
    F = np.exp(lp - lp.max(0, keepdims=True))
    F = (F / (F.sum(0, keepdims=True) * DX)).astype(np.float32)
    grid_shape = (len(LAM),)

Z_REF = {"ZTF": 0.1, "LSST": 0.2}[args.SURVEY]     # same as prep_pseudo_data.py
SIG_REF = 0.3


def to_weights(x):
    """x: (events, samples) -> W: (events, NB) fraction of each event's samples per bin."""
    E, S = x.shape
    idx = np.clip(((x - LO) / DX).astype(np.int64), 0, NB - 1)
    flat = (np.arange(E)[:, None] * NB + idx).ravel()
    return (np.bincount(flat, minlength=E * NB).reshape(E, NB) / S).astype(np.float32)


def logL_grid(W, chunk=4000):
    L = np.zeros(F.shape[1])
    for s in range(0, W.shape[0], chunk):
        L += np.log(W[s:s + chunk] @ F + 1e-30).sum(0, dtype=np.float64)
    return L.reshape(grid_shape)


path = f"{base}/true_data_TDE_{args.SURVEY}_pseudo_data_{args.POP}_{args.SCEN}.h5"
with h5py.File(path, "r") as f:
    true = f["mbh_6"][:].astype(np.float64) + 6.0
    z = f["redshift"][:].astype(np.float64)
    post = f["mbh_6_posterior"][:, :args.NSAMP].astype(np.float64) + 6.0
E = len(true)
sig = SIG_REF * (1.0 + z / Z_REF)                  # exact per-event sigma used by the generator

L_as = logL_grid(to_weights(post))
L_ex = logL_grid(to_weights(true[:, None]))        # one "sample" per event = true value

L_sc = []
for r in range(args.NREP):
    obs = true + sig * rng.standard_normal(E)
    a, b = (LO - obs) / sig, (HI - obs) / sig
    xs = truncnorm.rvs(a[:, None], b[:, None], loc=obs[:, None], scale=sig[:, None],
                       size=(E, args.NSAMP), random_state=rng)
    L_sc.append(logL_grid(to_weights(xs)))
L_sc = np.array(L_sc)

key = next(k for k, v in SCENARIO_KEY.items() if "_".join(k) == args.SCEN)
t = (mass_scenarios_A if args.POP == "A" else mass_scenarios_B)[SCENARIO_KEY[key]]
truth = np.array([t["xc"], t["lam_schechter"]]) if args.POP == "A" else np.array([t["lam"]])

np.savez(f"{outdir}/{args.POP}_{args.SCEN}_{args.SURVEY}.npz",
         L_as=L_as, L_ex=L_ex, L_sc=L_sc, LAM=LAM, XC=(XC if args.POP == "A" else np.array([])),
         truth=truth, N=E, sig_mean=sig.mean(), pop=args.POP, scen=args.SCEN, survey=args.SURVEY)
print(f"done {args.POP} {args.SCEN} {args.SURVEY}  N={E}  <sigma>={sig.mean():.2f}")
