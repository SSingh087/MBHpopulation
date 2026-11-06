"""priors.py -- Lambda definition and prior box (light module: no heavy imports, shared by simulator and flow code)."""
import numpy as np

LAMBDA_NAMES = ["A", "B", "scatter", "logM0", "delta", "theta"]
FIDUCIAL = dict(A=7.87, B=4.55, scatter=0.55, logM0=9.46, delta=3.91, theta=0.80)
PRIOR_BOX = dict(A=(7.57, 8.17), B=(3.25, 5.85), scatter=(0.35, 0.75),
                 logM0=(8.5, 10.5), delta=(0.5, 20.0), theta=(0.2, 2.0))   # delta is log-uniform
EVENT_KEYS = ["lgMBH", "spin", "z", "e_plunge", "lgMgal", "sigma_km_s"]       # EMRI sources
TDE_KEYS = ["lgMBH", "z", "lgMgal", "sigma_km_s"]                               # TDE sources (spin not modelled)


def sample_prior(rng):
    """One Lambda draw from the v1 prior box (delta log-uniform, rest uniform)."""
    lam = {}
    for k, (lo, hi) in PRIOR_BOX.items():
        lam[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if k == "delta" else float(rng.uniform(lo, hi))
    return lam
