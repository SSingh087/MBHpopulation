#!/usr/bin/env python
"""run_nessai_tde.py -- population inference from one mock TDE catalogue with nessai.

Parameters Lambda = (A, B, scatter, logM0, log delta, theta); prior = the training box (delta log-uniform).  Likelihood = TDELikelihood (tde_pop_lib.py).
  python run_nessai_tde.py --mock mock.npz --out runs/ztf_seed1 [--nlive 1000] [--device cuda]
Writes nessai's result file plus <out>/posterior.npz (equal-weight samples in Lambda, log-evidence) and prints the truth-coverage summary.
"""
import argparse, os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tde_pop_lib import *
from nessai.model import Model
from nessai.flowsampler import FlowSampler

ap = argparse.ArgumentParser(); ap.add_argument("--mock", nargs="+", required=True, help="one or more mock catalogues (one per survey); the likelihood is the product over them"); ap.add_argument("--out", required=True); ap.add_argument("--nlive", type=int, default=1000)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--threads", type=int, default=4, help="torch/CPU threads"); ap.add_argument("--max_iter", type=int, default=None, help="stop after this many nested-sampling iterations (benchmark runs)"); ap.add_argument("--refine", type=int, default=2); ap.add_argument("--no_resume", action="store_true")
ap.add_argument("--flow_device", default=None, help="device of nessai's own proposal flow (default cpu)")
a = ap.parse_args()
Ms = [np.load(f, allow_pickle=True) for f in a.mock]; surveys = [str(M["survey"]) for M in Ms]; M = Ms[0]
likes = [TDELikelihood(TDEPopulation(sv, a.device, refine=a.refine), m["obs"], m["cov"]) for sv, m in zip(surveys, Ms)]
class _Joint:
    n = sum(l.n for l in likes)
    def loglike_batch(self, lams): return sum(l.loglike_batch(lams) for l in likes)
like = _Joint()
print(f"{'+'.join(surveys)}: {like.n} events; device {a.device}", flush=True)
lo = {k: PRIOR_BOX[k][0] for k in LAMBDA_NAMES}; hi = {k: PRIOR_BOX[k][1] for k in LAMBDA_NAMES}
NAMES = ["A", "B", "scatter", "logM0", "log_delta", "theta"]
BND = {k: [lo[k], hi[k]] for k in NAMES if k != "log_delta"}; BND["log_delta"] = [np.log(lo["delta"]), np.log(hi["delta"])]
LOGNORM = -sum(np.log(b[1] - b[0]) for b in BND.values())


class TDEModel(Model):
    names = NAMES; bounds = BND

    def log_prior(self, x):
        ok = np.ones(np.atleast_1d(x["A"]).shape, bool)
        for k in NAMES: ok &= (x[k] >= BND[k][0]) & (x[k] <= BND[k][1])
        return np.where(ok, LOGNORM, -np.inf)

    def log_likelihood(self, x):
        lam = np.stack([np.atleast_1d(x[k]) for k in NAMES], 1).astype(np.float64); lam[:, 4] = np.exp(lam[:, 4])
        out = like.loglike_batch(lam)
        return out if np.ndim(x["A"]) else out[0]


os.makedirs(a.out, exist_ok=True)
from nessai.utils import setup_logger; setup_logger(output=a.out, log_level="INFO")   # progress (iteration, logL, acceptance) to the terminal and <out>/nessai.log
fs = FlowSampler(TDEModel(), output=a.out, nlive=a.nlive, seed=a.seed, resume=not a.no_resume, pytorch_threads=a.threads, flow_config=dict(device_tag=a.flow_device or "cpu"), **({} if a.max_iter is None else dict(max_iteration=a.max_iter)))
t = time.time(); fs.run(); wall = time.time() - t; nev = fs.ns.model.likelihood_evaluations
print(f"nessai finished in {wall/60:.2f} min; likelihood evaluations {nev}; {wall/max(nev,1):.4f} s per evaluation (all overheads included)"); json.dump(dict(wall_s=wall, likelihood_evaluations=int(nev), device=a.device, threads=a.threads, max_iter=a.max_iter, seed=a.seed), open(f"{a.out}/timing.json", "w"))
ns = fs.ns; post = fs.posterior_samples
S = np.stack([post[k] for k in NAMES], 1); S[:, 4] = np.exp(S[:, 4]); tr = M["lam_true"]
np.savez(f"{a.out}/posterior.npz", samples=S, names=np.array(["A", "B", "scatter", "logM0", "delta", "theta"]), lam_true=tr, log_evidence=float(ns.log_evidence), n_events=like.n, surveys=np.array(surveys), mocks=np.array(a.mock))
q = (S < tr[None]).mean(0)
for k, name in enumerate(["A", "B", "scatter", "logM0", "delta", "theta"]):
    print(f"{name:8s} truth {tr[k]:8.3f}  posterior median {np.median(S[:, k]):8.3f}  68% [{np.percentile(S[:, k], 16):.3f}, {np.percentile(S[:, k], 84):.3f}]  truth quantile {q[k]:.2f}")
