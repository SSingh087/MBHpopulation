#!/usr/bin/env python
"""make_tde_mock.py -- one mock TDE catalogue for the population inference.

  universe at Lambda_true (source_sim, seed)  ->  nuisance draws  ->  classifier probability p  ->  detected ~ Bernoulli(p)  (consistent with the soft selection P=p used in the likelihood)
  -> error flow at the truth gives C1 -> noisy estimate theta_hat = theta + N(0, C1) (z, lgM, log eta, log tvisc, bb, log rph0, lphoto)
  -> event posterior N(theta_hat, C1) (same C: Fisher at theta_hat ~ Fisher at theta); the (lgM, z) block is what the likelihood uses.
Lambda_true = FIDUCIAL, or a prior draw with --prior_seed.  --n_events keeps a random subset of the detected events (the likelihood is shape-only, so this just sets the sample size).
  python make_tde_mock.py --survey ZTF --seed 1 --n_events 300 --out mock.npz
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tde_pop_lib import *
import source_sim as es
from tde_classifier import TDEClassifier

ap = argparse.ArgumentParser(); ap.add_argument("--survey", required=True, choices=["ZTF", "LSST"]); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--prior_seed", type=int, default=None); ap.add_argument("--n_events", type=int, default=0, help=">0: keep exactly this many detected events; 0: draw a Poisson number around --mu"); ap.add_argument("--mu", type=float, default=0.0, help="expected number of detections (Lambda_tot x alpha at Lambda_true)"); ap.add_argument("--source", default="flow", choices=["flow", "sim"], help="truth events from the flow (self-consistent test of the inference) or one simulator universe"); ap.add_argument("--n_flow", type=int, default=400000); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cpu")
a = ap.parse_args()
from priors import sample_prior
lam = FIDUCIAL if a.prior_seed is None else sample_prior(np.random.default_rng(a.prior_seed))
rng = np.random.default_rng([777, a.seed]); zmax = ZMAX[a.survey]; t0 = time.time()
ef = ErrorFlow(a.survey, a.device); clf = TDEClassifier(a.survey)

if a.source == "sim":
    r = es.simulate_sources(lam, seed=97_500_000 + a.seed, max_tde_store=10 ** 8); td = r["tde_events"]
    ok = (td["z"] > 1e-3) & (td["z"] < zmax) & (td["lgMBH"] > 4) & (td["lgMBH"] < 7); lgM, z = np.asarray(td["lgMBH"], float)[ok], np.asarray(td["z"], float)[ok]
else:   # draw (lgM, z) from the flow itself (conditioned on Lambda_true), keep the in-range ones
    pop = TDEPopulation(a.survey, a.device); torch.manual_seed(a.seed); ls = pop.scaler.forward(torch.tensor(lam_array(lam)[None])).to(pop.dev)
    x = pop.xt.inverse(pop.flow.sample(ls.expand(a.n_flow, -1))).cpu().numpy().astype(float)
    ok = (x[:, 1] > 1e-3) & (x[:, 1] < zmax) & (x[:, 0] > 4) & (x[:, 0] < 7); lgM, z = x[ok, 0], x[ok, 1]
n = len(z); lg = lambda lo, hi: np.exp(rng.uniform(np.log(lo), np.log(hi), n))
d = dict(lgM=lgM, z=z, ra=rng.uniform(0, 2 * np.pi, n), dec=np.arcsin(rng.uniform(-1, 1, n)), t0=rng.uniform(ef.t0min, ef.t0min + ef.trange, n), eta=lg(1e-3, 1e-1), alpha=lg(1e-2, 1e-1),
         beta=rng.uniform(1.001, 30, n), tvisc=lg(1., 30.), bb=rng.uniform(0, 2, n), rph0=lg(1e-4, 1e4), lphoto=rng.uniform(0, 2, n))
p = clf.prob(dict(lgMBH=d["lgM"], z_gal=d["z"], ra_rad=d["ra"], dec_rad=d["dec"], eta=d["eta"], alpha=d["alpha"], beta=d["beta"], t0_mjd_transient=d["t0"], tvisc=d["tvisc"], bb=d["bb"], rph0=d["rph0"], lphoto=d["lphoto"]))
det = np.where(rng.random(n) < p)[0]; n_det_all = len(det)
n_keep = a.n_events if a.n_events > 0 else int(rng.poisson(a.mu))
if n_keep > len(det): raise SystemExit(f"need {n_keep} detected events but the flow sample has only {len(det)}: raise --n_flow")
det = rng.choice(det, n_keep, replace=False)
d = {k: v[det] for k, v in d.items()}; m = len(det)
print(f"{a.survey}: Lambda_true {lam}; {a.source} sample has {n} in-range TDEs, {n_det_all} detected (sum p = {p.sum():.1f}); keeping {m}", flush=True)

C1 = ef.sample_cov(d)
mu = np.stack([d["z"], d["lgM"] - 6, np.log10(d["eta"]), np.log10(d["tvisc"]), d["bb"], np.log10(d["rph0"]), d["lphoto"]], 1)
hat = mu + np.einsum("nij,nj->ni", np.linalg.cholesky(C1 + 1e-15 * np.eye(NP)), rng.standard_normal((m, NP)))
# the Fisher covariance is a deterministic function of the event, but the error flow SAMPLES one covariance from its conditional spread (0.5-1.4 dex in sigma);
# drawing a second, independent sample at theta_hat would decouple the quoted error from the realised one.  So the same C is used for the noise and for the quoted posterior (Fisher at theta_hat ~ Fisher at theta).
obs = np.stack([hat[:, 1] + 6, hat[:, 0]], 1); cov = C1[:, [1, 0]][:, :, [1, 0]]
np.savez(a.out, obs=obs, cov=cov, truth=np.stack([d["lgM"], d["z"]], 1), lam_true=lam_array(lam), survey=a.survey, n_universe=n, n_detected_universe=n_det_all, mu=a.mu, p=p[det])
print(f"saved {a.out}: {m} events, median sigma_lgM {np.median(np.sqrt(cov[:,0,0])):.3f} dex, sigma_z/z {np.median(np.sqrt(cov[:,1,1])/np.maximum(obs[:,1],1e-3)):.3f}; {time.time()-t0:.0f}s")
