#!/usr/bin/env python
"""tde_fin_check.py -- fraction of TDEs at z<0.5 (the ZTF classifier range) in fresh simulated universes vs the flow integral, at highA_steep (the outlier of the TDE end-to-end test) and fiducial (control).
  python tde_fin_check.py --lam 3 --chunk 0 [--nuniv 10]     (one DAG node; seeds 97M+, new)      python tde_fin_check.py --report"""
import argparse, glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "flow")); sys.path.insert(0, os.path.join(HERE, "..", "src"))
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"; OUT = f"{DATA}/simulations/tde_fin_check"
from priors import FIDUCIAL
LAMS = {0: FIDUCIAL, 3: dict(A=8.10, B=5.50, scatter=0.40, logM0=8.60, delta=15.0, theta=0.3)}; NAMES = {0: "fiducial", 3: "highA_steep"}
ap = argparse.ArgumentParser(); ap.add_argument("--lam", type=int, default=3); ap.add_argument("--chunk", type=int, default=0); ap.add_argument("--nuniv", type=int, default=10); ap.add_argument("--report", action="store_true")
a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
if not a.report:
    import source_sim as es
    nin, nall = [], []
    for j in range(a.nuniv):
        r = es.simulate_sources(LAMS[a.lam], seed=97_000_000 + a.lam * 100_000 + a.chunk * 100 + j, max_tde_store=10 ** 8); z = r["tde_events"]["z"]
        nin.append(int(((z > 1e-3) & (z < 0.5)).sum())); nall.append(int(len(z)))
    np.savez(f"{OUT}/{a.lam}_{a.chunk:03d}.npz", n_in=nin, n_all=nall); print(f"lam {a.lam} chunk {a.chunk}: f_in per universe {np.round(np.array(nin)/np.array(nall), 4)}"); sys.exit(0)
import torch
from common import LamScaler, lam_dict_to_array
from diagnostics import load_flow
flow, xt, _ = load_flow(f"{DATA}/models/flow_tde_best/best.pt", torch.device("cpu")); LO, HI, STD, EPS = xt.lo, xt.hi, xt.std, xt.EPS; ls_ = LamScaler(); res = {}
def logp(x, ls):
    u = xt.forward(x); t = ((x - LO) / (HI - LO)).clamp(EPS, 1 - EPS)
    return flow.log_prob(u, ls.expand(len(x), -1)).double() + (-torch.log(STD) - torch.log(t * (1 - t)) - torch.log(HI - LO)).sum(1).double()
nlf, nzf = 240, 320; lgf = np.linspace(4, 7, nlf + 1); zlf = np.linspace(np.log(1e-3), np.log(0.5), nzf + 1)
Gl, Gz = np.meshgrid(0.5 * (lgf[1:] + lgf[:-1]), np.exp(0.5 * (zlf[1:] + zlf[:-1])), indexing="ij"); Wf = np.diff(lgf)[:, None] * (Gz * np.diff(zlf)[None, :])
pts = torch.tensor(np.stack([Gl.reshape(-1), Gz.reshape(-1)], 1), dtype=torch.float32)
for k, lam in LAMS.items():
    fs = sorted(glob.glob(f"{OUT}/{k}_*.npz"))
    if not fs: continue
    ni = np.concatenate([np.load(f)["n_in"] for f in fs]); na = np.concatenate([np.load(f)["n_all"] for f in fs]); f = ni / na
    with torch.no_grad(): m = float((np.exp(logp(pts, ls_.forward(torch.tensor(lam_dict_to_array(lam))[None])).numpy()).reshape(nlf, nzf) * Wf).sum())
    pooled = ni.sum() / na.sum(); err = f.std(ddof=1) / np.sqrt(len(f))
    res[NAMES[k]] = dict(n_univ=len(f), f_in_mean_per_universe=float(f.mean()), f_in_pooled=float(pooled), err=float(err), flow_mass=m, sim_over_flow=float(f.mean() / m), sim_over_flow_err=float(err / m))
    print(f"{NAMES[k]:12s} {len(f)} universes: simulator f_in {f.mean():.4f} +- {err:.4f} | flow {m:.4f} | sim/flow {f.mean()/m:.3f} +- {err/m:.3f}")
json.dump(res, open(f"{DATA}/logs/tde_fin_check.json", "w"), indent=1)
