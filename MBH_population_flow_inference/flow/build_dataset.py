#!/usr/bin/env python
"""
Consolidate the 10k per-universe .npz files into 3 arrays files for fast training:
  <DATA>/simulations/consolidated/meta.npz  : per-universe Lambda, expected/realised counts, n_eff, train/val flag
  .../emri.npz  x (N,4) [lgMBH, spin, z, e_plunge], uid (N,)   (all EMRI sources)
  .../tde.npz   x (N,2) [lgMBH, z],                 uid (N,)   (random subsample of --tde_per_univ per universe)
Validation universes: universe id % 10 == 0 (10%).
"""
import argparse, glob, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FULL_RUN, CONSOL, LAMBDA_NAMES

ap = argparse.ArgumentParser()
ap.add_argument("--tde_per_univ", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)
files = sorted(glob.glob(f"{FULL_RUN}/chunk_*/univ_*.npz"))
print(len(files), "universe files"); t0 = time.time()
lam, gid, Lt, Ltt, nE, nT, ne, net = [], [], [], [], [], [], [], []
xe, ue, xt, ut = [], [], [], []
for i, f in enumerate(files):
    d = np.load(f)
    assert list(d["lam_names"]) == LAMBDA_NAMES
    lam.append(d["lam"]); gid.append(int(d["universe"])); Lt.append(float(d["Lambda_tot"])); Ltt.append(float(d["Lambda_tot_tde"]))
    nE.append(int(d["n_events"])); nT.append(int(d["n_tde_total"])); ne.append(float(d["n_eff"])); net.append(float(d["n_eff_tde"]))
    if nE[-1] > 0:
        xe.append(np.stack([d["lgMBH"], d["spin"], d["z"], d["e_plunge"]], -1).astype(np.float32)); ue.append(np.full(nE[-1], i, np.int32))
    n = len(d["tde_z"])
    if n > 0:
        sel = rng.choice(n, size=min(n, args.tde_per_univ), replace=False)
        xt.append(np.stack([d["tde_lgMBH"][sel], d["tde_z"][sel]], -1).astype(np.float32)); ut.append(np.full(len(sel), i, np.int32))
    if (i + 1) % 1000 == 0: print(f"  {i+1}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
gid = np.array(gid)
os.makedirs(CONSOL, exist_ok=True)
np.savez(f"{CONSOL}/meta.npz", lam=np.array(lam, np.float32), universe=gid, Lambda_tot=np.array(Lt), Lambda_tot_tde=np.array(Ltt),
         n_emri=np.array(nE), n_tde_total=np.array(nT), n_eff=np.array(ne), n_eff_tde=np.array(net), is_val=(gid % 10 == 0))
np.savez(f"{CONSOL}/emri.npz", x=np.concatenate(xe), uid=np.concatenate(ue))
np.savez(f"{CONSOL}/tde.npz", x=np.concatenate(xt), uid=np.concatenate(ut))
print(f"EMRI events {sum(len(a) for a in xe)}, TDE events {sum(len(a) for a in xt)}, universes {len(files)}, "
      f"val universes {(gid % 10 == 0).sum()}, {time.time()-t0:.0f}s")
