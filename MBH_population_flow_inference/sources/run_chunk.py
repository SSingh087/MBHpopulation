#!/usr/bin/env python
"""
Source-generation worker: simulate one chunk of universes (EMRI + TDE sources, clean mode).

Universe g (global index) is fully determined by (--master_seed, g):
    Lambda_g ~ prior box   (rng = default_rng([master_seed, g]))
    simulator seed         = master_seed * 1_000_003 + g
so any universe can be regenerated alone, and re-running a chunk skips universes that already exist (resumable).

Output: <outdir>/chunk_<c>/univ_<g>.npz   (same fields as the smoke test)
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import source_sim as es

DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"
ap = argparse.ArgumentParser()
ap.add_argument("--chunk", type=int, required=True)
ap.add_argument("--per_chunk", type=int, required=True)
ap.add_argument("--master_seed", type=int, default=2026)
ap.add_argument("--n_pool", type=int, default=100_000)
ap.add_argument("--nucleation", default="hannah", choices=["hannah", "neumayer"])
ap.add_argument("--occupation", default="varying", choices=["varying", "uniform"])
ap.add_argument("--alpha_Re", type=float, default=None, help="Re size-evolution exponent (default 0 = no evolution, as the original v1 runs)")
ap.add_argument("--legacy", type=int, default=0, help="1: old-pipeline sigma treatment (independent 2nd sigma draw in the M-sigma step; as the catalogues)")
ap.add_argument("--outdir", default=f"{DATA}/simulations/full_run")
args = ap.parse_args()
SC = es.set_scenario(args.nucleation, args.occupation, args.alpha_Re, legacy=bool(args.legacy))
print("scenario", SC, flush=True)

cdir = os.path.join(args.outdir, f"chunk_{args.chunk:04d}")
os.makedirs(cdir, exist_ok=True)
t_all, n_done, n_skip, n_fail = time.time(), 0, 0, 0
for j in range(args.per_chunk):
    g = args.chunk * args.per_chunk + j
    path = os.path.join(cdir, f"univ_{g:06d}.npz")
    if os.path.exists(path):
        n_skip += 1; continue
    lam = es.sample_prior(np.random.default_rng([args.master_seed, g]))
    seed = args.master_seed * 1_000_003 + g
    try:
        r = es.simulate_sources(lam, seed=seed, n_pool=args.n_pool)
    except Exception as exc:
        print(f"[g={g}] FAILED {type(exc).__name__}: {exc}  Lambda={lam}", flush=True); n_fail += 1; continue
    ev, tde = r["events"], r["tde_events"]
    tmp = path + ".tmp.npz"                                  # write-then-rename: no half-written files on eviction
    np.savez_compressed(tmp, lam=np.array([lam[k] for k in es.LAMBDA_NAMES]), lam_names=np.array(es.LAMBDA_NAMES),
                        universe=g, seed=seed, scenario=str(SC), master_seed=args.master_seed,
                        Lambda_tot=r["Lambda_tot"], n_events=r["n_events"], n_eff=r["n_eff"],
                        Lambda_tot_tde=r["Lambda_tot_tde"], n_tde_total=r["n_tde_total"], n_eff_tde=r["n_eff_tde"],
                        band=np.array(r["band"] if r["band"] else [np.nan, np.nan]),
                        **{k: ev[k] for k in es.EVENT_KEYS},
                        **{"tde_" + k: tde[k].astype(np.float32) for k in es.TDE_KEYS})
    os.replace(tmp, path)
    n_done += 1
print(f"chunk {args.chunk}: done {n_done}, skipped(existing) {n_skip}, failed {n_fail}, wall {time.time()-t_all:.0f}s", flush=True)
sys.exit(1 if n_fail else 0)          # non-zero -> DAGMan RETRY re-runs only the missing universes
