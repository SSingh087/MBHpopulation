#!/usr/bin/env python
"""
Source-generation smoke test: sample Lambda from the prior box, simulate each universe's EMRI *source* catalogue
(no detectability), save (Lambda, events, Lambda_tot, N) per draw and report timing / size / red flags.

Run:   python run_smoke_test.py --n_draws 30
Output: /data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference/simulations/smoke_test/draw_XXXX.npz
        + logs/sources_smoke_summary.json
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import source_sim as es

DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"
ap = argparse.ArgumentParser()
ap.add_argument("--n_draws", type=int, default=30)
ap.add_argument("--n_pool", type=int, default=100_000)
ap.add_argument("--seed", type=int, default=12345, help="master seed; draw i uses seed+i for the simulator")
ap.add_argument("--outdir", default=f"{DATA}/simulations/smoke_test")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)
os.makedirs(f"{DATA}/logs", exist_ok=True)

master = np.random.default_rng(args.seed)
rows, t_all = [], time.time()
for i in range(args.n_draws):
    lam = es.sample_prior(master)
    sim_seed = args.seed + 1 + i
    t0 = time.time()
    try:
        r = es.simulate_sources(lam, seed=sim_seed, n_pool=args.n_pool)
    except Exception as exc:                                # record crashes instead of dying
        print(f"[{i}] CRASH at Lambda={lam}: {type(exc).__name__}: {exc}", flush=True)
        rows.append(dict(i=i, lam=lam, crashed=repr(exc)))
        continue
    dt = time.time() - t0
    ev = r["events"]
    np.savez_compressed(f"{args.outdir}/draw_{i:04d}.npz",
                        lam=np.array([lam[k] for k in es.LAMBDA_NAMES]), lam_names=np.array(es.LAMBDA_NAMES),
                        Lambda_tot=r["Lambda_tot"], n_events=r["n_events"], n_eff=r["n_eff"],
                        seed=sim_seed, band=np.array(r["band"] if r["band"] else [np.nan, np.nan]),
                        Lambda_tot_tde=r["Lambda_tot_tde"], n_tde_total=r["n_tde_total"], n_eff_tde=r["n_eff_tde"],
                        **{k: ev[k] for k in es.EVENT_KEYS},
                        **{"tde_" + k: r["tde_events"][k].astype(np.float32) for k in es.TDE_KEYS})
    rows.append(dict(i=i, lam=lam, Lambda_tot=r["Lambda_tot"], n_events=r["n_events"], n_eff=r["n_eff"],
                     Lambda_tot_tde=r["Lambda_tot_tde"], n_tde_total=r["n_tde_total"], n_eff_tde=r["n_eff_tde"],
                     active_frac=r.get("active_frac"), band=r["band"], time_s=dt, failed=r.get("failed")))
    print(f"[{i:3d}] EMRI N={r['n_events']:4d} L={r['Lambda_tot']:7.2f} neff={r['n_eff']:6.1f} | TDE N={r['n_tde_total']:7d} L={r['Lambda_tot_tde']:9.0f} neff={r['n_eff_tde']:6.0f} t={dt:5.1f}s  "
          f"A={lam['A']:.2f} B={lam['B']:.2f} sc={lam['scatter']:.2f} logM0={lam['logM0']:.2f} "
          f"d={lam['delta']:.2f} th={lam['theta']:.2f}", flush=True)

total = time.time() - t_all
ok = [r for r in rows if "Lambda_tot" in r]
size = sum(os.path.getsize(os.path.join(args.outdir, f)) for f in os.listdir(args.outdir))
NT = np.array([r["n_tde_total"] for r in ok]); neT = np.array([r["n_eff_tde"] for r in ok])
N = np.array([r["n_events"] for r in ok]); L = np.array([r["Lambda_tot"] for r in ok]); ne = np.array([r["n_eff"] for r in ok])
summ = dict(n_draws=args.n_draws, n_ok=len(ok), n_crashed=len(rows) - len(ok), total_wall_s=total,
            time_per_draw_s=total / max(args.n_draws, 1), disk_bytes=size,
            N_min=int(N.min()), N_median=float(np.median(N)), N_max=int(N.max()),
            n_zero_events=int((N == 0).sum()), n_events_lt5=int((N < 5).sum()),
            N_tde_min=int(NT.min()), N_tde_median=float(np.median(NT)), N_tde_max=int(NT.max()), n_eff_tde_min=float(neT.min()),
            n_eff_min=float(ne.min()), n_eff_median=float(np.median(ne)), rows=rows)
with open(f"{DATA}/logs/sources_smoke_summary.json", "w") as f:
    json.dump(summ, f, indent=1, default=float)
print("\n=== SUMMARY ===")
for k in ["n_draws", "n_ok", "n_crashed", "total_wall_s", "time_per_draw_s", "disk_bytes", "N_min", "N_median", "N_max",
          "n_zero_events", "n_events_lt5", "n_eff_min", "n_eff_median", "N_tde_min", "N_tde_median", "N_tde_max", "n_eff_tde_min"]:
    print(f"{k:16s} {summ[k]}")
print("flags: draws with n_eff < 30 are noisy (importance weights dominated by few galaxies);")
print("       draws with N == 0 give an empty catalogue (flow context needs a rule for them).")
