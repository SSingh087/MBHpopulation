#!/usr/bin/env python
"""
Detected-count prediction per scenario at the fiducial Lambda (the analogue of the paper's Fig. 5 box plot), WITHOUT the flows:
simulate N universes with the scenario simulator (source_sim), then apply the shared detectability (scenario independent):
  EMRI : SNR model (EMRISF.pdet_points, nuisance Y0~U(0.1,0.7), T~U(1,6) yr, rho_net, threshold 20) -> Bernoulli per event
  TDE  : classifier detection map P(lgM, z) on the fixed grid (simulations/sf/tde_grid_<survey>.npz, 'Psoft'), z > zmax -> 0
Output: <DATA>/scenarios/counts_check/<name>.npz  (N_emri, N_ZTF, N_LSST per realisation, Lambda_tot, ...)
usage: python scenario_counts.py --name sigmaNV_a075 --spec neumayer,varying,0.75 --n_real 100 --workers 16
"""
import argparse, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, f"{ROOT}/src")
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--spec", required=True)
ap.add_argument("--n_real", type=int, default=100); ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--legacy", type=int, default=0, help="1: old-pipeline quirk (independent 2nd sigma draw in the M-sigma step)")
ap.add_argument("--K", type=int, default=16); ap.add_argument("--seed0", type=int, default=77_000_000)
a = ap.parse_args()
os.environ["EMRI_SCENARIO_SPEC"] = a.spec
import source_sim as es                                    # picks the scenario from the env var
ZMAX = {"ZTF": 0.5, "LSST": 4.0}
# The old pipeline draws t0 ~ U(58200, 64423) MJD for every TDE and keeps only those inside the survey window; the classifier map P(lgM,z)
# is the detection probability GIVEN the event is in the window, so the expected detections carry the window fraction.
WIN = {"ZTF": (61396 - 58200) / (64423 - 58200), "LSST": (64423 - 60827) / (64423 - 58200)}


def one(i):
    r = es.simulate_sources(es.FIDUCIAL, seed=a.seed0 + i, legacy=bool(a.legacy))
    ev, td = r["events"], r["tde_events"]
    return dict(Lambda_tot=r["Lambda_tot"], Lambda_tot_tde=r["Lambda_tot_tde"], n_emri=r["n_events"], n_tde_total=r["n_tde_total"],
                emri=np.stack([ev["lgMBH"], ev["spin"], ev["z"], ev["e_plunge"]], 1).astype(np.float32) if r["n_events"] else np.zeros((0, 4), np.float32),
                tde=np.stack([td["lgMBH"], td["z"]], 1).astype(np.float32) if len(td["z"]) else np.zeros((0, 2), np.float32))


if __name__ == "__main__":
    import multiprocessing as mp
    t0 = time.time(); print("scenario", es.SCENARIO, flush=True)
    with mp.get_context("fork").Pool(a.workers) as pool:
        sims = pool.map(one, range(a.n_real), chunksize=1)
    print(f"simulated {len(sims)} universes in {time.time()-t0:.0f}s", flush=True)
    rng = np.random.default_rng(1)
    from scipy.interpolate import RegularGridInterpolator
    N = {"ZTF": [], "LSST": []}
    for sv in ("ZTF", "LSST"):
        g = np.load(f"{DATA}/simulations/sf/tde_grid_{sv}.npz")
        f = RegularGridInterpolator((g["lgc"], np.log(g["zc"])), g["Psoft"], bounds_error=False, fill_value=None)
        for s in sims:
            x = s["tde"]
            if len(x) == 0: N[sv].append(0); continue
            P = np.where(x[:, 1] < ZMAX[sv], np.clip(f(np.stack([x[:, 0], np.log(np.maximum(x[:, 1], 1e-3))], 1)), 0, 1), 0.0)
            N[sv].append(int(rng.binomial(s["n_tde_total"], P.mean() * WIN[sv])))      # n_total sources, each detected with mean P (stored events are an iid subsample)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sys.path.insert(0, f"{HERE}")
    from sf_lib import EMRISF
    sf = EMRISF(device=dev); gen = torch.Generator(device=dev).manual_seed(0)
    Ne = []
    for s in sims:
        if s["n_emri"] == 0: Ne.append(0); continue
        P = sf.pdet_points(torch.tensor(s["emri"], device=dev), a.K, gen).cpu().numpy()
        Ne.append(int(rng.binomial(1, np.clip(P, 0, 1)).sum()))
    out = f"{DATA}/scenarios/counts_check/{a.name}.npz"
    np.savez(out, N_emri=np.array(Ne), N_ZTF=np.array(N["ZTF"]), N_LSST=np.array(N["LSST"]),
             Lambda_tot=np.array([s["Lambda_tot"] for s in sims]), Lambda_tot_tde=np.array([s["Lambda_tot_tde"] for s in sims]),
             n_emri_src=np.array([s["n_emri"] for s in sims]), spec=a.spec)
    print(f"{a.name}: EMRI det mean {np.mean(Ne):.2f} (src {np.mean([s['n_emri'] for s in sims]):.1f}), ZTF {np.mean(N['ZTF']):.1f}, LSST {np.mean(N['LSST']):.0f}  TDE total {np.mean([s['n_tde_total'] for s in sims]):.0f}  wall {time.time()-t0:.0f}s", flush=True)
