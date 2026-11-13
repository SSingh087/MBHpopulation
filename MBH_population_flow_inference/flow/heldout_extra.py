#!/usr/bin/env python
"""
Step-1 extra held-out tests (fresh universes; seeds 95M+, disjoint from training, the sweep, and acceptance.py's 90M+).
  (A) "stats": the 5 acceptance Lambda with 400 universes each (instead of 20) -> EMRI KS/correlations with ~20x the events.
  (B) "edges": 24 Lambda pushed to the edges of the prior box (each parameter randomly in its lowest or highest 10%, delta on a log axis,
      fixed seed) with 200 universes each -> does the flow degrade toward the box edge?
Stages:  --sim SET LAMIDX CHUNK   (100 universes per chunk, parallel over seeds)    --analyse
Data: <DATA>/simulations/heldout_extra/<set>_<lam>_<chunk>.npz ; outputs <DATA>/plots/flow_best/step1_extra/, <DATA>/logs/flow/step1_extra.json
"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, POPS, LAMBDA_NAMES, FIDUCIAL, PRIOR_BOX

NU = 100
CH = {"stats": 4, "edges": 2}
STATS = {   # = acceptance.py TEST_LAMBDAS
    "fiducial": FIDUCIAL,
    "heavy_corner": dict(A=8.10, B=3.50, scatter=0.70, logM0=8.70, delta=10.0, theta=1.8),
    "light_corner": dict(A=7.65, B=5.60, scatter=0.40, logM0=10.2, delta=1.0, theta=0.5),
    "highA_steep": dict(A=8.10, B=5.50, scatter=0.40, logM0=8.60, delta=15.0, theta=0.3),
    "lowA_shallow": dict(A=7.60, B=3.40, scatter=0.70, logM0=9.80, delta=2.0, theta=1.2)}


def edge_lams(n=24, seed=5):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        lam = {}
        for k in LAMBDA_NAMES:
            lo, hi = PRIOR_BOX[k]; lg = k == "delta"; l, h = (np.log(lo), np.log(hi)) if lg else (lo, hi)
            u = rng.uniform(0, 0.1) if rng.random() < 0.5 else rng.uniform(0.9, 1.0)
            lam[k] = float(np.exp(l + u * (h - l)) if lg else l + u * (h - l))
        out.append(lam)
    return out


def lam_list(s): return list(STATS.values()) if s == "stats" else edge_lams()
def lam_names(s): return list(STATS) if s == "stats" else [f"edge{i:02d}" for i in range(24)]


ap = argparse.ArgumentParser()
ap.add_argument("--sim", nargs=3, metavar=("SET", "LAMIDX", "CHUNK")); ap.add_argument("--analyse", action="store_true")
ap.add_argument("--tag", default="_best"); ap.add_argument("--extra_out", default=None)
args = ap.parse_args()
D = f"{DATA}/simulations/heldout_extra"; os.makedirs(D, exist_ok=True)

if args.sim:
    s, k, c = args.sim[0], int(args.sim[1]), int(args.sim[2]); lam = lam_list(s)[k]
    import source_sim as es
    E, T, ne, nt, Le, Lt = [], [], [], [], [], []
    for j in range(NU):
        seed = 95_000_000 + (0 if s == "stats" else 1_000_000) + k * 10_000 + c * 100 + j
        r = es.simulate_sources(lam, seed=seed); ev, td = r["events"], r["tde_events"]
        E.append(np.stack([ev["lgMBH"], ev["spin"], ev["z"], ev["e_plunge"]], -1).astype(np.float32))
        T.append(np.stack([td["lgMBH"], td["z"]], -1).astype(np.float32)); ne.append(r["n_events"]); nt.append(r["n_tde_total"]); Le.append(r["Lambda_tot"]); Lt.append(r["Lambda_tot_tde"])
    T = np.concatenate(T); T = T[np.random.default_rng(k * 100 + c).choice(len(T), min(len(T), 30_000), replace=False)]
    np.savez(f"{D}/{s}_{k:02d}_{c}.npz", emri=np.concatenate(E), tde=T, n_e=ne, n_t=nt, L_e=Le, L_t=Lt, lam=np.array([lam[x] for x in LAMBDA_NAMES]))
    print(f"{s} lam {k} chunk {c}: {NU} universes, {sum(ne)} EMRI, {len(T)} TDE kept"); sys.exit(0)

# ---------------------------------------------------------------- analysis
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from common import LamScaler, lam_dict_to_array
from diagnostics import load_flow
out = f"{DATA}/plots/flow_best/step1_extra"; os.makedirs(out, exist_ok=True)
ls = LamScaler(); rng = np.random.default_rng(0); dev = torch.device("cpu"); plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "legend.frameon": False})
flows = {p: load_flow(f"{DATA}/models/flow_{p}{args.tag}/best.pt", dev) for p in POPS}
M = {}
for s in ("stats", "edges"):
    names, lams = lam_names(s), lam_list(s); M[s] = {}
    for k, (n, lam) in enumerate(zip(names, lams)):
        fs = sorted(glob.glob(f"{D}/{s}_{k:02d}_*.npz"))
        if not fs: continue
        ds = [np.load(f) for f in fs]; M[s][n] = dict(lam=lam, n_univ=len(fs) * NU, Lambda_tot_emri=float(np.mean(np.concatenate([d["L_e"] for d in ds]))))
        for pop, info in POPS.items():
            xs = np.concatenate([d[pop] for d in ds]); d_ = len(info["names"])
            if len(xs) < 20: M[s][n][pop] = dict(n=int(len(xs))); continue
            m = min(len(xs) * (10 if pop == "emri" else 3), 300_000); fl, xt, _ = flows[pop]
            ctx = ls.forward(torch.tensor(lam_dict_to_array(lam))[None]).repeat(m, 1)
            with torch.no_grad(): xf = xt.inverse(torch.cat([fl.sample(ctx[i:i + 50000]) for i in range(0, m, 50000)])).numpy()
            ks = [ks_2samp(xs[:, i], xf[:, i]).statistic for i in range(d_)]; crit = 1.36 * np.sqrt((len(xs) + m) / (len(xs) * m))
            cs, cf = np.corrcoef(xs.T), np.corrcoef(xf.T); pr = [(i, j) for i in range(d_) for j in range(i + 1, d_)]
            z = [(cf[i, j] - cs[i, j]) / ((1 - cs[i, j] ** 2) / np.sqrt(len(xs))) for i, j in pr]
            q = lambda a: np.quantile(a, [0.01, 0.05, 0.5, 0.95, 0.99])
            M[s][n][pop] = dict(n=int(len(xs)), ks=[float(v) for v in ks], ks_crit95=float(crit), corr_z=[float(v) for v in z], corr_pairs=[[int(i), int(j)] for i, j in pr],
                                mean_shift_over_std=[float((xf[:, i].mean() - xs[:, i].mean()) / xs[:, i].std()) for i in range(d_)],
                                std_ratio=[float(xf[:, i].std() / xs[:, i].std()) for i in range(d_)],
                                tail_q_shift_over_std=[[float(v) for v in (q(xf[:, i]) - q(xs[:, i])) / xs[:, i].std()] for i in range(d_)])
json.dump(M, open(f"{DATA}/logs/flow/step1_extra.json", "w"), indent=1)

COL = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"]
# fig 1: EMRI KS with 20x events (stats set) vs noise level
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
for pi, pop in enumerate(POPS):
    info = POPS[pop]; d_ = len(info["names"])
    for c, (n, r) in enumerate(M["stats"].items()):
        if "ks" not in r[pop]: continue
        ax[pi].plot(range(d_), r[pop]["ks"], "o-", color=COL[c], ms=3, lw=0.8, label=f"{n} (n={r[pop]['n']})")
        ax[pi].plot(range(d_), [r[pop]["ks_crit95"]] * d_, "-", color=COL[c], lw=0.5, alpha=0.4)
    ax[pi].set_xticks(range(d_)); ax[pi].set_xticklabels(info["labels"]); ax[pi].set_ylabel("KS(sim, flow)"); ax[pi].set_title(f"{pop.upper()}: more events (thin lines: 95% noise level)", fontsize=7); ax[pi].legend(fontsize=5)
fig.tight_layout(); fig.savefig(f"{out}/stats_ks.png", dpi=170); fig.savefig(f"{out}/stats_ks.pdf"); plt.close(fig)
# fig 2: KS at box edges, EMRI & TDE; with max |corr z|
fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.8))
for pi, pop in enumerate(POPS):
    info = POPS[pop]; d_ = len(info["names"]); es_ = [(n, r) for n, r in M["edges"].items() if "ks" in r[pop]]
    for i in range(d_):
        ax[0, pi].plot(range(len(es_)), [r[pop]["ks"][i] for n, r in es_], "o", ms=3, label=info["labels"][i])
    ax[0, pi].plot(range(len(es_)), [r[pop]["ks_crit95"] for n, r in es_], "k-", lw=0.6, label="95% noise level")
    ax[0, pi].set_title(f"{pop.upper()}: KS at 24 prior-edge $\\Lambda$", fontsize=7); ax[0, pi].set_xlabel("edge point"); ax[0, pi].set_ylabel("KS"); ax[0, pi].legend(fontsize=5, ncol=2)
    ax[1, pi].plot(range(len(es_)), [np.max(np.abs(r[pop]["corr_z"])) for n, r in es_], "o", ms=3, color="#0072B2"); ax[1, pi].axhline(3, color="k", lw=0.6, ls=":")
    ax[1, pi].set_ylabel("max |$\\Delta\\rho$| / $\\sigma$"); ax[1, pi].set_xlabel("edge point"); ax[1, pi].set_yscale("log")
fig.tight_layout(); fig.savefig(f"{out}/edges_ks.png", dpi=170); fig.savefig(f"{out}/edges_ks.pdf"); plt.close(fig)
for s in M:
    for n, r in M[s].items():
        for pop in POPS:
            if "ks" in r[pop]: print(s, n, pop, "n=%d" % r[pop]["n"], "KS", np.round(r[pop]["ks"], 3), "crit %.3f" % r[pop]["ks_crit95"], "max|corr z| %.1f" % np.max(np.abs(r[pop]["corr_z"])))
print("->", out)
