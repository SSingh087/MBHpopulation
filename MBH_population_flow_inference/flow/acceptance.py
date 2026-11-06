#!/usr/bin/env python
"""
Stage-1 acceptance test: does the trained flow reproduce the simulator at Lambda values it was never trained on?

For each test Lambda (the fiducial + 4 points near the edges of the prior box):
  (a) simulate fresh held-out universes at EXACTLY that Lambda (seeds disjoint from training; cached),
  (b) pool their sources, draw the same number of samples from the trained flow conditioned on that Lambda,
  (c) overlay corner / marginal plots and compute per-variable KS statistics.
Outputs: <DATA>/plots/acceptance/*.png , <DATA>/logs/flow/acceptance_<tag>.json
  python acceptance.py --n_univ 20 --workers 5
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from scipy.stats import ks_2samp, wasserstein_distance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, POPS, FIDUCIAL, LAMBDA_NAMES, LAM_LABELS, XTransform, LamScaler, lam_dict_to_array
from flows import ConditionalFlow
import plotting

TEST_LAMBDAS = {
    "fiducial":      FIDUCIAL,
    "heavy_corner":  dict(A=8.10, B=3.50, scatter=0.70, logM0=8.70, delta=10.0, theta=1.8),   # many, massive, high-scatter
    "light_corner":  dict(A=7.65, B=5.60, scatter=0.40, logM0=10.2, delta=1.0, theta=0.5),     # few, low-mass
    "highA_steep":   dict(A=8.10, B=5.50, scatter=0.40, logM0=8.60, delta=15.0, theta=0.3),
    "lowA_shallow":  dict(A=7.60, B=3.40, scatter=0.70, logM0=9.80, delta=2.0, theta=1.2),
}

ap = argparse.ArgumentParser()
ap.add_argument("--n_univ", type=int, default=20, help="held-out universes simulated per test Lambda")
ap.add_argument("--workers", type=int, default=5)
ap.add_argument("--seed_base", type=int, default=90_000_000)
ap.add_argument("--max_plot", type=int, default=50_000)
ap.add_argument("--tag", default="", help="model tag, e.g. '_quick'")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()
out = f"{DATA}/plots/acceptance{args.tag}"; os.makedirs(out, exist_ok=True)
HELD = f"{DATA}/simulations/heldout"; os.makedirs(HELD, exist_ok=True)


def _sim_one(task):
    k, j, lam = task
    import source_sim as es
    r = es.simulate_sources(lam, seed=args.seed_base + k * 1000 + j)
    ev, td = r["events"], r["tde_events"]
    return k, dict(emri=np.stack([ev["lgMBH"], ev["spin"], ev["z"], ev["e_plunge"]], -1).astype(np.float32),
                   tde=np.stack([td["lgMBH"], td["z"]], -1).astype(np.float32), n_e=r["n_events"], n_t=r["n_tde_total"],
                   L_e=r["Lambda_tot"], L_t=r["Lambda_tot_tde"])


# ---------------- (a) held-out simulations (cached per Lambda & n_univ)
sims = {}
todo = []
for k, (name, lam) in enumerate(TEST_LAMBDAS.items()):
    f = f"{HELD}/{name}_n{args.n_univ}.npz"
    if os.path.exists(f):
        d = np.load(f); sims[name] = {key: d[key] for key in d.files}
    else:
        todo += [(k, j, lam) for j in range(args.n_univ)]
if todo:
    import multiprocessing as mp
    t0 = time.time(); print(f"simulating {len(todo)} held-out universes ({args.workers} workers)...", flush=True)
    with mp.get_context("fork").Pool(args.workers) as pool:
        res = pool.map(_sim_one, todo, chunksize=1)
    names = list(TEST_LAMBDAS)
    for k in sorted({r[0] for r in res}):
        rs = [r[1] for r in res if r[0] == k]
        rng = np.random.default_rng(k)
        tde = np.concatenate([r["tde"] for r in rs]); tde = tde[rng.choice(len(tde), min(len(tde), 200_000), replace=False)]
        d = dict(emri=np.concatenate([r["emri"] for r in rs]), tde=tde, n_e=np.array([r["n_e"] for r in rs]), n_t=np.array([r["n_t"] for r in rs]),
                 L_e=np.array([r["L_e"] for r in rs]), L_t=np.array([r["L_t"] for r in rs]))
        np.savez(f"{HELD}/{names[k]}_n{args.n_univ}.npz", **d); sims[names[k]] = d
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

# ---------------- (b) flows
dev = torch.device(args.device); ls = LamScaler(); flows = {}
for pop in POPS:
    mp_ = f"{DATA}/models/flow_{pop}{args.tag}/best.pt"
    if not os.path.exists(mp_): print("no model for", pop, "->", mp_); continue
    ck = torch.load(mp_, map_location=dev); c = ck["cfg"]
    fl = ConditionalFlow(c["d"], 6, n_layers=c["n_layers"], hidden=c["hidden"], K=c["K"], tail=c["tail"]).to(dev); fl.load_state_dict(ck["model"]); fl.eval()
    xt = XTransform(POPS[pop]["bounds"], ck["x_mean"], ck["x_std"]); flows[pop] = (fl, xt, ck["epoch"], ck["val"])

def flow_draw(pop, lam, n):
    fl, xt, _, _ = flows[pop]
    ctx = ls.forward(torch.tensor(lam_dict_to_array(lam))[None]).to(dev).repeat(n, 1)
    with torch.no_grad():
        u = torch.cat([fl.sample(ctx[i:i + 50000]) for i in range(0, n, 50000)])
    return xt.inverse(u).cpu().numpy()

# ---------------- (c) plots + metrics
metrics = {}
for pop, (fl, xt, ep, vl) in flows.items():
    info = POPS[pop]; d = len(info["names"]); names = list(TEST_LAMBDAS)
    metrics[pop] = {}
    fig_m, ax_m = plt.subplots(len(names), d, figsize=(3.4 * d, 2.5 * len(names)), squeeze=False)
    ks_tab = np.zeros((len(names), d)); crit = np.zeros(len(names))
    for r, name in enumerate(names):
        lam = TEST_LAMBDAS[name]; xs = sims[name][pop]
        if len(xs) < 20: print(f"{pop}/{name}: only {len(xs)} simulated events, skipping"); continue
        xs = xs[np.random.default_rng(0).permutation(len(xs))[: args.max_plot]]
        xf = flow_draw(pop, lam, len(xs))
        plotting.corner_overlay(xs, xf, info["labels"], f"{out}/{pop}_corner_{name}.png",
                                title=f"{pop.upper()} @ {name}: " + ", ".join(f"{k}={lam[k]:.2f}" for k in LAMBDA_NAMES))
        crit[r] = 1.36 * np.sqrt(2.0 / len(xs))
        m = {}
        for i in range(d):
            ks = ks_2samp(xs[:, i], xf[:, i]).statistic; ks_tab[r, i] = ks
            m[info["names"][i]] = dict(ks=float(ks), w1_over_std=float(wasserstein_distance(xs[:, i], xf[:, i]) / xs[:, i].std()),
                                       mean_sim=float(xs[:, i].mean()), mean_flow=float(xf[:, i].mean()), std_sim=float(xs[:, i].std()), std_flow=float(xf[:, i].std()))
            lo, hi = min(xs[:, i].min(), xf[:, i].min()), max(xs[:, i].max(), xf[:, i].max())
            ax_m[r, i].hist(xs[:, i], bins=40, range=(lo, hi), density=True, histtype="step", lw=2, color=plotting.C_DATA, label="simulated")
            ax_m[r, i].hist(xf[:, i], bins=40, range=(lo, hi), density=True, histtype="step", lw=2, color=plotting.C_FLOW, label="flow")
            ax_m[r, i].set_yticks([]); ax_m[r, i].set_title(f"KS={ks:.3f}", fontsize=8)
            if r == len(names) - 1: ax_m[r, i].set_xlabel(info["labels"][i])
        ax_m[r, 0].set_ylabel(name, fontsize=9)
        metrics[pop][name] = dict(n_sim_events=int(len(xs)), ks_crit_95=float(crit[r]), per_variable=m,
                                  Lambda=lam, expected_count_mean=float(np.mean(sims[name]["L_e" if pop == "emri" else "L_t"])))
    ax_m[0, 0].legend(fontsize=7)
    fig_m.suptitle(f"{pop.upper()}: held-out simulation vs flow at 5 Lambda values (flow epoch {ep}, val NLL {vl:.3f})")
    fig_m.tight_layout(); fig_m.savefig(f"{out}/{pop}_marginals_all_lambdas.png", dpi=120); plt.close(fig_m)
    fig, ax = plt.subplots(figsize=(1.6 * d + 3, 0.7 * len(names) + 1.5))
    im = ax.imshow(ks_tab, cmap="viridis_r", vmin=0, vmax=max(0.15, ks_tab.max()))
    ax.set_xticks(range(d)); ax.set_xticklabels(info["names"]); ax.set_yticks(range(len(names))); ax.set_yticklabels([f"{n}  (n={metrics[pop][n]['n_sim_events']})" for n in names])
    for r in range(len(names)):
        for i in range(d): ax.text(i, r, f"{ks_tab[r,i]:.3f}" + ("*" if ks_tab[r, i] > crit[r] else ""), ha="center", va="center", color="w", fontsize=9)
    plt.colorbar(im, label="KS statistic"); ax.set_title(f"{pop.upper()}: KS(sim, flow); * = above 95% two-sample noise level for that n")
    fig.tight_layout(); fig.savefig(f"{out}/{pop}_ks_summary.png", dpi=120); plt.close(fig)
    print(f"\n{pop.upper()} KS(sim,flow)  [* = above 95% noise level]")
    for r, name in enumerate(names):
        print(f"  {name:13s} n={metrics[pop][name]['n_sim_events']:6d} crit={crit[r]:.3f}  " + "  ".join(f"{info['names'][i]}={ks_tab[r,i]:.3f}{'*' if ks_tab[r,i]>crit[r] else ' '}" for i in range(d)))
os.makedirs(f"{DATA}/logs/flow", exist_ok=True)
json.dump(metrics, open(f"{DATA}/logs/flow/acceptance{args.tag}.json", "w"), indent=1)
print("plots ->", out)
