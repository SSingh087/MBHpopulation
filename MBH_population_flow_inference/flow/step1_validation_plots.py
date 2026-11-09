#!/usr/bin/env python
"""
Step-1 validation figures (source populations p(x|Lambda)): fresh held-out universes (simulated at 5 test Lambda with seeds disjoint
from training AND from the sweep selection; cached in simulations/heldout) against flow draws at the same Lambda.

Per population one figure: (row 1) marginals, linear; (row 2) marginals, log density (tails); (row 3) quantile-quantile (flow vs sim) with
the sampling band expected from n simulated events.  Plus one figure of pairwise-correlation residuals (flow - sim) against the noise band.
  python step1_validation_plots.py [--tag _best] [--nflow_mult 20]
Outputs: <DATA>/plots/flow_best/step1/*.png|pdf  and  <DATA>/logs/flow/step1_validation.json
"""
import argparse, json, os, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, POPS, LamScaler, lam_dict_to_array, FIDUCIAL
from diagnostics import load_flow

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="_best"); ap.add_argument("--nflow_mult", type=int, default=20); ap.add_argument("--n_univ", type=int, default=20)
ap.add_argument("--device", default="cpu"); ap.add_argument("--extra_out", default=None, help="also copy the PDFs here")
args = ap.parse_args()

TEST = {  # must match acceptance.py
    "fiducial": FIDUCIAL,
    "heavy_corner": dict(A=8.10, B=3.50, scatter=0.70, logM0=8.70, delta=10.0, theta=1.8),
    "light_corner": dict(A=7.65, B=5.60, scatter=0.40, logM0=10.2, delta=1.0, theta=0.5),
    "highA_steep": dict(A=8.10, B=5.50, scatter=0.40, logM0=8.60, delta=15.0, theta=0.3),
    "lowA_shallow": dict(A=7.60, B=3.40, scatter=0.70, logM0=9.80, delta=2.0, theta=1.2)}
COL = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"]
out = f"{DATA}/plots/flow_best/step1"; os.makedirs(out, exist_ok=True)
dev = torch.device(args.device); ls = LamScaler(); rng = np.random.default_rng(0)
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "legend.frameon": False})

sims = {n: np.load(f"{DATA}/simulations/heldout/{n}_n{args.n_univ}.npz") for n in TEST}
res = {}
for pop, info in POPS.items():
    fl, xt, ck = load_flow(f"{DATA}/models/flow_{pop}{args.tag}/best.pt", dev)
    d = len(info["names"]); xs, xf = {}, {}
    for n, lam in TEST.items():
        s = sims[n][pop]; s = s[rng.permutation(len(s))[:50000]]; m = min(len(s) * args.nflow_mult, 400000)
        ctx = ls.forward(torch.tensor(lam_dict_to_array(lam))[None]).to(dev).repeat(m, 1)
        with torch.no_grad(): u = torch.cat([fl.sample(ctx[i:i + 50000]) for i in range(0, m, 50000)])
        xs[n], xf[n] = s, xt.inverse(u).cpu().numpy()
    res[pop] = dict(xs=xs, xf=xf)
    fig, ax = plt.subplots(3, d, figsize=(1.9 * d + 0.4, 6.0), squeeze=False)
    for i in range(d):
        lo, hi = info["bounds"][i]
        allv = np.concatenate([xs[n][:, i] for n in TEST]); b = np.linspace(allv.min(), allv.max(), 36)
        for c, n in enumerate(TEST):
            for r in (0, 1):
                h, _ = np.histogram(xs[n][:, i], b, density=True); hf, _ = np.histogram(xf[n][:, i], b, density=True)
                ax[r, i].step(b[:-1], h, where="post", color=COL[c], lw=1.0, alpha=0.9, label=n if (r == 0 and i == 0) else None)
                ax[r, i].plot(0.5 * (b[1:] + b[:-1]), hf, color="k", lw=0.7, ls=(0, (2, 1.5)), label="flow" if (c == 0 and r == 0 and i == 0) else None)
            q = np.linspace(0.005, 0.995, 120); qs, qf = np.quantile(xs[n][:, i], q), np.quantile(xf[n][:, i], q)
            sd = info["bounds"][i][1] - info["bounds"][i][0]; ax[2, i].plot(qs, qf - qs, color=COL[c], lw=1.0)
            # +-2 sigma sampling band of the sim quantile: sqrt(q(1-q)/n)/pdf(q)
            pdf = np.maximum(np.interp(qs, 0.5 * (b[1:] + b[:-1]), h), 1e-3 * h.max()); band = 2 * np.sqrt(q * (1 - q) / len(xs[n])) / pdf
            ax[2, i].fill_between(qs, -band, band, color=COL[c], alpha=0.08, lw=0)
        ax[1, i].set_yscale("log"); ax[1, i].set_ylim(1e-4 * ax[0, i].get_ylim()[1], None)
        ax[2, i].axhline(0, color="0.4", lw=0.5); ax[2, i].set_xlabel(info["labels"][i])
        s_ = np.std(np.concatenate([xs[n][:, i] for n in TEST])); ax[2, i].set_ylim(-0.35 * s_, 0.35 * s_)
        for r in range(2): ax[r, i].set_yticks([]) if r == 0 else None
    ax[0, 0].set_ylabel("density"); ax[1, 0].set_ylabel("density (log)"); ax[2, 0].set_ylabel("flow $-$ sim quantile")
    ax[0, 0].legend(fontsize=6, loc="upper left")
    nev = ", ".join(f"{n} {len(xs[n])}" for n in TEST)
    fig.suptitle(f"{pop.upper()}: held-out universes (colour: simulated, dashed: flow)\nshading = 2$\\sigma$ sampling band of the simulated quantile", fontsize=7)
    fig.tight_layout(); fig.savefig(f"{out}/{pop}_marginals_tails_qq.png", dpi=170); fig.savefig(f"{out}/{pop}_marginals_tails_qq.pdf"); plt.close(fig)

# ---- correlation residuals + KS vs noise
metrics = {}; fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
for ax, (pop, info) in zip(axs, POPS.items()):
    d = len(info["names"]); pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]; xs, xf = res[pop]["xs"], res[pop]["xf"]; metrics[pop] = {}
    for c, n in enumerate(TEST):
        cs, cf = np.corrcoef(xs[n].T), np.corrcoef(xf[n].T); dr = [cf[i, j] - cs[i, j] for i, j in pairs]
        noise = [(1 - cs[i, j] ** 2) / np.sqrt(len(xs[n])) for i, j in pairs]  # 1-sigma sampling error of the sim correlation
        ax.errorbar(np.arange(len(pairs)) + 0.13 * (c - 2), dr, yerr=noise, fmt="o", ms=3, color=COL[c], lw=0.8, capsize=0, label=n if pop == "emri" else None)
        ks = [ks_2samp(xs[n][:, i], xf[n][:, i]).statistic for i in range(d)]
        metrics[pop][n] = dict(n=int(len(xs[n])), ks=[float(k) for k in ks], ks_crit95=float(1.36 * np.sqrt((len(xs[n]) + len(xf[n])) / (len(xs[n]) * len(xf[n])))),
                               corr_resid=[float(v) for v in dr], corr_resid_over_sigma=[float(a / b) for a, b in zip(dr, noise)])
    ax.axhline(0, color="0.4", lw=0.5); ax.set_xticks(range(len(pairs))); ax.set_xticklabels([f"{info['labels'][i]}\n{info['labels'][j]}" for i, j in pairs], fontsize=5.5)
    ax.set_title(f"{pop.upper()}: correlation, flow $-$ sim (bars: 1$\\sigma$)", fontsize=7); ax.set_ylabel("$\\Delta\\rho$")
axs[0].legend(fontsize=5.5, ncol=2); fig.tight_layout(); fig.savefig(f"{out}/correlation_residuals.png", dpi=170); fig.savefig(f"{out}/correlation_residuals.pdf"); plt.close(fig)
json.dump(metrics, open(f"{DATA}/logs/flow/step1_validation.json", "w"), indent=1)
for pop in metrics:
    for n, m in metrics[pop].items():
        print(pop, n, "n=%d" % m["n"], "KS", np.round(m["ks"], 3), "crit %.3f" % m["ks_crit95"], "dRho/sigma max %.1f" % np.max(np.abs(m["corr_resid_over_sigma"])))
if args.extra_out:
    import shutil
    for f in ("emri_marginals_tails_qq.pdf", "tde_marginals_tails_qq.pdf", "correlation_residuals.pdf"): shutil.copy(f"{out}/{f}", args.extra_out)
print("->", out)
