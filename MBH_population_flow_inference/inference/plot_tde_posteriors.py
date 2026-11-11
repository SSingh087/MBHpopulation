#!/usr/bin/env python
"""plot_tde_posteriors.py -- violin plot of the hyper-parameter posteriors and posterior predictive distributions (PPDs) of the TDE population inference.

Layout of a run directory <base>/:   ZTF/ LSST/ ALL/  each with posterior.npz (from run_nessai_tde.py) ; the mocks are read from the paths stored in posterior.npz.
  violin:  one violin per Lambda parameter and analysis (ZTF, LSST, ZTF+LSST), dashed line = injected truth.
  PPD:     for posterior draws of Lambda: the predicted distribution of DETECTED events in lgM and log10 z, p(x|Lambda) P(x) / alpha(Lambda) x N_events (smooth expectation, no sampling noise),
           mean and 16-84% band per analysis, the expectation at the true Lambda (black) and the realised mock events (grey steps).  The PPD arrays are CACHED in <base>/ppd_cache/ -- rerun with
           --recompute to refresh; plots are redrawn from the cache in seconds.
  python plot_tde_posteriors.py --base $DATA/inference_runs/tde_fid_seed1 [--nppd 300] [--recompute]
Colours: the paper's palette (hierarchical_inference/palette.py): ZTF green, LSST blue, joint tan.
"""
import argparse, os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
matplotlib.rc("font", family="serif", size=11); matplotlib.rc("mathtext", fontset="cm")
ZTF_C, LSST_C, TDE_C = "#029e73", "#0173b2", "#ca9161"      # same values as the Paper-I palette.py (dataset colours)
ANALYSES = [("ZTF", ZTF_C, ["ZTF"]), ("LSST", LSST_C, ["LSST"])]   # (the joint ZTF+LSST analysis is not run: LSST has ~57x more events, ZTF adds ~2%)
LAB = dict(A=r"$A$", B=r"$B$", scatter=r"$\sigma_\mathrm{scatter}$", logM0=r"$\log_{10}M_0$", delta=r"$\delta$", theta=r"$\theta$")
ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--nppd", type=int, default=300); ap.add_argument("--recompute", action="store_true"); ap.add_argument("--device", default="cpu")
a = ap.parse_args(); NB = 30
P = {k: np.load(f"{a.base}/{k}/posterior.npz", allow_pickle=True) for k, _, _ in ANALYSES if os.path.exists(f"{a.base}/{k}/posterior.npz")}
names = [str(n) for n in P[next(iter(P))]["names"]]; truth = P[next(iter(P))]["lam_true"]
out = f"{a.base}/plots"; os.makedirs(out, exist_ok=True); os.makedirs(f"{a.base}/ppd_cache", exist_ok=True); R = {}

# ------------------------------------------------------------------ violins
fig, ax = plt.subplots(1, 6, figsize=(15, 3.1))
for j, nm in enumerate(names):
    for i, (k, c, _) in enumerate(ANALYSES):
        if k not in P: continue
        x = P[k]["samples"][:, j]; vp = ax[j].violinplot(x, positions=[i], widths=0.85, showextrema=False)
        for b in vp["bodies"]: b.set_facecolor(c); b.set_edgecolor(c); b.set_alpha(0.75)
        ax[j].plot([i - 0.42, i + 0.42], [np.median(x)] * 2, color="k", lw=1)
        R.setdefault(nm, {})[k] = dict(median=float(np.median(x)), lo16=float(np.percentile(x, 16)), hi84=float(np.percentile(x, 84)), truth_quantile=float((x < truth[j]).mean()))
    ax[j].axhline(truth[j], color="k", ls="--", lw=1.4); ax[j].set_xticks(range(len(ANALYSES))); ax[j].set_xticklabels([{"ZTF": "ZTF", "LSST": "LSST", "ALL": "ZTF\n+LSST"}[k] for k, _, _ in ANALYSES], fontsize=8); ax[j].set_xlim(-0.6, len(ANALYSES) - 0.4); ax[j].set_ylabel(LAB[nm])
fig.tight_layout(); fig.savefig(f"{out}/tde_hyper_violins.png", dpi=200); fig.savefig(f"{out}/tde_hyper_violins.pdf"); plt.close(fig)
json.dump(R, open(f"{a.base}/posterior_summary.json", "w"), indent=1)

# ------------------------------------------------------------------ PPDs (cached)
from tde_pop_lib import TDEPopulation, ZMAX
import torch


def ppd_for(analysis, survey, pop, N, rng):
    f = f"{a.base}/ppd_cache/{analysis}__{survey}.npz"
    if os.path.exists(f) and not a.recompute:
        d = np.load(f); return d["c_lgM"], d["h_lgM"], d["c_lz"], d["h_lz"]
    S = P[analysis]["samples"]; S = S[rng.choice(len(S), min(a.nppd, len(S)), replace=False)]
    e_l = np.linspace(4, 7, NB + 1); e_z = np.linspace(-3, np.log10(ZMAX[survey]), NB + 1)
    il = torch.tensor(np.clip(np.digitize(pop.lgM, e_l) - 1, 0, NB - 1), device=pop.dev); iz = torch.tensor(np.clip(np.digitize(np.log10(pop.z), e_z) - 1, 0, NB - 1), device=pop.dev)
    hl, hz = [], []
    for lam in S:
        w = pop.weights(lam.astype(np.float32)); w = w / w.sum()
        hl.append(torch.zeros(NB, dtype=torch.float64, device=pop.dev).index_add_(0, il, w).cpu().numpy() * N); hz.append(torch.zeros(NB, dtype=torch.float64, device=pop.dev).index_add_(0, iz, w).cpu().numpy() * N)
    cl, cz = 0.5 * (e_l[1:] + e_l[:-1]), 0.5 * (e_z[1:] + e_z[:-1]); hl, hz = np.array(hl), np.array(hz)
    np.savez(f, c_lgM=cl, h_lgM=hl, c_lz=cz, h_lz=hz); return cl, hl, cz, hz


rng = np.random.default_rng(0); surveys = ["ZTF", "LSST"]
mock = {}; pops = {}
for k in P:
    for sv, mf in zip(P[k]["surveys"], P[k]["mocks"]): mock[str(sv)] = np.load(str(mf), allow_pickle=True)
for sv in surveys:
    if sv in mock: pops[sv] = TDEPopulation(sv, a.device)
fig, ax = plt.subplots(2, 2, figsize=(9, 6.2))
for r, sv in enumerate(surveys):
    if sv not in mock: continue
    N = len(mock[sv]["obs"]); tr = mock[sv]["truth"]; e_l = np.linspace(4, 7, NB + 1); e_z = np.linspace(-3, np.log10(ZMAX[sv]), NB + 1)
    w0 = pops[sv].weights(truth.astype(np.float32)); w0 = (w0 / w0.sum()).cpu().numpy()
    for cidx, (col, key, e, lab) in enumerate([(0, "lgM", e_l, r"$\log_{10}(M_\mathrm{BH}/M_\odot)$"), (1, "lz", e_z, r"$\log_{10}z$")]):
        axx = ax[r, cidx]
        for k, c, sv_list in ANALYSES:
            if k not in P or sv not in sv_list: continue
            cl, hl, cz, hz = ppd_for(k, sv, pops[sv], N, rng); cc, hh = (cl, hl) if key == "lgM" else (cz, hz)
            axx.plot(cc, hh.mean(0), color=c, lw=2, ls="--", label={"ZTF": "ZTF", "LSST": "LSST", "ALL": "ZTF+LSST"}[k]); axx.fill_between(cc, *np.percentile(hh, [16, 84], axis=0), color=c, alpha=0.3)
        g = pops[sv].lgM if key == "lgM" else np.log10(pops[sv].z); ib = np.clip(np.digitize(g, e) - 1, 0, NB - 1); axx.plot(0.5 * (e[1:] + e[:-1]), np.bincount(ib, weights=w0, minlength=NB) * N, color="k", lw=2, label="truth")
        v = tr[:, 0] if key == "lgM" else np.log10(tr[:, 1]); axx.step(e, np.append(np.histogram(v, bins=e)[0], 0), where="post", color="grey", lw=1, label="mock events")
        tc = np.bincount(ib, weights=w0, minlength=NB) * N; nz = np.where(tc > 0.01 * tc.max())[0]; axx.set_xlim(e[max(nz.min() - 3, 0)], e[min(nz.max() + 4, NB)])   # show where the events are
        axx.set_xlabel(lab); axx.set_ylabel(f"{sv}: expected events per bin"); axx.minorticks_on()
ax[0, 0].legend(fontsize=8, frameon=False); fig.tight_layout(); fig.savefig(f"{out}/tde_ppd.png", dpi=200); fig.savefig(f"{out}/tde_ppd.pdf")
print("saved", out); [print(nm, {k: f"{v['median']:.3f} [{v['lo16']:.3f},{v['hi84']:.3f}] q={v['truth_quantile']:.2f}" for k, v in d.items()}) for nm, d in R.items()]
