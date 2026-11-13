#!/usr/bin/env python
"""plot_tde_posteriors.py -- paper-style violins and posterior predictive distributions (PPDs) of the TDE population inference.

Style and content follow the Paper-I hierarchical-inference plots (hierarchical_inference/PLOTTING: ppd_io.draw_bands, plot_violin):
  PPD     smooth ANALYTIC densities (no counting noise) of the DETECTED events, p(x|Lambda) P(x) / alpha(Lambda), marginalised to lgM and log10 z on the
          fine integration grid, for posterior draws of Lambda: dashed mean, 68% and 95% bands, the realised mock events as black steps (normalised),
          the injected-truth curve as a thin dotted line (--no_truth removes it); y axis without ticks.  One row per survey (ZTF and LSST analysed separately).
  violin  one violin per hyper-parameter and survey, dashed 16/50/84 percentiles, injected value dashed, y range = prior box, truth quantile printed.
Run layout <base>/{ZTF,LSST}/posterior.npz (from run_nessai_tde.py); outputs <base>/plots/tde_hyper_violins.{pdf,png}, tde_ppd.{pdf,png}; the PPD arrays are
CACHED in <base>/ppd_cache/fine__<survey>.npz (--recompute refreshes; --cache_only <survey> computes that survey's cache and stops -- used by the DAG).
  python plot_tde_posteriors.py --base $DATA/inference_runs/tde_fid_mu_seed2 [--nppd 500] [--recompute] [--no_truth]
Colours: the paper palette (hierarchical_inference/palette.py): ZTF green, LSST blue.
"""
import argparse, os, sys, json, shutil
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ZTF_C, LSST_C = "#029e73", "#0173b2"       # same values as the Paper-I palette.py
COL = {"ZTF": ZTF_C, "LSST": LSST_C}; SURVEYS = ["ZTF", "LSST"]
COL_W, TEXT_W = 3.375, 6.97                  # MNRAS column / text width (inches), as the paper
YLAB = {"lgM": r"$p(\log_{10}M_\bullet)$", "lz": r"$p(\log_{10}z)$"}
LAB = dict(A=r"$A$", B=r"$B$", scatter=r"$\sigma_\mathrm{scatter}$", logM0=r"$\log_{10}M_0$", delta=r"$\delta$", theta=r"$\theta$")
ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--nppd", type=int, default=500); ap.add_argument("--recompute", action="store_true")
ap.add_argument("--device", default="cpu"); ap.add_argument("--cache_only", default=None, choices=SURVEYS, help="compute and cache the PPD of this survey only, no plots")
ap.add_argument("--no_truth", action="store_true", help="do not draw the injected-truth curve in the PPD"); ap.add_argument("--nbins_cat", type=int, default=60)
a = ap.parse_args()


def setup():
    """paper typeface (Times via newtx, usetex) when LaTeX is available; sizes as hierarchical_inference/PLOTTING/paper_style.py"""
    tex = bool(shutil.which("latex")) and bool(os.popen("kpsewhich newtxtext.sty").read().strip())
    matplotlib.rc("text", usetex=tex)
    if tex: matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{newtxtext,newtxmath}\usepackage{amsmath}"
    matplotlib.rc("font", family="serif", serif=["Times New Roman", "Times", "Nimbus Roman"]); matplotlib.rc("mathtext", fontset="cm" if not tex else "dejavusans")
    matplotlib.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
                                "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
                                "legend.frameon": False, "legend.handlelength": 1.4, "legend.labelspacing": 0.3, "legend.borderpad": 0.2, "savefig.dpi": 300, "pdf.fonttype": 42})


def save(fig, name):
    out = f"{a.base}/plots"; os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"): fig.savefig(f"{out}/{name}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig); print("saved", f"{out}/{name}.pdf (+png)", flush=True)


os.makedirs(f"{a.base}/ppd_cache", exist_ok=True)
P = {k: np.load(f"{a.base}/{k}/posterior.npz", allow_pickle=True) for k in SURVEYS if os.path.exists(f"{a.base}/{k}/posterior.npz")}
names = [str(n) for n in P[next(iter(P))]["names"]]; truth = P[next(iter(P))]["lam_true"]


# ------------------------------------------------------------------ PPD cache (fine grid, analytic)
def marginals(pop, w, nl, nz):
    """normalised densities of the DETECTED events in lgM and log10 z from grid weights w (n_l*n_z, lgM-major)"""
    w = w.reshape(nl, nz); dl = pop.dlgM; dz = pop.dlog10z
    ml = w.sum(1) / dl; mz = w.sum(0) / dz
    return ml / (ml.sum() * dl), mz / (mz.sum() * dz)


def fine_ppd(sv, rng):
    f = f"{a.base}/ppd_cache/fine__{sv}.npz"
    if os.path.exists(f) and not a.recompute: return dict(np.load(f))
    from tde_pop_lib import TDEPopulation
    pop = TDEPopulation(sv, a.device); ul, uz = np.unique(pop.lgM), np.unique(pop.z); nl, nz = len(ul), len(uz)
    pop.dlgM = float(np.diff(ul)[0]); pop.dlog10z = float(np.diff(np.log10(uz)).mean())
    S = P[sv]["samples"]; S = S[rng.choice(len(S), min(a.nppd, len(S)), replace=False)]
    ml, mz = zip(*[marginals(pop, (lambda w: (w / w.sum()).cpu().numpy())(pop.weights(lam.astype(np.float32))), nl, nz) for lam in S])
    ml, mz = np.array(ml), np.array(mz); t0, t1 = marginals(pop, (lambda w: (w / w.sum()).cpu().numpy())(pop.weights(truth.astype(np.float32))), nl, nz)
    mk = np.load(str(P[sv]["mocks"][0]), allow_pickle=True)["truth"]; el = np.linspace(4, 7, a.nbins_cat + 1)
    ez = np.linspace(np.log10(uz[0]) - 0.5 * pop.dlog10z, np.log10(uz[-1]) + 0.5 * pop.dlog10z, a.nbins_cat + 1)
    d = dict(c_lgM=ul, c_lz=np.log10(uz), truth_lgM=t0, truth_lz=t1, cat_e_lgM=el, cat_e_lz=ez, n_events=len(mk),
             cat_d_lgM=np.histogram(mk[:, 0], bins=el, density=True)[0], cat_d_lz=np.histogram(np.log10(mk[:, 1]), bins=ez, density=True)[0])
    for key, h in (("lgM", ml), ("lz", mz)):
        for nm, q in (("p2p5", 2.5), ("p16", 16), ("p50", 50), ("p84", 84), ("p97p5", 97.5)): d[f"{key}_{nm}"] = np.percentile(h, q, axis=0)
        d[f"{key}_mean"] = h.mean(0)
    np.savez(f, **d); return d


rng = np.random.default_rng(0)
if a.cache_only:
    fine_ppd(a.cache_only, rng); print("cached PPD", a.cache_only); sys.exit(0)
setup()

# ------------------------------------------------------------------ violins (paper style: filled, dashed quartile lines, truth dashed, prior range)
from tde_pop_lib import PRIOR_BOX
R = {}; fig, ax = plt.subplots(1, 6, figsize=(TEXT_W, 0.36 * TEXT_W))
for j, nm in enumerate(names):
    for i, k in enumerate(SURVEYS):
        if k not in P: continue
        x = P[k]["samples"][:, j]; vp = ax[j].violinplot(x, positions=[i], widths=0.85, showextrema=False)
        for b in vp["bodies"]: b.set_facecolor(COL[k]); b.set_edgecolor("0.15"); b.set_linewidth(0.5); b.set_alpha(0.85)
        for q, ls in ((16, ":"), (50, "--"), (84, ":")): ax[j].plot([i - 0.38, i + 0.38], [np.percentile(x, q)] * 2, color="0.1", lw=0.6, ls=ls)
        R.setdefault(nm, {})[k] = dict(median=float(np.median(x)), lo16=float(np.percentile(x, 16)), hi84=float(np.percentile(x, 84)), truth_quantile=float((x < truth[j]).mean()))
    lo, hi = PRIOR_BOX["delta" if nm == "delta" else nm]; ax[j].set_ylim(lo, hi); ax[j].axhline(truth[j], color="k", ls="--", lw=0.9)
    ax[j].set_xticks(range(len(SURVEYS))); ax[j].set_xticklabels(SURVEYS, fontsize=7.5); ax[j].set_xlim(-0.6, len(SURVEYS) - 0.4); ax[j].set_ylabel(LAB[nm]); ax[j].minorticks_on(); ax[j].tick_params(which="minor", length=1.5)
    ax[j].tick_params(axis="x", which="both", top=False); ax[j].set_title("  ".join(f"{R[nm][k]['truth_quantile']:.2f}" for k in SURVEYS if k in R.get(nm, {})), fontsize=6.5, color="0.35")
fig.legend(handles=[plt.Line2D([], [], color=COL[k], lw=4, label=k) for k in SURVEYS if k in P] + [plt.Line2D([], [], color="k", ls="--", lw=0.9, label="injected")], loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.99))
fig.tight_layout(w_pad=0.6); save(fig, "tde_hyper_violins"); json.dump(R, open(f"{a.base}/posterior_summary.json", "w"), indent=1)

# ------------------------------------------------------------------ PPDs: one row per survey (lgM, log10 z)
rows = [sv for sv in SURVEYS if sv in P]; fig, ax = plt.subplots(len(rows), 2, figsize=(TEXT_W, 0.29 * TEXT_W * len(rows) + 0.06 * TEXT_W), squeeze=False)
for r, sv in enumerate(rows):
    d = fine_ppd(sv, rng); c = COL[sv]
    for cidx, (key, xl) in enumerate((("lgM", r"$\log_{10}(M_\bullet/M_\odot)$"), ("lz", r"$\log_{10}z$"))):
        axx = ax[r, cidx]; x = d[f"c_{key}"]
        axx.fill_between(x, d[f"{key}_p2p5"], d[f"{key}_p97p5"], color=c, alpha=0.13, lw=0); axx.fill_between(x, d[f"{key}_p16"], d[f"{key}_p84"], color=c, alpha=0.32, lw=0)
        axx.plot(x, d[f"{key}_mean"], color=c, lw=1.0, ls="--")
        if not a.no_truth: axx.plot(x, d[f"truth_{key}"], color="k", lw=0.6, ls=":", zorder=4)
        axx.stairs(d[f"cat_d_{key}"], d[f"cat_e_{key}"], color="k", lw=0.9, zorder=5)
        nz = np.where(d[f"{key}_p97p5"] > 1e-3 * d[f"{key}_p97p5"].max())[0]
        axx.set_xlim(x[max(nz.min() - 3, 0)], x[min(nz.max() + 3, len(x) - 1)]); axx.set_ylim(bottom=0); axx.set_yticks([]); axx.minorticks_on(); axx.tick_params(which="minor", length=1.5)
        axx.set_ylabel(YLAB[key])
        axx.set_xlabel(xl) if r == len(rows) - 1 else axx.tick_params(labelbottom=True)
        if cidx == 0: axx.text(0.03, 0.93, f"{sv}  ($N={int(d['n_events'])}$)", transform=axx.transAxes, va="top", fontsize=8)
fig.legend(handles=[plt.Line2D([], [], color=COL[s], lw=1.4, ls="--", label=s) for s in rows] + [plt.Line2D([], [], color="k", lw=1.2, label="mock events")]
           + ([] if a.no_truth else [plt.Line2D([], [], color="k", lw=0.8, ls=":", label="injected $\\Lambda$")]), loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.995))
fig.tight_layout(h_pad=0.8, w_pad=0.8); save(fig, "tde_ppd")
print("posterior summary:"); [print(nm, {k: f"{v['median']:.3f} [{v['lo16']:.3f},{v['hi84']:.3f}] q={v['truth_quantile']:.2f}" for k, v in dd.items()}) for nm, dd in R.items()]
