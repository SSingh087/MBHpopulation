"""Diagnostic plots of the TDE Fisher bank (ZTF / LSST).
00 coverage of the bank in (lgM, z) vs the source catalogue | 01 sigma distributions | 02 sigma vs (z, lgM) maps | 03 correlation structure
04 example posteriors (Gaussian ellipses from the Fisher covariance) | 05 classifier P(detect) vs usable-Fisher fraction | 06 trimming diagnostics"""
import argparse, glob, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
D = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"
ap = argparse.ArgumentParser(); ap.add_argument("--bank_dir", default=f"{D}/simulations/tde_fisher_bank"); ap.add_argument("--out", default=f"{D}/plots/detectability/tde_fisher/bank_prod")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
LAB = {"redshift": "z", "mbh6": "lgM", "eta": "log10 eta", "tvisc": "log10 tvisc", "bb": "bb", "rph0": "log10 rph0", "lphoto": "lphoto"}
catalog = np.load(f"{D}/simulations/consolidated/tde.npz")["x"][::100]          # lgM, z   (200k subsample)
def load(s):
    d = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(f"{a.bank_dir}/{s}_chunk_*.npz"))]; c = lambda k: np.concatenate([x[k] for x in d])
    r = {k: c(k) for k in ["lgM", "z", "eta", "tvisc", "bb", "rph0", "lphoto", "p_detect", "ok", "npts", "cond", "wall_s"]}; r["F"], r["C"] = c("F"), c("C"); r["names"] = list(d[0]["param_names"]); r["ok"] = r["ok"].astype(bool); return r
B = {s: load(s) for s in ("ZTF", "LSST")}
truth = lambda r, n: {"redshift": r["z"], "mbh6": r["lgM"], "eta": np.log10(r["eta"]), "tvisc": np.log10(r["tvisc"]), "bb": r["bb"], "rph0": np.log10(r["rph0"]), "lphoto": r["lphoto"]}[n]

# 00 coverage
fig, ax = plt.subplots(2, 3, figsize=(17, 9))
for i, s in enumerate(B):
    r = B[s]; ok = r["ok"]
    ax[i, 0].hexbin(catalog[:, 1], catalog[:, 0], gridsize=60, bins="log", cmap="Greys", xscale="log", mincnt=1)
    ax[i, 0].scatter(r["z"][ok], r["lgM"][ok], s=2, c="C3", alpha=.3, label="bank (usable Fisher)"); ax[i, 0].set_xlabel("z"); ax[i, 0].set_ylabel("lgM"); ax[i, 0].legend(); ax[i, 0].set_title(f"{s}: bank vs source catalogue (grey)")
    bins = np.linspace(4, 7, 31); ax[i, 1].hist(catalog[:, 0], bins, density=True, histtype="step", color="k", label="catalogue"); ax[i, 1].hist(r["lgM"][ok], bins, density=True, histtype="step", color="C3", label="bank usable"); ax[i, 1].hist(r["lgM"], bins, density=True, histtype="step", color="C0", ls="--", label="bank attempted (after pre-filter)"); ax[i, 1].set_xlabel("lgM"); ax[i, 1].legend(fontsize=7)
    zb = np.geomspace(0.01, 8, 40); ax[i, 2].hist(catalog[:, 1], zb, density=True, histtype="step", color="k"); ax[i, 2].hist(r["z"][ok], zb, density=True, histtype="step", color="C3"); ax[i, 2].set_xscale("log"); ax[i, 2].set_xlabel("z"); ax[i, 2].set_title(f"redshift marginal ({s})")
fig.tight_layout(); fig.savefig(f"{a.out}/00_coverage.png", dpi=110); plt.close(fig)

# 01 sigma distributions
fig, ax = plt.subplots(2, 7, figsize=(26, 7))
for i, s in enumerate(B):
    r = B[s]; ok = r["ok"]; sig = np.sqrt(np.clip(np.diagonal(r["C"], axis1=1, axis2=2), 1e-300, None))
    for k, n in enumerate(r["names"]):
        v = np.log10(sig[ok, k]); hi = ok & (r["p_detect"] > 0.5)
        ax[i, k].hist(v, 40, color="C0", alpha=.6, label="all usable"); ax[i, k].hist(np.log10(sig[hi, k]), 40, color="C3", alpha=.6, label="P(detect)>0.5")
        ax[i, k].set_xlabel(f"log10 sigma_{LAB[n]}"); ax[i, k].set_title(f"{s}: median {np.median(sig[ok,k]):.3g}")
        if k == 0: ax[i, k].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{a.out}/01_sigma_distributions.png", dpi=105); plt.close(fig)

# 02 sigma maps
fig, ax = plt.subplots(2, 4, figsize=(22, 9))
for i, s in enumerate(B):
    r = B[s]; ok = r["ok"]; sig = np.sqrt(np.clip(np.diagonal(r["C"], axis1=1, axis2=2), 1e-300, None))
    for j, (k, nm) in enumerate([(1, "sigma_lgM [dex]"), (0, "sigma_z"), (2, "sigma_log10_eta"), (3, "sigma_log10_tvisc")]):
        h = ax[i, j].hexbin(r["z"][ok], r["lgM"][ok], C=np.log10(sig[ok, k]), reduce_C_function=np.median, gridsize=28, xscale="log", cmap="viridis", mincnt=3)
        plt.colorbar(h, ax=ax[i, j], label=f"median log10 {nm}"); ax[i, j].set_xlabel("z"); ax[i, j].set_ylabel("lgM"); ax[i, j].set_title(f"{s}: {nm}")
fig.tight_layout(); fig.savefig(f"{a.out}/02_sigma_maps.png", dpi=105); plt.close(fig)

# 03 correlations
fig, ax = plt.subplots(2, 4, figsize=(21, 9))
for i, s in enumerate(B):
    r = B[s]; ok = r["ok"]; C = r["C"][ok]; sd = np.sqrt(np.clip(np.diagonal(C, axis1=1, axis2=2), 1e-300, None)); R = C / sd[:, :, None] / sd[:, None, :]; sigl = sd[:, 1]
    order = np.argsort(sigl); picks = [order[len(order) // 20], order[len(order) // 2], order[-len(order) // 20]]
    for j, (mat, ttl) in enumerate([(np.nanmean(np.abs(R), 0), "mean |corr| over events")] + [(R[p], f"event: sigma_lgM={sigl[p]:.3g}, z={r['z'][ok][p]:.2f}") for p in picks]):
        im = ax[i, j].imshow(mat, vmin=-1 if j else 0, vmax=1, cmap="RdBu_r" if j else "viridis"); ax[i, j].set_xticks(range(7)); ax[i, j].set_yticks(range(7))
        ax[i, j].set_xticklabels([LAB[n] for n in r["names"]], rotation=60, fontsize=7); ax[i, j].set_yticklabels([LAB[n] for n in r["names"]], fontsize=7); ax[i, j].set_title(f"{s}: {ttl}", fontsize=8); plt.colorbar(im, ax=ax[i, j], fraction=.046)
fig.tight_layout(); fig.savefig(f"{a.out}/03_correlations.png", dpi=105); plt.close(fig)

# 04 example posteriors: 2D ellipses for (lgM, z), (lgM, log10 eta), (lgM, log10 tvisc) for 3 events per survey
fig, ax = plt.subplots(2, 3, figsize=(18, 11))
for i, s in enumerate(B):
    r = B[s]; ok = np.where(r["ok"] & (r["p_detect"] > 0.5))[0]; sg = np.sqrt(r["C"][ok, 1, 1]); ev = ok[np.argsort(sg)[[len(ok) // 10, len(ok) // 2, 9 * len(ok) // 10]]]
    for j, (k2, nm) in enumerate([(0, "z"), (2, "log10 eta"), (3, "log10 tvisc")]):
        for e, col in zip(ev, ("C0", "C1", "C3")):
            Cs = r["C"][e][np.ix_([1, k2], [1, k2])]; w, V = np.linalg.eigh(Cs); w = np.clip(w, 1e-30, None); ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
            c0 = (r["lgM"][e], truth(r, r["names"][k2])[e])
            for nsig, al in ((1, .5), (2, .25)): ax[i, j].add_patch(Ellipse(c0, 2 * nsig * np.sqrt(w[1]), 2 * nsig * np.sqrt(w[0]), angle=ang, color=col, alpha=al))
            ax[i, j].plot(*c0, "k+")
        ax[i, j].autoscale_view(); ax[i, j].set_xlabel("lgM"); ax[i, j].set_ylabel(nm); ax[i, j].set_title(f"{s}: 1,2 sigma ellipses of 3 events (blue=best, red=poor)")
        lo, hi = ax[i, j].get_xlim(); ax[i, j].set_xlim(max(lo, 3), min(hi, 8))
fig.tight_layout(); fig.savefig(f"{a.out}/04_example_posteriors.png", dpi=105); plt.close(fig)

# 05 classifier vs simulator
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
for s, col in zip(B, ("C0", "C3")):
    r = B[s]; pb = np.linspace(0, 1, 11); idx = np.clip(np.digitize(r["p_detect"], pb) - 1, 0, 9)
    frac = [r["ok"][idx == b].mean() if (idx == b).sum() > 20 else np.nan for b in range(10)]; ax[0].plot(0.5 * (pb[1:] + pb[:-1]), frac, "o-", color=col, label=s)
    ax[1].hist(r["npts"], np.arange(0, 200, 5), histtype="step", color=col, label=s, density=True)
    ax[2].hist(np.log10(np.clip(r["cond"][r["ok"]], 1, None)), 40, histtype="step", color=col, label=s, density=True)
ax[0].plot([0, 1], [0, 1], "k--", lw=.6); ax[0].set_xlabel("classifier P(detect)"); ax[0].set_ylabel("fraction with a usable Fisher (>=5 detected pts)"); ax[0].legend(); ax[0].set_title("classifier vs simulator")
ax[1].set_xlabel("detected points per event"); ax[1].legend(); ax[2].set_xlabel("log10 cond(F + prior)"); ax[2].legend()
fig.tight_layout(); fig.savefig(f"{a.out}/05_classifier_vs_simulator.png", dpi=110); plt.close(fig)

# 06 trimming diagnostics (candidate rules)
print("candidate trimming rules (fraction of usable events kept):")
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for s, col in zip(B, ("C0", "C3")):
    r = B[s]; ok = r["ok"]; sig = np.sqrt(np.clip(np.diagonal(r["C"], axis1=1, axis2=2), 1e-300, None)); good = ok & (sig[:, 1] < 3.0) & (r["cond"] < 1e12)
    print(f"  {s}: usable {ok.sum()} | sigma_lgM<3 dex: {(ok & (sig[:,1] < 3)).sum()} ({(ok & (sig[:,1] < 3)).sum()/ok.sum():.1%}) | cond<1e12: {(ok & (r['cond'] < 1e12)).sum()} | both: {good.sum()} ({good.sum()/ok.sum():.1%}) | sigma_lgM<0.3: {(ok & (sig[:,1] < .3)).sum()/ok.sum():.1%}")
    zb = np.geomspace(0.01, r["z"].max(), 15); idx = np.digitize(r["z"], zb); ax[0].plot([np.sqrt(zb[b-1]*zb[b]) for b in range(1, len(zb))], [good[(idx == b) & ok].mean() if ((idx == b) & ok).sum() > 10 else np.nan for b in range(1, len(zb))], "o-", color=col, label=s)
    ax[1].hist(np.log10(sig[ok, 1]), 60, histtype="step", color=col, label=s)
ax[0].set_xscale("log"); ax[0].set_xlabel("z"); ax[0].set_ylabel("fraction kept (sigma_lgM<3 dex & cond<1e12)"); ax[0].legend(); ax[1].axvline(np.log10(3), color="k", ls="--", label="3 dex (prior width)"); ax[1].set_xlabel("log10 sigma_lgM [dex]"); ax[1].legend()
fig.tight_layout(); fig.savefig(f"{a.out}/06_trimming.png", dpi=110); plt.close(fig)
print("plots in", a.out)
