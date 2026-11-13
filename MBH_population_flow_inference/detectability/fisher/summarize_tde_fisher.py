"""Summary + plots of the TDE Fisher bank: acceptance, timing, conditioning, sigma vs (z, lgM), pre-filter behaviour."""
import argparse, glob, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("--bank_dir", required=True); ap.add_argument("--tag", default="")
ap.add_argument("--plot_dir", default="/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference/plots/detectability/tde_fisher"); a = ap.parse_args()
os.makedirs(a.plot_dir, exist_ok=True)
for surv in ("ZTF", "LSST"):
    fs = sorted(glob.glob(f"{a.bank_dir}/{surv}_chunk_*.npz"))
    if not fs: print(surv, "no files"); continue
    D = [np.load(f, allow_pickle=True) for f in fs]; names = list(D[0]["param_names"])
    cat = lambda k: np.concatenate([d[k] for d in D]); ok = cat("ok").astype(bool)
    z, lgM, p, npts, wall, cond, nd = cat("z"), cat("lgM"), cat("p_detect"), cat("npts"), cat("wall_s"), cat("cond"), cat("n_prefilter_draws")
    C = cat("C"); sig = np.sqrt(np.clip(np.diagonal(C, axis1=1, axis2=2), 0, None))
    print(f"\n=== {surv}: {len(ok)} events in {len(fs)} chunks | Fisher ok {ok.sum()} ({ok.mean():.0%}) | pre-filter: {nd.mean():.2f} prior draws per kept event")
    print(f"  wall/event: median {np.median(wall):.0f}s  p90 {np.percentile(wall,90):.0f}s  max {wall.max():.0f}s | detected points: median {np.median(npts[ok]):.0f}")
    if ok.sum() == 0:
        print(f"  no usable Fisher for {surv}: detected points per event were {sorted(set(npts.astype(int).tolist()))} -- check the survey simulator / t0 window"); continue
    print(f"  of events with p_detect>0.5: {(ok & (p > 0.5)).sum()}/{(p > 0.5).sum()} gave a usable Fisher; among p<0.5: {(ok & (p <= 0.5)).sum()}/{(p <= 0.5).sum()}")
    print(f"  cond(F+prior): median {np.median(cond[ok]):.1e}  max {np.nanmax(cond[ok]):.1e}")
    F = cat("F")[ok]; sym = np.nanmax(np.abs(F - np.swapaxes(F, 1, 2))); ev = np.linalg.eigvalsh(0.5 * (F + np.swapaxes(F, 1, 2)))
    print(f"  Fisher symmetric to {sym:.1e}; min eigenvalue over events {ev.min():.2e} (should be >= ~0)")
    for i, n in enumerate(names): print(f"  sigma_{n:9s} median {np.median(sig[ok, i]):.3g}   p10-p90 {np.percentile(sig[ok, i], 10):.3g} - {np.percentile(sig[ok, i], 90):.3g}")
    fig, ax = plt.subplots(2, 4, figsize=(20, 9)); ax = ax.ravel()
    ax[0].scatter(z, p, c=lgM, s=10); ax[0].set_xscale("log"); ax[0].set_xlabel("z"); ax[0].set_ylabel("P(detect)"); ax[0].set_title("pre-filter output (colour=lgM)")
    ax[1].hist(wall, 20); ax[1].set_xlabel("wall time per event [s]"); ax[1].set_title("timing")
    ax[2].hist(np.log10(cond[ok]), 15); ax[2].set_xlabel("log10 cond(F+prior)"); ax[2].set_title("conditioning")
    ax[3].hist(npts[ok], 20); ax[3].set_xlabel("detected points"); ax[3].set_title("photometry per event")
    for j, (i, lab) in enumerate([(1, "sigma_lgM"), (0, "sigma_z"), (2, "sigma_log10_eta"), (3, "sigma_log10_tvisc")]):
        sc = ax[4 + j].scatter(z[ok], sig[ok, i], c=lgM[ok], s=14); ax[4 + j].set_xscale("log"); ax[4 + j].set_yscale("log"); ax[4 + j].set_xlabel("z"); ax[4 + j].set_ylabel(lab); ax[4 + j].set_title(lab + " vs z")
    fig.suptitle(f"TDE Fisher bank {a.tag}: {surv}"); fig.tight_layout(); fig.savefig(f"{a.plot_dir}/bank_{a.tag}_{surv}.png", dpi=110)
