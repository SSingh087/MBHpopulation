"""Plateau plots of F_kk vs finite-difference step for tde_fallback; recommends a step per parameter.
F_kk(m)/F_kk(m=1) should be flat (within ~10%) over a range of multipliers m; pick the centre of the widest plateau."""
import argparse, glob, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("--scan_dir", required=True)
ap.add_argument("--plot_dir", default="/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference/plots/detectability/tde_fisher"); ap.add_argument("--tol", type=float, default=0.10)
a = ap.parse_args(); os.makedirs(a.plot_dir, exist_ok=True)
for surv in ("ZTF", "LSST"):
    fs = sorted(glob.glob(f"{a.scan_dir}/{surv}_chunk_*.npz"))
    if not fs: continue
    D = [np.load(f, allow_pickle=True) for f in fs]
    S = np.concatenate([d["scan_Fkk"][d["ok"].astype(bool)] for d in D]); m = D[0]["scan_mults"]; names = list(D[0]["param_names"])
    print(f"{surv}: {len(S)} events with scans"); 
    if len(S) == 0: continue
    ref = S[:, :, list(m).index(1.0)][:, :, None]; R = S / ref
    fig, ax = plt.subplots(2, 4, figsize=(18, 8)); ax = ax.ravel()
    print(f"{surv} recommended step multiplier (plateau centre, tol {a.tol:.0%}), median over events:")
    for k, n in enumerate(names):
        for e in range(len(S)): ax[k].plot(m, R[e, k], color="C0", alpha=.25)
        med = np.nanmedian(R[:, k], axis=0); ax[k].plot(m, med, "k", lw=2); ax[k].axhspan(1 - a.tol, 1 + a.tol, color="g", alpha=.15)
        ax[k].set_xscale("log"); ax[k].set_yscale("log"); ax[k].set_ylim(0.1, 10); ax[k].set_title(n); ax[k].set_xlabel("step multiplier"); ax[k].set_ylabel(r"$F_{kk}/F_{kk}(1)$")
        good = (np.abs(R[:, k] - 1) < a.tol).mean(axis=0)            # fraction of events within tol at each multiplier
        ax[k].plot(m, good, "r--", label="frac. events within tol"); ax[k].legend(fontsize=7)
        ok = good > 0.7; print(f"  {n:9s}: multipliers with >70% events stable: {list(m[ok])}")
    ax[-1].axis("off"); fig.suptitle(f"tde_fallback Fisher step scan, {surv}"); fig.tight_layout(); fig.savefig(f"{a.plot_dir}/step_scan_{surv}.png", dpi=110)
