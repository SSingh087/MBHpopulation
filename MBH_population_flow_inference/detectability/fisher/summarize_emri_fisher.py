"""Summary + plots of the EMRI Fisher bank: timing, failures, conditioning, sigma vs SNR, C1 vs C2 consistency."""
import argparse, glob, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("--bank_dir", required=True); ap.add_argument("--tag", default="")
ap.add_argument("--plot_dir", default="/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference/plots/detectability/emri_fisher"); a = ap.parse_args()
os.makedirs(a.plot_dir, exist_ok=True)
D = [np.load(f) for f in sorted(glob.glob(f"{a.bank_dir}/chunk_*.npz"))]
cat = lambda k: np.concatenate([d[k] for d in D if len(d[k])]); names = list(D[0]["param_names"])
ok, ok2, wall = cat("ok").astype(bool), cat("ok2").astype(bool), cat("wall")
print(f"events attempted {len(ok)}, Fisher ok {ok.sum()}, C2 ok {ok2.sum()}; wall/event median {np.median(wall):.0f}s max {wall.max():.0f}s")
C1, th, s1, s2, sn, z, cond = cat("C1")[ok], cat("theta")[ok], cat("snr1")[ok], cat("snr2"), cat("snr_net")[ok], cat("z")[ok], cat("cond1")[ok]
sig = np.sqrt(np.clip(np.diagonal(C1, axis1=1, axis2=2), 0, None))
print("cond(F) percentiles 10/50/90:", np.percentile(cond, [10, 50, 90]).format if False else np.percentile(cond, [10, 50, 90]))
print("SNR network vs Fisher-code SNR: median ratio", np.median(s1 / sn).round(3), " (should be ~1; the network error is ~0.08 dex)")
print("median relative sigma per parameter:"); [print(f"  {n:11s} {np.median(sig[:, i] / np.maximum(np.abs(th[:, i]), 1e-9)):.3g}") for i, n in enumerate(names)]
fig, ax = plt.subplots(2, 3, figsize=(16, 9)); ax = ax.ravel()
ax[0].loglog(sn, s1, "o"); ax[0].plot([5, 500], [5, 500], "k--"); ax[0].set_xlabel("SNR network"); ax[0].set_ylabel("SNR from Fisher code"); ax[0].set_title("SNR consistency")
ax[1].semilogy(wall, "o"); ax[1].set_ylabel("wall time per event [s]"); ax[1].set_title("timing")
ax[2].hist(np.log10(cond), 15); ax[2].set_xlabel("log10 cond(F)"); ax[2].set_title("conditioning")
for j, (i, lab) in enumerate([(0, "sigma_M/M"), (2, "sigma_a"), (6, "sigma_dist/dist")]):
    rel = sig[:, i] / (np.abs(th[:, i]) if i in (0, 6) else 1.0); ax[3 + j].loglog(s1, rel, "o"); ax[3 + j].set_xlabel("SNR (Fisher code)"); ax[3 + j].set_ylabel(lab); ax[3 + j].set_title(lab + " vs SNR (expect ~1/SNR)")
fig.tight_layout(); fig.savefig(f"{a.plot_dir}/{a.tag}_summary.png", dpi=110)
if ok2.any():
    k = ok & ok2; C2 = cat("C2")[k]; s_2 = np.sqrt(np.clip(np.diagonal(C2, axis1=1, axis2=2), 0, None)); s_1 = np.sqrt(np.clip(np.diagonal(cat("C1")[k], axis1=1, axis2=2), 0, None))
    print("C2/C1 sigma ratio median per parameter:", dict(zip(names, np.median(s_2 / s_1, axis=0).round(2))))
