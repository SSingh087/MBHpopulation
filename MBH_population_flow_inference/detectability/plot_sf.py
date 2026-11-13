#!/usr/bin/env python
"""plot_sf.py -- saved figures for the selection-function work (for later reference).
   python plot_sf.py fp_check            # fixed-points check (logs/check_fixed_points_cuda.json)  -> plots/detectability/sf/fp_check.png
   python plot_sf.py validation [--tag _quick]   # sf_validation DAG outputs (simulations/sf/*.json|npz) -> plots/detectability/sf/*.png
Every figure has the quantity, the numbers and the question it answers in its title."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"; SF = f"{DATA}/simulations/sf"; PD = f"{DATA}/plots/detectability/sf"; os.makedirs(PD, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
CL = ["#1b6ca8", "#d1495b", "#2a9d8f", "#e9a03b", "#6a4c93", "#555555"]
J = lambda f: json.load(open(f)) if os.path.exists(f) else None


def fp_check():
    R = J(f"{DATA}/logs/check_fixed_points_cuda.json")
    if R is None: print("no fp_check json"); return
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    # T2: single-estimate scatter vs N_x
    a = ax[0, 0]; ms = sorted(int(m) for m in R["T2"]); nl = len(R["T2"][str(ms[0])])
    for k in range(nl): a.loglog([2 ** m for m in ms], [R["T2"][str(m)][k][1] * 100 for m in ms], "o-", color=CL[k % 6], label=f"$\\Lambda_{k}$")
    xs = np.array([2 ** m for m in ms]); a.loglog(xs, 0.55 * (xs / xs[-1]) ** -0.5, "k--", lw=1, label="$N_x^{-1/2}$"); a.axhline(1, color="gray", ls=":"); a.set_xlabel("$N_x$ fixed points"); a.set_ylabel("scatter of one $\\hat\\alpha$ [%]")
    a.set_title("T2 convergence: ~$N_x^{-0.55}$; <1% needs $2^{20}$ (2.6 s/$\\Lambda$ on GPU)"); a.legend(fontsize=8, ncol=2)
    # T4: fixed vs brute force
    a = ax[0, 1]; T4 = R["T4"]; d = [(t["fp"] - t["mc"]) / t["mc"] * 100 for t in T4]; e = [np.hypot(t["mc_err"], t["fp_err"]) / t["mc"] * 100 for t in T4]
    a.errorbar(range(len(T4)), d, yerr=e, fmt="o", color=CL[0], capsize=4); a.axhline(0, color="k"); a.set_xticks(range(len(T4))); a.set_xticklabels([f"$\\Lambda_{t['lam']}$" for t in T4]); a.set_ylabel("(fixed - brute)/brute [%]")
    a.set_title("T4 vs brute force (5e6 samples): consistent within 0.4-0.9% noise;\nbias below ~0.5% not excluded")
    # T1: indicator integrals
    a = ax[0, 2]; inds = sorted(set(t["ind"] for t in R["T1"] if t["ind"] != "z<0.05"))
    for i, nm in enumerate(inds):
        v = [(t["fp"] - t["mc"]) / t["mc"] * 100 for t in R["T1"] if t["ind"] == nm]; a.plot(range(len(v)), v, "o-", color=CL[i], label=nm)
    a.axhline(0, color="k"); a.set_xlabel("$\\Lambda$ index"); a.set_ylabel("(fixed - flow-sampling)/flow-sampling [%]"); a.legend(fontsize=8); a.set_title("T1 known-answer indicator integrals (no network involved):\nmostly 0-0.5%, a hint of a +0.3% bias")
    # T3: K dependence
    a = ax[1, 0]; Ks = sorted(int(k) for k in R["T3"])
    for k in range(nl): a.semilogx(Ks, [R["T3"][str(K)][k][1] * 100 for K in Ks], "o-", color=CL[k % 6])
    a.set_xlabel("K nuisance draws per point"); a.set_ylabel("scatter of one $\\hat\\alpha$ [%] at $N_x=2^{20}$"); a.set_title("T3: scatter nearly independent of K\n-> the points dominate the noise")
    # T6: P_det vs z
    a = ax[1, 1]; T6 = np.array(R["T6"]); zc = np.sqrt(T6[:, 0] * T6[:, 1]); a.loglog(zc, T6[:, 3], "o-", color=CL[1], label="mean $P_{det}$ (mixture-proposal points)"); a.set_xlabel("z"); a.set_ylabel("mean $P_{det}$")
    a.set_title("T6: $P_{det}$ rises again at z>4\n(low-mass, high-e sources the network rates detectable)"); a.legend(fontsize=8)
    # T7: sensitivity
    a = ax[1, 2]; s = R["T7"]; a.bar(range(len(s)), s, color=CL[2]); a.set_xticks(range(len(s))); a.set_xticklabels([f"$\\Lambda_{k}$" for k in range(len(s))]); a.set_ylabel("d ln $\\alpha$ / d ln $\\rho$")
    a.set_title("T7: sensitivity of $\\alpha$ to a coherent rescaling of $\\rho$:\n0.083 dex (x1.21) bias -> 11-43% in $\\alpha$")
    fig.suptitle("Fixed-points selection-function check (EMRI, mixture proposal, GPU run 2026-11-08): normalisation 1.0002 +- 0.0004 (T0)"); fig.tight_layout(); fig.savefig(f"{PD}/fp_check.png", dpi=130); plt.close(fig); print("saved fp_check.png")


def validation(tag=""):
    def sv(fig, name, title): fig.suptitle(title, fontsize=11); fig.tight_layout(); fig.savefig(f"{PD}/{name}{tag}.png", dpi=130); plt.close(fig); print("saved", f"{name}{tag}.png")
    # ---- residual model
    f = f"{SF}/resid_model{tag}.npz"
    if os.path.exists(f):
        r = np.load(f); pool, ok, cnt = r["pool"], r["ok"], r["counts"]; std = np.where(ok, pool.std(1), np.nan).reshape(4, 4, 6); med = np.where(ok, np.median(pool, 1), np.nan).reshape(4, 4, 6)
        zl = ["<0.1", "0.1-0.3", "0.3-0.7", "0.7-1.5", "1.5-3", ">3"]; fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
        for j, (iz, nm) in enumerate([(1, "z 0.1-0.3"), (3, "z 0.7-1.5"), (5, "z>3")]):
            for row, (arr, lab, cm) in enumerate([(std[:, :, iz], "std of r [dex]", "viridis"), (med[:, :, iz], "median of r [dex]", "coolwarm")]):
                a = ax[row, j]; im = a.imshow(arr, origin="lower", cmap=cm, aspect="auto", **({} if row == 0 else dict(vmin=-0.3, vmax=0.3))); plt.colorbar(im, ax=a, label=lab)
                a.set_xticks(range(4)); a.set_xticklabels(["0-.3", ".3-.6", ".6-.9", ".9-1"]); a.set_yticks(range(4)); a.set_yticklabels(["<4.5", "4.5-5.5", "5.5-6.5", ">6.5"]); a.set_xlabel("$e_0$"); a.set_ylabel("lgM"); a.set_title(nm); a.grid(False)
                for (y, x), v in np.ndenumerate(arr): a.text(x, y, "-" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=8, color="w")
        sv(fig, "resid_model", "SNR-network residual model r=log10($\\rho_{net}/\\rho_{true}$) on 57k OUT-OF-SAMPLE injections, cells (lgM, $e_0$, z); '-' = fewer than 30 rows (fallback to z-bin pool)")
    # ---- table: P_det vs z, per proposal point set
    f = f"{SF}/table_m19{tag}.npz" if not tag else f"{SF}/table_m11{tag}.npz"
    if os.path.exists(f):
        t = np.load(f); z = t["x"][:, 2]; e = np.geomspace(1e-3, 8, 16); zc = np.sqrt(e[1:] * e[:-1]); fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
        for k, c, lab in (("P_naive", CL[0], "naive (network $\\rho$)"), ("P_cal", CL[1], "calibrated (residual model)")): ax[0].loglog(zc, [t[k][(z >= lo) & (z < hi)].mean() if ((z >= lo) & (z < hi)).any() else np.nan for lo, hi in zip(e[:-1], e[1:])], "o-", color=c, label=lab)
        ax[0].set_xlabel("z"); ax[0].set_ylabel("mean $P_{det}$ over the proposal points"); ax[0].legend(); ax[0].set_title("detection probability vs z")
        ax[1].hist(np.log10(t["x"][:, 0] * 0 + 10 ** t["x"][:, 0]), bins=40, color=CL[2], alpha=0.8); ax[1].set_xlabel("lgM of the proposal points"); ax[1].set_title("where the mixture proposal puts points")
        sv(fig, "table", "Fixed-points table: $2^{m}$ points from a mixture of flow densities over the prior (+10% box)")
    # ---- zbreak
    Z = J(f"{SF}/zbreak{tag}.json")
    if Z:
        fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))
        for k, r in enumerate(Z):
            for col, nm in enumerate(("z", "lgM", "e_plunge")):
                v = np.array(r[nm]); mid = np.sqrt(v[:, 0] * v[:, 1]) if nm == "z" else 0.5 * (v[:, 0] + v[:, 1])
                ax[col].plot(mid, v[:, 2], "-", color=CL[k % 6], label=f"$\\Lambda_{k}$"); ax[col].plot(mid, v[:, 3], "--", color=CL[k % 6])
        ax[0].set_xscale("log"); [a.set_ylabel("share of $\\alpha$") for a in ax[:3]]; [a.set_xlabel(x) for a, x in zip(ax[:3], ["z", "lgM", "$e_{plunge}$"])]; ax[0].legend(fontsize=8)
        ax[0].set_title("where does $\\alpha$ come from?\nsolid = naive, dashed = calibrated"); x = np.arange(len(Z)); ax[3].bar(x - 0.2, [r["alpha_naive"] for r in Z], 0.4, yerr=[r["err_naive"] for r in Z], color=CL[0], label="naive"); ax[3].bar(x + 0.2, [r["alpha_cal"] for r in Z], 0.4, yerr=[r["err_cal"] for r in Z], color=CL[1], label="calibrated")
        ax[3].set_yscale("log"); ax[3].set_xticks(x); ax[3].set_xticklabels([f"$\\Lambda_{k}$" for k in x]); ax[3].set_ylabel("$\\alpha$"); ax[3].legend(); ax[3].set_title("$\\alpha$: calibrated/naive = " + ", ".join(f"{r['ratio_cal_naive']:.2f}" for r in Z))
        sv(fig, "zbreak", "EMRI selection function: contributions to $\\alpha$ by z, lgM and $e_{plunge}$ for 6 $\\Lambda$")
    # ---- brute
    B = J(f"{SF}/brute{tag}.json")
    if B:
        fig, ax = plt.subplots(figsize=(7, 4.5)); d = [(b["fp"] - b["mc"]) / b["mc"] * 100 for b in B]; e = [np.hypot(b["mc_err"], b["fp_err"]) / b["mc"] * 100 for b in B]
        ax.errorbar(range(len(B)), d, yerr=e, fmt="o", color=CL[0], capsize=5); ax.axhline(0, color="k"); ax.set_xticks(range(len(B))); ax.set_xticklabels([f"$\\Lambda_{b['lam']}$ (N={b['n']:.0e})" for b in B]); ax.set_ylabel("(fixed-points - brute)/brute [%]")
        sv(fig, "brute", "Bias test: fixed-points $\\alpha$ vs very large brute-force Monte Carlo")
    # ---- mu
    M = J(f"{SF}/mu{tag}.json")
    if M:
        R = np.array(M["rows"]); mun, muc = R[:, 1] * R[:, 2], R[:, 1] * R[:, 3]; fig, ax = plt.subplots(1, 2, figsize=(13, 4.8)); b = np.geomspace(max(min(mun.min(), muc.min()), 1e-4), max(mun.max(), muc.max()) * 1.2, 30)
        ax[0].hist(mun, b, color=CL[0], alpha=0.7, label=f"naive: median {np.median(mun):.3f}"); ax[0].hist(muc, b, color=CL[1], alpha=0.6, label=f"calibrated: median {np.median(muc):.3f}"); ax[0].axvline(1, color="k", ls=":"); ax[0].set_xscale("log"); ax[0].set_xlabel("expected EMRI detections $\\mu=\\Lambda_{tot}\\alpha$ per universe"); ax[0].legend(); ax[0].set_title("expected detections over the prior")
        ax[1].scatter(R[:, 1], mun, s=10, color=CL[0], label="naive"); ax[1].scatter(R[:, 1], muc, s=10, color=CL[1], label="calibrated"); ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("$\\Lambda_{tot}$ (expected EMRIs)"); ax[1].set_ylabel("$\\mu$"); ax[1].legend()
        sv(fig, "mu", "Sanity check: expected number of detected EMRIs per universe (compare with the old paper)")
    # ---- flowsim merge
    Fm = J(f"{SF}/flowsim_merge{tag}.json")
    if Fm:
        fig, ax = plt.subplots(1, 2, figsize=(14, 4.8)); keys = [k for k in Fm if "/" in k and not k.endswith("quartiles")]; x = np.arange(len(keys))
        ax[0].errorbar(x, [Fm[k]["ratio"] for k in keys], yerr=[Fm[k]["ratio_err"] for k in keys], fmt="o", color=CL[0], capsize=4); ax[0].axhline(1, color="k"); ax[0].set_xticks(x); ax[0].set_xticklabels([k.replace("/", "\n") for k in keys], fontsize=8); ax[0].set_ylabel("$\\sum P_{det}$(simulated sources) / $\\sum\\Lambda_{tot}\\alpha_{flow}$"); ax[0].set_title("does the flow carry the right mass where detection happens?")
        for i, nm in enumerate(("naive", "calibrated")):
            q = Fm.get(f"{nm}/quartiles")
            if q: q = np.array(q); ax[1].errorbar(np.arange(len(q)) + 0.12 * i, q[:, 2], yerr=q[:, 3], fmt="o", color=CL[i], capsize=3, label=nm)
        ax[1].axhline(1, color="k"); ax[1].set_xlabel("quartile of expected detections per universe"); ax[1].set_ylabel("sim/flow"); ax[1].legend(); ax[1].set_title("by expected-detection quartile")
        sv(fig, "flowsim", "EMRI flow tail test: simulated-source detection count vs flow-integrated expectation (all 10,000 universes)")
    # ---- TDE
    T = J(f"{SF}/tde{tag}.json")
    if T:
        fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
        for si, sv_ in enumerate(("ZTF", "LSST")):
            if sv_ not in T: continue
            r = T[sv_]; ks = [k for k in r if k.endswith("/cut") or k.endswith("/soft")]; x = np.arange(len(ks))
            ax[si, 0].errorbar(x, [r[k]["ratio"] for k in ks], yerr=[r[k]["err"] for k in ks], fmt="o", color=CL[si], capsize=4); ax[si, 0].axhline(1, color="k"); ax[si, 0].set_xticks(x); ax[si, 0].set_xticklabels([k.replace("/", "\n") for k in ks], fontsize=8); ax[si, 0].set_ylabel("sim / flow"); ax[si, 0].set_title(f"{sv_}: detected count, simulator vs flow")
            g = f"{SF}/tde_grid_{sv_}{tag}.npz"
            if os.path.exists(g):
                d = np.load(g); im = ax[si, 1].pcolormesh(np.log10(d["zc"]), d["lgc"], d["Pcut"], cmap="viridis", vmin=0, vmax=1, shading="auto"); plt.colorbar(im, ax=ax[si, 1], label="$P_{cut}$ (p>0.5 fraction)"); ax[si, 1].set_xlabel("log10 z"); ax[si, 1].set_ylabel("lgM"); ax[si, 1].set_title(f"{sv_}: detection probability over (lgM, z), nuisance-averaged"); ax[si, 1].grid(False)
            dg = r.get("diag")
            if dg:
                dg = np.array(dg); xx = np.arange(len(dg)); w = 0.27
                for i, (lab, col) in enumerate((("grid quadrature", 1), ("flow-samples MC", 2), ("simulator events", 3))): ax[si, 2].bar(xx + (i - 1) * w, dg[:, col], w, color=CL[i], label=lab)
                ax[si, 2].set_xticks(xx); ax[si, 2].set_xticklabels([f"u{int(u)}" for u in dg[:, 0]]); ax[si, 2].set_ylabel("$\\alpha_{cut}$"); ax[si, 2].legend(fontsize=8); ax[si, 2].set_title(f"{sv_}: three ways of computing $\\alpha$")
        sv(fig, "tde", "TDE selection function (2-D grid; classifier p>0.5): flow vs simulator, detection map, quadrature check")


def snr_oos():
    """SNR network vs exact SNR on the 57k out-of-sample injections (what the network error means for detection)."""
    import torch; torch.set_grad_enabled(False)
    sys.path.insert(0, "/home/2673888s/poplar"); sys.path.insert(0, "/home/2673888s/LISA_selection_function/lisa_snr_package"); os.environ["POPLAR_PATH"] = "/home/2673888s/poplar"
    import h5py; from snr_model import load_snr_model, build_features, luminosity_distance
    net = load_snr_model(f"{DATA}/models/detectability/emri_snr/model.pth", device="cpu"); f = h5py.File("/data/wiay/postgrads/shashwat/LISA_SF_DATA/injection_param_with_snr_another_100k_clean.h5")
    d = {k: f[k][:] for k in ["M", "mu", "a", "e0", "p0", "Y0", "t_back", "z", "SNR"]}; X = np.stack([d[k] for k in ["M", "mu", "a", "e0", "p0", "Y0", "t_back"]], 1); dl = torch.tensor(luminosity_distance(d["z"]), dtype=torch.float32)
    rho = np.concatenate([net.run_on_dataset(build_features(X[i:i + 50000]), n_batches=1, luminosity_distances=dl[i:i + 50000]).reshape(-1).numpy() for i in range(0, len(X), 50000)])
    t = d["SNR"]; ok = np.isfinite(t) & (t > 0) & np.isfinite(rho) & (rho > 0); r = np.log10(rho[ok] / t[ok]); tt = t[ok]; lgM = np.log10(d["M"][ok]); e0 = d["e0"][ok]
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.8)); a = ax[0]; a.hexbin(np.log10(tt), np.log10(rho[ok]), gridsize=70, bins="log", cmap="viridis", mincnt=1); lim = [-1, 3.5]; a.plot(lim, lim, "r-", lw=1); a.axvline(np.log10(20), color="w", ls=":"); a.axhline(np.log10(20), color="w", ls=":")
    a.set_xlim(lim); a.set_ylim(lim); a.set_xlabel("log10 true SNR (exact waveforms)"); a.set_ylabel("log10 network $\\rho$"); a.set_title(f"network vs truth, {ok.sum()} held-out injections\nmedian |r| {np.median(abs(r)):.3f} dex, p90 {np.percentile(abs(r),90):.2f}")
    a = ax[1]; e = np.array([0, 3, 10, 20, 40, 100, 1e5]); c = [f"{lo:g}-{hi:g}" if hi < 1e4 else f">{lo:g}" for lo, hi in zip(e[:-1], e[1:])]
    bs = [r[(tt >= lo) & (tt < hi)] for lo, hi in zip(e[:-1], e[1:])]; a.bar(range(6), [b.mean() for b in bs], yerr=[b.std() for b in bs], color=CL[0], capsize=4); a.axhline(0, color="k"); a.set_xticks(range(6)); a.set_xticklabels(c); a.set_xlabel("true SNR bin"); a.set_ylabel("r = log10($\\rho_{net}/\\rho_{true}$): mean +- std [dex]"); a.set_title("bias ~0 near threshold (10-40), scatter 0.23-0.26 dex")
    b = (tt >= 10) & (tt < 40); a = ax[2]; ed = [3, 4.5, 5.5, 6.5, 7.1]; a.bar(range(4), [r[b & (lgM >= lo) & (lgM < hi)].mean() for lo, hi in zip(ed[:-1], ed[1:])], yerr=[r[b & (lgM >= lo) & (lgM < hi)].std() for lo, hi in zip(ed[:-1], ed[1:])], color=CL[1], capsize=4)
    a.axhline(0, color="k"); a.set_xticks(range(4)); a.set_xticklabels([f"{lo}-{hi}" for lo, hi in zip(ed[:-1], ed[1:])]); a.set_xlabel("lgM (true SNR 10-40)"); a.set_ylabel("r [dex]"); a.set_title("residual vs mass: heavy tails at lgM<4.5 and >6.5")
    a = ax[3]; ee = [0, .3, .6, .9, 1.0]; a.bar(range(4), [r[b & (e0 >= lo) & (e0 < hi)].mean() for lo, hi in zip(ee[:-1], ee[1:])], yerr=[r[b & (e0 >= lo) & (e0 < hi)].std() for lo, hi in zip(ee[:-1], ee[1:])], color=CL[2], capsize=4)
    a.axhline(0, color="k"); a.set_xticks(range(4)); a.set_xticklabels([f"{lo}-{hi}" for lo, hi in zip(ee[:-1], ee[1:])]); a.set_xlabel("$e_0$ (true SNR 10-40)"); a.set_ylabel("r [dex]"); a.set_title("residual vs eccentricity: heavy tail at $e_0$>0.9")
    fig.suptitle("EMRI SNR network accuracy on out-of-sample injections (selection uses the network; the Fisher-package SNR is not used)"); fig.tight_layout(); fig.savefig(f"{PD}/snr_network_oos.png", dpi=130); plt.close(fig); print("saved snr_network_oos.png")


def bench():
    fig, ax = plt.subplots(figsize=(9, 4.8)); w = 0.38
    for i, dev in enumerate(("cpu", "cuda")):
        R = J(f"{DATA}/logs/bench_selection_{dev}.json")
        if not R: continue
        r = R[max(R, key=int)]; ks = list(r); ax.bar(np.arange(len(ks)) + (i - 0.5) * w, [r[k] for k in ks], w, color=CL[i], label=f"{dev} ({'8 threads' if dev == 'cpu' else '1 GPU'})")
    ax.set_yscale("log"); ax.set_xticks(range(len(ks))); ax.set_xticklabels([k.replace("_", "\n") for k in ks], fontsize=8); ax.set_ylabel("sources / s (N=1e5)"); ax.legend(); ax.set_title("EMRI selection-function pipeline throughput per stage: flow sampling is the bottleneck")
    fig.tight_layout(); fig.savefig(f"{PD}/benchmark_throughput.png", dpi=130); plt.close(fig); print("saved benchmark_throughput.png")


if __name__ == "__main__":
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
    if sys.argv[1] == "fp_check": fp_check()
    elif sys.argv[1] == "snr_oos": snr_oos()
    elif sys.argv[1] == "bench": bench()
    elif sys.argv[1] == "all_existing": fp_check(); bench(); snr_oos()
    else: validation(tag)
