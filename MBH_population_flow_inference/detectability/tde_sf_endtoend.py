#!/usr/bin/env python
"""
End-to-end test of the TDE selection function: alpha_true (full redback survey pipeline on simulator events) vs the classifier-based alpha.

For each benchmark Lambda (5, as in the Stage-1 tests) and survey (ZTF, LSST):
  1. simulate FRESH universes at that Lambda (seeds 96M+, disjoint from training, acceptance 90M+, heldout_extra 95M+); keep ALL stored TDE events;
     the in-range ones (z<zmax of the classifier) are used, f_in = in-range fraction (events outside count as undetected, as in the SF);
  2. one nuisance draw per event from the classifier's training prior, classifier probability p;
  3. STRATIFIED full-pipeline label (>=2 detections SNR>5, add_source_noise=False, same as the training label): p>=P_LO labelled for every event,
     p<P_LO labelled with probability F_LOW (weight 1/F_LOW).
Estimators over the N in-range draws (all multiplied by f_in):
     alpha_true = sum(w*y)/N ;  alpha_cut = mean(p>0.5) ;  alpha_soft = mean(p)    (the last two on the SAME events -> paired)
  python tde_sf_endtoend.py --survey ZTF --lam 0 --chunk 3 --n_draw 20000 --workers 8     (one DAG node)
  python tde_sf_endtoend.py --report                                                        (merge, figure, JSON)
"""
import argparse, glob, json, logging, os, sys, time
import numpy as np
logging.getLogger("bilby").disabled = True; logging.getLogger("redback").disabled = True
if not hasattr(np, "trapezoid"): np.trapezoid = np.trapz   # redback calls the numpy>=2 name; env has numpy 1.x
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "fisher")); sys.path.insert(0, os.path.join(HERE, "..", "flow")); sys.path.insert(0, os.path.join(HERE, "..", "src"))
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"; OUT = f"{DATA}/simulations/tde_sf_e2e"
from priors import FIDUCIAL
CFG = {"ZTF": dict(sim="ztf", L=500., bands=["ztfg", "ztfr", "ztfi"], zmax=0.5, t0=(58200., 61396.), k=0),
       "LSST": dict(sim="Rubin_10yr_baseline", L=700., bands=["lsstu", "lsstg", "lsstr", "lssti", "lsstz", "lssty"], zmax=4.0, t0=(61220., 64872.), k=1)}
LAMS = {   # = acceptance.py / heldout_extra.py benchmark Lambda
    "fiducial": FIDUCIAL,
    "heavy_corner": dict(A=8.10, B=3.50, scatter=0.70, logM0=8.70, delta=10.0, theta=1.8),
    "light_corner": dict(A=7.65, B=5.60, scatter=0.40, logM0=10.2, delta=1.0, theta=0.5),
    "highA_steep": dict(A=8.10, B=5.50, scatter=0.40, logM0=8.60, delta=15.0, theta=0.3),
    "lowA_shallow": dict(A=7.60, B=3.40, scatter=0.70, logM0=9.80, delta=2.0, theta=1.2)}
P_LO, F_LOW, MIN_DET = 0.01, 0.02, 2
ap = argparse.ArgumentParser()
ap.add_argument("--survey"); ap.add_argument("--lam", type=int, default=0); ap.add_argument("--chunk", type=int, default=0); ap.add_argument("--n_draw", type=int, default=20000)
ap.add_argument("--workers", type=int, default=8); ap.add_argument("--report", action="store_true"); ap.add_argument("--tag", default="")
a = ap.parse_args()


def simulate_label(args):
    sv, th = args
    from redback.simulate_transients import SimulateOpticalTransient
    c = CFG[sv]; fn = SimulateOpticalTransient.simulate_transient_in_ztf if sv == "ZTF" else SimulateOpticalTransient.simulate_transient_in_rubin
    theta = dict(redshift=th["z_gal"], t0_mjd_transient=th["t0_mjd_transient"], t0=th["t0_mjd_transient"], ra=th["ra_rad"], dec=th["dec_rad"], mbh6=10 ** (th["lgMBH"] - 6),
                 mstar=1.0, eta=th["eta"], tvisc=th["tvisc"], bb=th["bb"], rph0=th["rph0"], lphoto=th["lphoto"], leddlimit=1.0)
    try:
        sim = fn(model="tde_fallback", survey=c["sim"], parameters=theta, model_kwargs={}, end_transient_time=c["L"], snr_threshold=5.0, add_source_noise=False)
        df = sim.observations
        if len(df) == 0: return 0
        return int(max([int(g["detected"].sum()) for b, g in df.groupby("band") if b in c["bands"]] or [0]) >= MIN_DET)
    except Exception:
        return -1


def nuisance(rng, n, c):
    lg = lambda lo, hi: np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    return dict(ra_rad=rng.uniform(0, 2 * np.pi, n), dec_rad=np.arcsin(rng.uniform(-1, 1, n)), eta=lg(1e-3, 1e-1), alpha=lg(1e-2, 1e-1), beta=rng.uniform(1.001, 30, n),
                t0_mjd_transient=rng.uniform(*c["t0"], n), tvisc=lg(1., 30.), bb=rng.uniform(0, 2, n), rph0=lg(1e-4, 1e4), lphoto=rng.uniform(0, 2, n))


if not a.report:
    import source_sim as es
    sv = a.survey; c = CFG[sv]; names = list(LAMS); lam = LAMS[names[a.lam]]; rng = np.random.default_rng([888, a.chunk, c["k"], a.lam]); t0 = time.time()
    X, n_all, n_in, u = [], 0, 0, 0
    while sum(len(x) for x in X) < a.n_draw:   # fresh universes until enough in-range events
        seed = 96_000_000 + a.lam * 1_000_000 + c["k"] * 500_000 + a.chunk * 1000 + u; u += 1
        r = es.simulate_sources(lam, seed=seed, max_tde_store=10 ** 8); td = r["tde_events"]
        x = np.stack([td["lgMBH"], td["z"]], 1).astype(float); ok = (x[:, 1] > 0) & (x[:, 1] < c["zmax"]); X.append(x[ok]); n_all += len(x); n_in += int(ok.sum())
    X = np.concatenate(X); X = X[rng.permutation(len(X))[: a.n_draw]]; t_sim = time.time() - t0
    d = dict(lgMBH=X[:, 0], z_gal=X[:, 1], **nuisance(rng, len(X), c))
    from tde_classifier import TDEClassifier
    p = TDEClassifier(sv).prob(d); lab = (p >= P_LO) | (rng.random(len(p)) < F_LOW); w = np.where(p >= P_LO, 1.0, 1.0 / F_LOW)
    idx = np.where(lab)[0]; t1 = time.time()
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = pool.map(simulate_label, [(sv, {k: float(v[i]) for k, v in d.items()}) for i in idx], chunksize=1)
    y = -np.ones(len(p), int); y[idx] = res
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/{sv}_{a.lam}_{a.chunk:04d}{a.tag}.npz", p=p, y=y, w=w, lgMBH=d["lgMBH"], z_gal=d["z_gal"], f_in=n_in / max(n_all, 1), n_univ=u, n_all=n_all, n_in=n_in)
    ok = y[idx] >= 0
    print(f"{sv} lam {a.lam} chunk {a.chunk}: {u} universes in {t_sim:.0f}s, f_in {n_in/n_all:.4f}, draws {len(p)}, labelled {len(idx)} ({len(idx)/len(p):.3f}), failed {(~ok).sum()}, "
          f"detected {int((y == 1).sum())}; labelling {time.time()-t1:.0f}s; alpha_true/in {np.sum(w[idx][ok]*y[idx][ok])/len(p):.3e} alpha_cut/in {np.mean(p>0.5):.3e} alpha_soft/in {p.mean():.3e}")
    sys.exit(0)

# ---------------------------------------------------------------- report
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6}); names = list(LAMS); R = {}
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
for si, sv in enumerate(CFG):
    R[sv] = {}
    for k, nm in enumerate(names):
        fs = sorted(glob.glob(f"{OUT}/{sv}_{k}_*{a.tag}.npz"))
        if not fs: continue
        T, C, S, FI, N, LB, DT = [], [], [], [], [], [], []
        for f in fs:
            D = np.load(f); lab = D["y"] >= 0; n = len(D["p"]); bad = (D["y"] < 0) & ((D["p"] >= P_LO))   # failed full-pipeline labels among must-label events
            T.append(np.sum(D["w"][lab] * D["y"][lab]) / n); C.append(np.mean(D["p"] > 0.5)); S.append(np.mean(D["p"])); FI.append(float(D["f_in"])); N.append(n); LB.append(int(lab.sum()))
        T, C, S, FI, N = map(np.array, (T, C, S, FI, N)); fin = FI.mean()
        m = lambda v: (v * FI).mean()                                      # alpha = f_in * mean over draws (chunk averages with equal n)
        e = lambda v: (v * FI).std(ddof=1) / np.sqrt(len(v))
        # ratio estimator error from chunk scatter (paired quantities)
        ratio_cut = T / np.maximum(C, 1e-12); rc = (T * FI).sum() / (C * FI).sum()
        rs = (T * FI).sum() / (S * FI).sum(); err_rc = np.std(T - rc * C, ddof=1) * np.mean(FI) / np.sqrt(len(T)) / (C * FI).mean(); err_rs = np.std(T - rs * S, ddof=1) * np.mean(FI) / np.sqrt(len(T)) / (S * FI).mean()
        R[sv][nm] = dict(n_chunks=len(fs), n_draws=int(N.sum()), n_labelled=int(sum(LB)), f_in=float(fin), alpha_true=float(m(T)), alpha_true_err=float(e(T)), alpha_cut=float(m(C)), alpha_soft=float(m(S)),
                         true_over_cut=float(rc), true_over_cut_err=float(err_rc), true_over_soft=float(rs), true_over_soft_err=float(err_rs))
        print(sv, nm, {q: (round(v, 4) if isinstance(v, float) else v) for q, v in R[sv][nm].items()})
    ks = [k for k, nm in enumerate(names) if nm in R[sv]]
    ax[si].errorbar(np.array(ks) - 0.1, [R[sv][names[k]]["true_over_cut"] for k in ks], [R[sv][names[k]]["true_over_cut_err"] for k in ks], fmt="o", color="#0072B2", label=r"$\alpha_\mathrm{true}/\alpha_{p>0.5}$")
    ax[si].errorbar(np.array(ks) + 0.1, [R[sv][names[k]]["true_over_soft"] for k in ks], [R[sv][names[k]]["true_over_soft_err"] for k in ks], fmt="s", color="#D55E00", label=r"$\alpha_\mathrm{true}/\alpha_\mathrm{soft}$")
    ax[si].axhline(1, color="k", lw=0.6); ax[si].set_xticks(range(len(names))); ax[si].set_xticklabels(names, rotation=30, fontsize=6); ax[si].set_title(sv, fontsize=8); ax[si].set_ylabel("ratio"); ax[si].legend(fontsize=6)
fig.tight_layout(); os.makedirs(f"{DATA}/plots/detectability/tde_sf_e2e", exist_ok=True)
fig.savefig(f"{DATA}/plots/detectability/tde_sf_e2e/tde_sf_e2e.png", dpi=170); fig.savefig(f"{DATA}/plots/detectability/tde_sf_e2e/tde_sf_e2e.pdf")
json.dump(R, open(f"{DATA}/logs/tde_sf_e2e.json", "w"), indent=1)

# ---------------------------------------------------------------- flow-integral alpha (closes the chain through the flow)
# alpha_grid = integral of p(lgM,z|Lambda) * P(lgM,z) over the fine (lgM, ln z) grid, with P the classifier detection probability marginalised over the nuisance parameters
# on the production grid (simulations/sf/tde_grid_<survey>.npz; same construction as sf_stage.py stage "tde"); compared with alpha_true and the paired classifier alpha.
import torch
from scipy.interpolate import RegularGridInterpolator
from common import DATA as _D, LamScaler, lam_dict_to_array
from diagnostics import load_flow
flow, xt, _ = load_flow(f"{DATA}/models/flow_tde_best/best.pt", torch.device("cpu")); LO, HI, STD, EPS = xt.lo, xt.hi, xt.std, xt.EPS; ls_ = LamScaler()
def logp(x, ls):
    u = xt.forward(x); t = ((x - LO) / (HI - LO)).clamp(EPS, 1 - EPS)
    return flow.log_prob(u, ls.expand(len(x), -1)).double() + (-torch.log(STD) - torch.log(t * (1 - t)) - torch.log(HI - LO)).sum(1).double()
G = {}
for sv, c in CFG.items():
    g = np.load(f"{DATA}/simulations/sf/tde_grid_{sv}.npz"); lgc, zc, Pc, Ps = g["lgc"], g["zc"], g["Pcut"], g["Psoft"]; nl, nz = len(lgc), len(zc); zmax = c["zmax"]
    mk = lambda P: (lambda q_: np.clip(RegularGridInterpolator((lgc, np.log(zc)), P, bounds_error=False, fill_value=None)(q_), 0, 1)); fc, fs = mk(Pc), mk(Ps)
    nlf, nzf = 2 * nl, 2 * nz; lgf = np.linspace(4, 7, nlf + 1); zlf = np.linspace(np.log(1e-3), np.log(zmax), nzf + 1)
    Gl, Gz = np.meshgrid(0.5 * (lgf[1:] + lgf[:-1]), np.exp(0.5 * (zlf[1:] + zlf[:-1])), indexing="ij"); Wf = np.diff(lgf)[:, None] * (Gz * np.diff(zlf)[None, :])
    qf = np.stack([Gl.reshape(-1), np.log(Gz.reshape(-1))], 1); Pcf, Psf = fc(qf).reshape(nlf, nzf), fs(qf).reshape(nlf, nzf)
    pts = torch.tensor(np.stack([Gl.reshape(-1), Gz.reshape(-1)], 1), dtype=torch.float32); G[sv] = {}
    for k, nm in enumerate(names):
        if nm not in R[sv]: continue
        lam = LAMS[nm]; ls = ls_.forward(torch.tensor(lam_dict_to_array(lam))[None])
        with torch.no_grad(): d = np.exp(logp(pts, ls).numpy()).reshape(nlf, nzf) * Wf
        ag_c, ag_s = float((d * Pcf).sum()), float((d * Psf).sum()); r = R[sv][nm]
        r.update(alpha_grid_cut=ag_c, alpha_grid_soft=ag_s, true_over_grid_cut=r["alpha_true"] / ag_c, true_over_grid_cut_err=r["alpha_true_err"] / ag_c, true_over_grid_soft=r["alpha_true"] / ag_s,
                 cut_over_grid_cut=r["alpha_cut"] / ag_c)
        print(sv, nm, f"alpha_true {r['alpha_true']:.4e} +- {r['alpha_true_err']:.1e} | classifier (paired) {r['alpha_cut']:.4e} | flow-grid cut {ag_c:.4e} soft {ag_s:.4e} | true/grid_cut {r['true_over_grid_cut']:.3f} +- {r['true_over_grid_cut_err']:.3f}  paired/grid {r['cut_over_grid_cut']:.3f}")
json.dump(R, open(f"{DATA}/logs/tde_sf_e2e.json", "w"), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
for si, sv in enumerate(CFG):
    ks = [nm for nm in names if nm in R[sv]]; x = np.arange(len(ks))
    ax[si].errorbar(x - 0.1, [R[sv][n]["true_over_grid_cut"] for n in ks], [R[sv][n]["true_over_grid_cut_err"] for n in ks], fmt="o", color="#0072B2", label=r"full pipeline / flow-integral $\alpha$ ($p>0.5$)")
    ax[si].plot(x + 0.1, [R[sv][n]["cut_over_grid_cut"] for n in ks], "s", color="#D55E00", label=r"classifier on simulated events / flow-integral $\alpha$")
    ax[si].axhline(1, color="k", lw=0.6); ax[si].set_xticks(x); ax[si].set_xticklabels(ks, rotation=30, fontsize=6); ax[si].set_title(sv, fontsize=8); ax[si].set_ylabel("ratio"); ax[si].legend(fontsize=5)
fig.tight_layout(); fig.savefig(f"{DATA}/plots/detectability/tde_sf_e2e/tde_sf_e2e_grid.png", dpi=170); fig.savefig(f"{DATA}/plots/detectability/tde_sf_e2e/tde_sf_e2e_grid.pdf")
