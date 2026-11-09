#!/usr/bin/env python
"""
Out-of-sample test of the TDE detectability classifiers (tde_fallback; ZTF 512x4, LSST 128x4) with a full-pipeline label.

For fresh events (seeds disjoint from the classifier training) we (i) get the classifier probability p and (ii) re-run the SAME survey simulation that
produced the training labels (redback SimulateOpticalTransient; ZTF: 'ztf', LSST: 'Rubin_10yr_baseline'; add_source_noise=False, snr_threshold=5,
detected if >= 2 detections in any band; window 500 d ZTF / 700 d LSST) -> label y.  Two event sets:
  prior : parameters from the classifier's training prior (lgM U(4,7), z U(0,zmax), eta/alpha log-U, beta U, tvisc, bb, rph0, lphoto, t0 in the survey window)
  cat   : (lgM, z) from the population catalogue (consolidated TDE events, z<=zmax), nuisance from their priors -- the sample the selection function uses.
  python tde_classifier_test.py --survey ZTF --set prior --chunk 3 --n 50 --workers 4     (one DAG node)
  python tde_classifier_test.py --report                                                 (merge + figures + JSON)
"""
import argparse, glob, json, logging, os, sys
import numpy as np
logging.getLogger("bilby").disabled = True; logging.getLogger("redback").disabled = True
if not hasattr(np, "trapezoid"): np.trapezoid = np.trapz   # redback calls numpy>=2 name; env has numpy 1.x
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "fisher")); sys.path.insert(0, os.path.join(HERE, "..", "flow"))
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"; OUT = f"{DATA}/simulations/tde_clf_test"
CFG = {"ZTF": dict(sim="ztf", L=500., bands=["ztfg", "ztfr", "ztfi"], zmax=0.5, zcap=2.0, t0=(58200., 61396.), bank="ZTF"),
       "LSST": dict(sim="Rubin_10yr_baseline", L=700., bands=["lsstu", "lsstg", "lsstr", "lssti", "lsstz", "lssty"], zmax=4.0, zcap=8.0, t0=(61220., 64872.), bank="LSST")}
MIN_DET = 2
ap = argparse.ArgumentParser()
ap.add_argument("--survey"); ap.add_argument("--set", default="prior"); ap.add_argument("--chunk", type=int, default=0); ap.add_argument("--n", type=int, default=50)
ap.add_argument("--workers", type=int, default=4); ap.add_argument("--report", action="store_true"); ap.add_argument("--report_far", action="store_true")
a = ap.parse_args()


def draw(rng, n, sv, setname):
    c = CFG[sv]; lg = lambda lo, hi: np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    d = dict(lgMBH=rng.uniform(4, 7, n), z_gal=rng.uniform(0, c["zmax"], n), ra_rad=rng.uniform(0, 2 * np.pi, n), dec_rad=np.arcsin(rng.uniform(-1, 1, n)),
             eta=lg(1e-3, 1e-1), alpha=lg(1e-2, 1e-1), beta=rng.uniform(1.001, 30, n), t0_mjd_transient=rng.uniform(*c["t0"], n),
             tvisc=lg(1., 30.), bb=rng.uniform(0, 2, n), rph0=lg(1e-4, 1e4), lphoto=rng.uniform(0, 2, n))
    if setname == "ext":   # extension of the TRAINING set: classifier training prior, z uniform in (zmax, zcap]
        d["z_gal"] = rng.uniform(c["zmax"], c["zcap"], n)
    if setname == "far":   # beyond the classifier's training range: z uniform in (zmax, zcap]; lgM from catalogue events at similar z (+-0.15)
        from common import load_consolidated
        _, x, _ = load_consolidated("tde"); z = rng.uniform(c["zmax"], c["zcap"], n); d["z_gal"] = z
        d["lgMBH"] = np.array([rng.choice(x[np.abs(x[:, 1] - zz) < 0.15, 0]) if (np.abs(x[:, 1] - zz) < 0.15).any() else rng.uniform(5.5, 7) for zz in z])
    if setname == "cat":
        from common import load_consolidated
        _, x, _ = load_consolidated("tde"); x = x[x[:, 1] <= c["zmax"]]; i = rng.integers(0, len(x), n)
        d["lgMBH"], d["z_gal"] = x[i, 0].astype(float), x[i, 1].astype(float)
    return d


def simulate_label(args):
    sv, th = args
    from redback.simulate_transients import SimulateOpticalTransient
    c = CFG[sv]; fn = SimulateOpticalTransient.simulate_transient_in_ztf if sv == "ZTF" else SimulateOpticalTransient.simulate_transient_in_rubin
    theta = dict(redshift=th["z_gal"], t0_mjd_transient=th["t0_mjd_transient"], t0=th["t0_mjd_transient"], ra=th["ra_rad"], dec=th["dec_rad"], mbh6=10 ** (th["lgMBH"] - 6),
                 mstar=1.0, eta=th["eta"], tvisc=th["tvisc"], bb=th["bb"], rph0=th["rph0"], lphoto=th["lphoto"], leddlimit=1.0)
    try:
        sim = fn(model="tde_fallback", survey=c["sim"], parameters=theta, model_kwargs={}, end_transient_time=c["L"], snr_threshold=5.0, add_source_noise=False)
        df = sim.observations
        if len(df) == 0: return 0, 0
        nd = max([int(g["detected"].sum()) for b, g in df.groupby("band") if b in c["bands"]] or [0])
        return int(nd >= MIN_DET), nd
    except Exception as e:
        return -1, -1


if not (a.report or a.report_far):
    sv = a.survey; rng = np.random.default_rng([777, a.chunk, 0 if sv == "ZTF" else 1, {"prior": 0, "cat": 1, "far": 2, "ext": 3}[a.set]])
    d = draw(rng, a.n, sv, a.set)
    from tde_classifier import TDEClassifier
    p = TDEClassifier(sv).prob(d)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = pool.map(simulate_label, [(sv, {k: float(v[i]) for k, v in d.items()}) for i in range(a.n)], chunksize=1)
    y = np.array([r[0] for r in res]); nd = np.array([r[1] for r in res])
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/{sv}_{a.set}_{a.chunk:04d}.npz", p=p, y=y, ndet=nd, **d)
    print(f"{sv} {a.set} chunk {a.chunk}: n={a.n} failed={(y < 0).sum()} detected={(y == 1).sum()} mean p={p.mean():.3f}")
    sys.exit(0)

# ---------------------------------------------------------------- report
if a.report_far:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from common import load_consolidated
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6}); _, xc, _ = load_consolidated("tde"); zc = xc[:, 1]; res = {}
    fig, ax = plt.subplots(2, 3, figsize=(10.5, 5.6))
    for r, sv in enumerate(("ZTF", "LSST")):
        c = CFG[sv]; zz = np.linspace(0, c["zcap"], 41); zm = 0.5 * (zz[1:] + zz[:-1])
        fn = sorted(glob.glob(f"{OUT}/{sv}_far_*.npz")); fc = sorted(glob.glob(f"{OUT}/{sv}_cat_*.npz")); fp = sorted(glob.glob(f"{OUT}/{sv}_cat_*.npz"))
        L = lambda fs: {k: np.concatenate([np.load(f)[k] for f in fs]) for k in np.load(fs[0]).files}
        Df = L(fn); Dp = L(fp); okf = Df["y"] >= 0; okp = Dp["y"] >= 0
        # detected fraction vs z: prior set inside the trained range, far set outside; (+ classifier p)
        z_all = np.concatenate([Dp["z_gal"][okp], Df["z_gal"][okf]]); y_all = np.concatenate([Dp["y"][okp], Df["y"][okf]]); p_all = np.concatenate([Dp["p"][okp], Df["p"][okf]])
        bi = np.digitize(z_all, zz) - 1; fy = np.array([y_all[bi == b].mean() if (bi == b).sum() > 20 else np.nan for b in range(40)])
        fp_ = np.array([p_all[bi == b].mean() if (bi == b).sum() > 20 else np.nan for b in range(40)]); ne = np.array([(bi == b).sum() for b in range(40)])
        err = np.sqrt(np.nan_to_num(fy * (1 - fy)) / np.maximum(ne, 1))
        ax[r, 0].errorbar(zm, fy, err, fmt="o", ms=3, color="#0072B2", label="simulated (full pipeline)"); ax[r, 0].plot(zm, fp_, "--", color="#D55E00", label="classifier $p$ (outside its training range beyond the line)")
        ax[r, 0].axvline(c["zmax"], color="k", ls=":", lw=0.8); ax[r, 0].set_xlabel("$z$"); ax[r, 0].set_ylabel(f"{sv}: detected fraction"); ax[r, 0].legend(fontsize=5.5)
        # population-weighted detected number density: catalogue z histogram x detected fraction
        h, _ = np.histogram(zc, zz); h = h / h.sum(); dens = h * np.nan_to_num(fy); cum = np.cumsum(dens) / np.nansum(dens)
        ax[r, 1].step(zm, dens / np.nansum(dens), where="mid", color="#0072B2"); ax[r, 1].axvline(c["zmax"], color="k", ls=":", lw=0.8); ax[r, 1].set_xlabel("$z$"); ax[r, 1].set_ylabel("share of detected sources per $z$ bin (catalogue-weighted)")
        ax[r, 2].plot(zm, 1 - cum, color="#009E73"); ax[r, 2].axvline(c["zmax"], color="k", ls=":", lw=0.8); ax[r, 2].set_yscale("log"); ax[r, 2].set_ylim(1e-4, 1)
        ax[r, 2].set_xlabel("$z_\\mathrm{cut}$"); ax[r, 2].set_ylabel("share of detections missed if the cut is at $z_\\mathrm{cut}$")
        miss = {f"{zc_:.2f}": float(np.nansum(dens[zm > zc_]) / np.nansum(dens)) for zc_ in ([0.3, 0.5, 0.7, 1.0, 1.5] if sv == "ZTF" else [2, 3, 4, 5, 6])}
        res[sv] = dict(n_far=int(okf.sum()), n_far_detected=int(Df["y"][okf].sum()), missed_share_if_cut=miss, f_det_far_bins=[[float(a_), float(b_)] for a_, b_ in zip(zm, fy) if not np.isnan(b_)])
        for k, v in miss.items(): ax[r, 2].annotate(f"{100*v:.1f}%", (float(k), max(v, 1.2e-4)), fontsize=6)
    fig.tight_layout(); os.makedirs(f"{DATA}/plots/detectability/tde_clf_test", exist_ok=True)
    fig.savefig(f"{DATA}/plots/detectability/tde_clf_test/tde_zcut_missed.png", dpi=150); fig.savefig(f"{DATA}/plots/detectability/tde_clf_test/tde_zcut_missed.pdf")
    json.dump(res, open(f"{DATA}/logs/tde_zcut_missed.json", "w"), indent=1); print(json.dumps({k: {a_: b_ for a_, b_ in v.items() if a_ != "f_det_far_bins"} for k, v in res.items()}, indent=1)); sys.exit(0)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6}); os.makedirs(f"{DATA}/plots/detectability/tde_clf_test", exist_ok=True)
R = {}; fig, ax = plt.subplots(2, 4, figsize=(11, 5.2)); COL = {"prior": "#0072B2", "cat": "#D55E00"}
for r, sv in enumerate(("ZTF", "LSST")):
    R[sv] = {}
    for st in ("prior", "cat"):
        fs = sorted(glob.glob(f"{OUT}/{sv}_{st}_*.npz"))
        if not fs: continue
        D = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in np.load(fs[0]).files}; ok = D["y"] >= 0; y, p = D["y"][ok], D["p"][ok]
        pred = p > 0.5; bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        cal = [(p[idx == b].mean(), y[idx == b].mean(), (idx == b).sum()) for b in range(10) if (idx == b).sum() > 5]
        R[sv][st] = dict(n=int(ok.sum()), n_failed=int((~ok).sum()), prevalence=float(y.mean()), mean_p=float(p.mean()), accuracy=float((pred == y).mean()),
                         auc=float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None, frac_pred_det=float(pred.mean()), frac_true_det=float(y.mean()),
                         brier=float(((p - y) ** 2).mean()), calibration=[[float(a_), float(b_), int(c_)] for a_, b_, c_ in cal],
                         purity=float(y[pred].mean()) if pred.any() else None, completeness=float(pred[y == 1].mean()) if y.sum() else None)
        ax[r, 0].plot([c[0] for c in cal], [c[1] for c in cal], "o-", ms=3, color=COL[st], label=f"{st} (n={ok.sum()})")
        if 0 < y.sum() < len(y): fpr, tpr, _ = roc_curve(y, p); ax[r, 1].plot(fpr, tpr, color=COL[st], label=f"{st}: AUC {R[sv][st]['auc']:.3f}")
        zb = np.linspace(0, CFG[sv]["zmax"], 9); zi = np.digitize(D["z_gal"][ok], zb) - 1
        for lab, ls_, v in (("simulated", "-", y), ("classifier p", "--", p)):
            m = [v[zi == b].mean() if (zi == b).sum() > 5 else np.nan for b in range(8)]; ax[r, 2].plot(0.5 * (zb[1:] + zb[:-1]), m, ls_, color=COL[st], lw=1.0, label=f"{st} {lab}" if True else None)
        lb = np.linspace(4, 7, 7); li = np.digitize(D["lgMBH"][ok], lb) - 1
        for lab, ls_, v in (("simulated", "-", y), ("p", "--", p)):
            m = [v[li == b].mean() if (li == b).sum() > 5 else np.nan for b in range(6)]; ax[r, 3].plot(0.5 * (lb[1:] + lb[:-1]), m, ls_, color=COL[st], lw=1.0)
    ax[r, 0].plot([0, 1], [0, 1], "k:", lw=0.6); ax[r, 0].set_xlabel("classifier p"); ax[r, 0].set_ylabel(f"{sv}: observed detected fraction"); ax[r, 0].legend(fontsize=6)
    ax[r, 1].plot([0, 1], [0, 1], "k:", lw=0.6); ax[r, 1].set_xlabel("false positive rate"); ax[r, 1].set_ylabel("true positive rate"); ax[r, 1].legend(fontsize=6)
    ax[r, 2].set_xlabel("$z$"); ax[r, 2].set_ylabel("detected fraction (solid: simulated, dashed: mean p)"); ax[r, 2].legend(fontsize=5)
    ax[r, 3].set_xlabel("$\\lg M_\\bullet$"); ax[r, 3].set_ylabel("detected fraction")
fig.tight_layout(); fig.savefig(f"{DATA}/plots/detectability/tde_clf_test/tde_clf_test.png", dpi=150); fig.savefig(f"{DATA}/plots/detectability/tde_clf_test/tde_clf_test.pdf")
json.dump(R, open(f"{DATA}/logs/tde_clf_test.json", "w"), indent=1)
for sv in R:
    for st, v in R[sv].items(): print(sv, st, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items() if k != "calibration"})
