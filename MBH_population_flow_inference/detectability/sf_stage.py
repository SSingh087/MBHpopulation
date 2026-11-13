#!/usr/bin/env python
"""sf_stage.py <stage> [options] -- one stage of the selection-function validation DAG (see make_sf_validation_dag.py).

stages:  resid | table | zbreak | brute | mu | flowsim | flowsim_merge | tde | report
All stages write JSON to <DATA>/simulations/sf/ and print a readable summary.  --quick = tiny sizes for a CPU smoke test.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, f"{HERE}/fisher")
from sf_lib import *

ap = argparse.ArgumentParser(); ap.add_argument("stage")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--quick", action="store_true"); ap.add_argument("--m", type=int, default=19); ap.add_argument("--L_mix", type=int, default=64)
ap.add_argument("--K", type=int, default=128); ap.add_argument("--seed", type=int, default=100); ap.add_argument("--n_lambda", type=int, default=6)
ap.add_argument("--n_brute", type=float, default=1e8); ap.add_argument("--n_mu", type=int, default=300)
ap.add_argument("--chunk", type=int, default=0); ap.add_argument("--n_chunks", type=int, default=10)
ap.add_argument("--n_univ_tde", type=int, default=1000); ap.add_argument("--tde_grid", type=int, nargs=3, default=None, help="nl nz K for the TDE grid")
a = ap.parse_args()
torch.set_num_threads(a.threads)
Q = a.quick
if Q: a.m, a.L_mix, a.K, a.n_lambda, a.n_brute, a.n_mu, a.n_univ_tde = 11, 4, 4, 2, 2e5, 4, 20
SF_DIR = f"{DATA}/simulations/sf"; os.makedirs(SF_DIR, exist_ok=True); TAG = "_quick" if Q else ""
RESID = f"{SF_DIR}/resid_model{TAG}.npz"; TABLE = f"{SF_DIR}/table_m{a.m}{TAG}.npz"
meta = {k: v for k, v in np.load(f"{DATA}/simulations/consolidated/meta.npz").items()}
dev = torch.device(a.device); gen = torch.Generator(device=dev).manual_seed(a.seed)
def out(name, obj): json.dump(obj, open(f"{SF_DIR}/{name}{TAG}.json", "w"), indent=1, default=float)
def emri(): return EMRISF(a.device, resid_file=RESID)
def lams(): L = lambda_set(meta, a.n_lambda); return L, [lam_scaled(l, dev) for l in L]
T0 = time.time()
print(f"[{a.stage}] device {dev} quick={Q}", flush=True)

# ===================================================================== resid
if a.stage == "resid":
    snr = load_snr_model(f"{DATA}/models/detectability/emri_snr/model.pth", device="cpu")
    def net_rho(X, dl):
        return np.concatenate([snr.run_on_dataset(build_features(X[i:i + 50000]), n_batches=1, luminosity_distances=dl[i:i + 50000]).reshape(-1).numpy() for i in range(0, len(X), 50000)])
    r = build_resid_model(net_rho, RESID); out("resid", r)
    print(f"residual model from {r['n_rows']} out-of-sample injections; cells with >=30 rows: {r['cells_ok']}/{r['cells_total']}; counts per cell:\n{np.array(r['counts']).reshape(4, 4, 6)}")

# ===================================================================== table
elif a.stage == "table":
    sf = emri(); idx = np.random.default_rng(999).choice(len(meta["lam"]), a.L_mix, replace=False)
    pool = [lam_scaled(meta["lam"][i], dev) for i in idx]
    t = time.time(); x, q = sf.make_points_mix(a.m, a.seed, pool); print(f"points 2^{a.m} (L={a.L_mix}): {time.time()-t:.0f}s")
    t = time.time(); Pn = sf.pdet_points(x, a.K, gen); print(f"P_naive K={a.K}: {time.time()-t:.0f}s")
    Pc = sf.pdet_points(x, a.K, gen, calibrated=True) if sf.resid is not None else Pn
    np.savez(TABLE, x=x.cpu().numpy(), q=q.cpu().numpy(), P_naive=Pn.cpu().numpy(), P_cal=Pc.cpu().numpy(), lam_pool_idx=idx)
    ls = lam_scaled(meta["lam"][0], dev); print("normalisation check at Lambda_0 (should be 1):", (torch.exp(sf.logp_x(x, ls)) / q).mean().item())
    print("saved", TABLE)

# ===================================================================== zbreak
elif a.stage == "zbreak":
    sf = emri(); tab = load_table(TABLE, dev); L, LS = lams(); res = []
    zb = np.geomspace(1e-3, 8, 13); mb = np.arange(4, 7.01, 0.5); eb = np.linspace(0, 1, 6); x = tab["x"].cpu().numpy()
    for k, ls in enumerate(LS):
        an, en, tn = sf.alpha(ls, tab, "P_naive"); ac, ec, tc = sf.alpha(ls, tab, "P_cal"); tn, tc = tn.cpu().numpy(), tc.cpu().numpy()   # (contributions need the per-point terms)
        r = dict(lam=k, alpha_naive=an, err_naive=en, alpha_cal=ac, err_cal=ec, ratio_cal_naive=ac / an)
        for nm, col, edges in [("z", 2, zb), ("lgM", 0, mb), ("e_plunge", 3, eb)]:
            r[nm] = [(float(lo), float(hi), float(tn[(x[:, col] >= lo) & (x[:, col] < hi)].sum() / tn.sum()), float(tc[(x[:, col] >= lo) & (x[:, col] < hi)].sum() / tc.sum())) for lo, hi in zip(edges[:-1], edges[1:])]
        res.append(r)
        print(f"\nL{k}: alpha naive {an:.4e} +- {en:.1e} ; calibrated {ac:.4e} +- {ec:.1e} ; calibrated/naive = {ac/an:.3f}")
        for nm in ("z", "lgM", "e_plunge"): print(f"  share of alpha by {nm} (naive | calibrated): " + "  ".join(f"[{lo:.3g},{hi:.3g}) {fn:.2f}|{fc:.2f}" for lo, hi, fn, fc in r[nm]))
    out("zbreak", res)

# ===================================================================== brute
elif a.stage == "brute":
    sf = emri(); tab = load_table(TABLE, dev); L, LS = lams(); res = []; chunk = 5_000_000 if not Q else 100_000
    for k, ls in list(enumerate(LS))[:2]:
        s1 = s2 = 0.0; n = 0; t = time.time()
        while n < a.n_brute:
            xs = sf.flow_samples(ls, chunk); p = sf.pdet_points(xs, 1, gen).double(); s1 += p.sum().item(); s2 += (p ** 2).sum().item(); n += chunk
        mc = s1 / n; mce = np.sqrt(max(s2 / n - mc ** 2, 0) / n); fp, fpe, _ = sf.alpha(ls, tab, "P_naive")
        res.append(dict(lam=k, mc=mc, mc_err=mce, fp=fp, fp_err=fpe, n=n)); print(f"L{k}: brute force N={n:.2e}: {mc:.5e} +- {mce:.1e} ({mce/mc:.2%}) [{time.time()-t:.0f}s]   fixed-points {fp:.5e} +- {fpe:.1e}   diff {(fp-mc)/mc:+.2%} = {(fp-mc)/np.hypot(mce,fpe):+.1f} sigma")
    out("brute", res)

# ===================================================================== mu check
elif a.stage == "mu":
    sf = emri(); tab = load_table(TABLE, dev); rng = np.random.default_rng(5); us = rng.choice(len(meta["lam"]), a.n_mu, replace=False); R = []
    for u in us:
        ls = lam_scaled(meta["lam"][u], dev); an, _, ac, _ = sf.alphas(ls, tab); R.append((int(u), float(meta["Lambda_tot"][u]), an, ac))
    R = np.array(R); mun, muc = R[:, 1] * R[:, 2], R[:, 1] * R[:, 3]
    print(f"{len(R)} random prior universes.  Lambda_tot: median {np.median(R[:,1]):.1f}.  alpha_naive: median {np.median(R[:,2]):.2e} [{R[:,2].min():.1e},{R[:,2].max():.1e}]")
    for nm, mu in (("naive", mun), ("calibrated", muc)):
        print(f"  expected EMRI detections mu = Lambda_tot*alpha ({nm}): median {np.median(mu):.3f}  mean {mu.mean():.3f}  p10 {np.percentile(mu,10):.3f}  p90 {np.percentile(mu,90):.3f}  max {mu.max():.2f}   frac(mu>1) {np.mean(mu>1):.3f}")
    out("mu", dict(rows=R.tolist(), cols=["universe", "Lambda_tot", "alpha_naive", "alpha_cal"]))

# ===================================================================== flowsim (per chunk) + merge
elif a.stage == "flowsim":
    sf = emri(); tab = load_table(TABLE, dev); us = np.arange(len(meta["lam"]))[a.chunk::a.n_chunks]
    if Q: us = us[:4]
    R = np.zeros((len(us), 3)); t = time.time()
    for i, u in enumerate(us):
        ls = lam_scaled(meta["lam"][u], dev); o = sf.alphas(ls, tab); R[i] = (u, o[0], o[2])
        if i % 100 == 0: print(f"  {i}/{len(us)} {time.time()-t:.0f}s", flush=True)
    np.save(f"{SF_DIR}/flowsim_chunk{a.chunk}{TAG}.npy", R); print("saved chunk", a.chunk)

elif a.stage == "flowsim_merge":
    import glob
    R = np.concatenate([np.load(f) for f in sorted(glob.glob(f"{SF_DIR}/flowsim_chunk*{TAG}.npy"))]); R = R[np.argsort(R[:, 0])]; u = R[:, 0].astype(int)
    sf = emri(); _, X, uid = load_consolidated("emri"); m = np.isin(uid, u); X, uid = torch.tensor(X[m], dtype=torch.float32, device=dev), uid[m]
    Pn = sf.pdet_points(X, a.K, gen).cpu().numpy(); Pc = sf.pdet_points(X, a.K, gen, calibrated=True).cpu().numpy() if sf.resid is not None else Pn
    nu = len(meta["lam"]); lt = meta["Lambda_tot"]; isval = meta["is_val"].astype(bool)
    res = {}
    for nm, P, al in (("naive", Pn, R[:, 1]), ("calibrated", Pc, R[:, 2])):
        S_sim = np.bincount(uid, weights=P, minlength=nu)[u]; V_sim = np.bincount(uid, weights=P ** 2, minlength=nu)[u]; S_fl = lt[u] * al
        for sel_nm, sel in (("all", np.ones(len(u), bool)), ("val universes", isval[u]), ("training universes", ~isval[u])):
            ss, vv, ff = S_sim[sel].sum(), V_sim[sel].sum(), S_fl[sel].sum()
            res[f"{nm}/{sel_nm}"] = dict(S_sim=ss, S_flow=ff, ratio=ss / ff, ratio_err=np.sqrt(vv) / ff, n_univ=int(sel.sum()))
            print(f"[{nm:10s}] {sel_nm:20s} {sel.sum():5d} universes: sum_sim P = {ss:9.2f} +- {np.sqrt(vv):.2f}   sum Lambda_tot*alpha_flow = {ff:9.2f}   sim/flow = {ss/ff:.3f} +- {np.sqrt(vv)/ff:.3f}")
        q4 = np.quantile(S_fl, [0, .25, .5, .75, 1.0]); res[f"{nm}/quartiles"] = []
        for lo, hi in zip(q4[:-1], q4[1:]):
            s = (S_fl >= lo) & (S_fl <= hi); res[f"{nm}/quartiles"].append((lo, hi, S_sim[s].sum() / S_fl[s].sum(), np.sqrt(V_sim[s].sum()) / S_fl[s].sum())); print(f"      [{nm}] expected-detection quartile {lo:.3g}-{hi:.3g}: sim/flow = {S_sim[s].sum()/S_fl[s].sum():.3f} +- {np.sqrt(V_sim[s].sum())/S_fl[s].sum():.3f}")
    out("flowsim_merge", res)

# ===================================================================== TDE
elif a.stage == "tde":
    from tde_classifier import TDEClassifier
    flow, xt, _ = load_flow(f"{DATA}/models/flow_tde_best/best.pt", dev); LO, HI, STD, EPS = xt.lo.to(dev), xt.hi.to(dev), xt.std.to(dev), xt.EPS
    _, Xs, uid = load_consolidated("tde"); isval = meta["is_val"].astype(bool); nU = len(meta["lam"]); rng = np.random.default_rng(3)
    nl, nz, Kd = a.tde_grid or ((24, 32, 8) if Q else (120, 160, 100))
    use_u = np.concatenate([np.where(isval)[0][: a.n_univ_tde], np.where(~isval)[0][: a.n_univ_tde]]) if not Q else np.arange(a.n_univ_tde)
    def logp(x, ls):
        u = xt.forward(x); t = ((x - LO) / (HI - LO)).clamp(EPS, 1 - EPS)
        return flow.log_prob(u, ls.expand(len(x), -1)).double() + (-torch.log(STD) - torch.log(t * (1 - t)) - torch.log(HI - LO)).sum(1).double()
    summary = {}
    for survey, zmax, win in (("ZTF", 0.5, (58200., 61396.)), ("LSST", 4.0, (61220., 64872.))):
        clf = TDEClassifier(survey)
        lg = np.linspace(4, 7, nl + 1); lgc = 0.5 * (lg[1:] + lg[:-1]); zl = np.linspace(np.log(1e-3), np.log(zmax), nz + 1); zc = np.exp(0.5 * (zl[1:] + zl[:-1])); dz = zc * np.diff(zl)
        G_l, G_z = np.meshgrid(lgc, zc, indexing="ij"); n = G_l.size; ev = {}
        rep = lambda v: np.repeat(v.reshape(-1), Kd)
        ev.update(lgMBH=rep(G_l), z_gal=rep(G_z), ra_rad=rng.uniform(0, 2 * np.pi, n * Kd), dec_rad=np.arcsin(rng.uniform(-1, 1, n * Kd)), t0_mjd_transient=rng.uniform(*win, n * Kd),
                  eta=np.exp(rng.uniform(np.log(1e-3), np.log(0.1), n * Kd)), alpha=np.exp(rng.uniform(np.log(0.01), np.log(0.1), n * Kd)), beta=rng.uniform(1.001, 30, n * Kd),
                  tvisc=np.exp(rng.uniform(0, np.log(30), n * Kd)), bb=rng.uniform(0, 2, n * Kd), rph0=np.exp(rng.uniform(np.log(1e-4), np.log(1e4), n * Kd)), lphoto=rng.uniform(0, 2, n * Kd))
        t = time.time(); p = clf.prob(ev).reshape(nl, nz, Kd); Pcut, Psoft = (p > 0.5).mean(2), p.mean(2); print(f"[{survey}] classifier grid {nl}x{nz}x{Kd}: {time.time()-t:.0f}s;  max P_cut {Pcut.max():.3f}")
        # P is tabulated on the (nl x nz) classifier grid; the density is integrated on a grid TWICE as fine with P bilinearly interpolated (same treatment as for the simulator events)
        from scipy.interpolate import RegularGridInterpolator
        mk = lambda P: (lambda q_: np.clip(RegularGridInterpolator((lgc, np.log(zc)), P, bounds_error=False, fill_value=None)(q_), 0, 1))
        fc, fs = mk(Pcut), mk(Psoft)
        nlf, nzf = 2 * nl, 2 * nz; lgf = np.linspace(4, 7, nlf + 1); zlf = np.linspace(np.log(1e-3), np.log(zmax), nzf + 1)
        Gl, Gz = np.meshgrid(0.5 * (lgf[1:] + lgf[:-1]), np.exp(0.5 * (zlf[1:] + zlf[:-1])), indexing="ij"); Wf = np.diff(lgf)[:, None] * (Gz * np.diff(zlf)[None, :])
        qf = np.stack([Gl.reshape(-1), np.log(Gz.reshape(-1))], 1); Pcut_f, Psoft_f = fc(qf).reshape(nlf, nzf), fs(qf).reshape(nlf, nzf)
        pts = torch.tensor(np.stack([Gl.reshape(-1), Gz.reshape(-1)], 1), dtype=torch.float32, device=dev)
        al_cut, al_soft, mass = {}, {}, {}
        for u in use_u:
            d = np.exp(logp(pts, lam_scaled(meta["lam"][u], dev)).cpu().numpy()).reshape(nlf, nzf) * Wf
            al_cut[u], al_soft[u], mass[u] = (d * Pcut_f).sum(), (d * Psoft_f).sum(), d.sum()
        # diagnostic: grid quadrature vs MC over FLOW SAMPLES vs the SIMULATOR events (same interpolated P)  -> separates quadrature error from flow error
        diag = []; np.savez(f"{SF_DIR}/tde_grid_{survey}{TAG}.npz", lgc=lgc, zc=zc, Pcut=Pcut, Psoft=Psoft)
        for u in use_u[:3]:
            ls = lam_scaled(meta["lam"][u], dev); xs_ = torch.cat([xt.inverse(flow.sample(ls.expand(min(500_000, (200_000 if Q else 3_000_000) - s), -1))) for s in range(0, 200_000 if Q else 3_000_000, 500_000)]).cpu().numpy()
            xe = Xs[uid == u]; ev_mean = lambda xx, ff: float(np.where((xx[:, 1] > 1e-3) & (xx[:, 1] < zmax), ff(np.stack([xx[:, 0], np.log(np.clip(xx[:, 1], 1e-3, zmax))], 1)), 0.0).mean())
            diag.append((int(u), float(al_cut[u]), ev_mean(xs_, fc), ev_mean(xe, fc)))
            print(f"[{survey}] universe {u}: alpha_cut  grid-quadrature {al_cut[u]:.4e} | flow-samples MC {ev_mean(xs_, fc):.4e} | simulator events ({len(xe)}) {ev_mean(xe, fc):.4e}")
        res = {}
        for nm, sel_u in (("val universes", [u for u in use_u if isval[u]]), ("training universes", [u for u in use_u if not isval[u]])):
            if not len(sel_u): continue
            m = np.isin(uid, sel_u); x = Xs[m]; ui = uid[m]; ok = (x[:, 1] > 1e-3) & (x[:, 1] < zmax)
            pc = np.where(ok, fc(np.stack([x[:, 0], np.log(np.clip(x[:, 1], 1e-3, zmax))], 1)), 0.0); ps = np.where(ok, fs(np.stack([x[:, 0], np.log(np.clip(x[:, 1], 1e-3, zmax))], 1)), 0.0)
            cnt = np.bincount(ui, minlength=nU).astype(float); lt = meta["Lambda_tot_tde"]
            for key, pp, al in (("cut", pc, al_cut), ("soft", ps, al_soft)):
                Ssim = np.bincount(ui, weights=pp, minlength=nU) / np.maximum(cnt, 1) * lt; Sfl = np.array([lt[u] * al[u] for u in sel_u]); Ss = Ssim[sel_u]
                var = np.bincount(ui, weights=pp ** 2, minlength=nU) / np.maximum(cnt, 1) ** 2 * lt ** 2
                r = dict(S_sim=Ss.sum(), S_flow=Sfl.sum(), ratio=Ss.sum() / Sfl.sum(), err=np.sqrt(var[sel_u].sum()) / Sfl.sum()); res[f"{nm}/{key}"] = r
                print(f"[{survey}] {nm:20s} P_{key:4s}: sim {Ss.sum():10.1f}  flow {Sfl.sum():10.1f}  sim/flow = {r['ratio']:.3f} +- {r['err']:.3f}   (mean expected detections per universe {Sfl.mean():.1f})")
            fm = np.array([mass[u] for u in sel_u]); fsim = np.array([np.mean(ok[ui == u]) for u in sel_u])
            print(f"[{survey}] {nm:20s} mass at z<{zmax}: flow {fm.mean():.4f}   simulator {fsim.mean():.4f}   ratio {fm.mean()/max(fsim.mean(),1e-12):.3f}")
            res[f"{nm}/mass"] = dict(flow=fm.mean(), sim=fsim.mean())
        res["diag"] = diag; summary[survey] = res
        u0 = use_u[0]; print(f"[{survey}] Lambda_u0: alpha_cut {al_cut[u0]:.3e}  alpha_soft {al_soft[u0]:.3e};  expected detections ~ {meta['Lambda_tot_tde'][u0]*al_cut[u0]:.1f}")
    out("tde", summary)

# ===================================================================== report
elif a.stage == "report":
    import plot_sf; plot_sf.validation(TAG)
    for n in ("resid", "zbreak", "brute", "mu", "flowsim_merge", "tde"):
        f = f"{SF_DIR}/{n}{TAG}.json"; print(f"\n===== {n} =====\n" + (open(f).read()[:6000] if os.path.exists(f) else "(missing)"))
print(f"[{a.stage}] done in {time.time()-T0:.0f}s")
