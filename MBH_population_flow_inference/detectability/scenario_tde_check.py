#!/usr/bin/env python
"""
Stage-2 TDE check per scenario, against the EMRI_TDE_data catalogue (oleary21, use_sigma, <nuc>_<occ>), at the fiducial Lambda:
 (1) FLOW:   p(lgM, z) from flow_tde_best vs all TDEs of the 100 catalogue universes (all_TDE_params): KS, Wasserstein, overlay.
 (2) RATE:   Lambda_tot_tde from source_sim (legacy) vs catalogue mean n_events per universe.
 (3) DETECT: catalogue detected counts (TDE_SNRs_<survey>, n_events_passed) vs flow x classifier map P(lgM,z) (Psoft) x window fraction x catalogue total;
             and the detected (lgM, z) distributions (weighted flow vs catalogue detected).
usage (condor):  EMRI_SCENARIO_ROOT=<scenario dir> python scenario_tde_check.py --name sigmaHV_legacy_a075 --nuc hannah --occ varying --alpha 0.75
"""
import argparse, glob, json, os, sys, time
import numpy as np, h5py, torch
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.interpolate import RegularGridInterpolator
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [f"{ROOT}/flow", f"{ROOT}/src"]
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--nuc", required=True); ap.add_argument("--occ", required=True)
ap.add_argument("--alpha", type=float, default=0.75); ap.add_argument("--legacy", type=int, default=1); ap.add_argument("--n_sim", type=int, default=4)
ap.add_argument("--n_flow", type=int, default=1_000_000); ap.add_argument("--per_file", type=int, default=5000); a = ap.parse_args()
BASE = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"; SC = f"{BASE}/scenarios/{a.name}"
os.environ["EMRI_SCENARIO_ROOT"] = SC; os.environ["EMRI_SCENARIO_SPEC"] = f"{a.nuc},{a.occ},{a.alpha},{a.legacy}"
CAT = "/data/wiay/postgrads/shashwat/EMRI_TDE_data/astrophysical_data/100000"; tag = f"use_sigma_{a.nuc}_{a.occ}_oleary21_tde_fallback"
from common import FIDUCIAL, LamScaler, lam_dict_to_array
from diagnostics import load_flow
import source_sim as es
out = f"{SC}/plots/tde_check"; os.makedirs(out, exist_ok=True); rng = np.random.default_rng(0)
WIN = {"ZTF": (61396 - 58200) / (64423 - 58200), "LSST": (64423 - 60827) / (64423 - 58200)}; ZMAX = {"ZTF": 0.5, "LSST": 4.0}
res = dict(name=a.name)

def sub(x): return x[rng.permutation(len(x))[: a.per_file]]
# ---- catalogue: all TDEs
cl, cz, ntot = [], [], []
for f in sorted(glob.glob(f"{CAT}/all_TDE_params_data_{tag}_*.h5")):
    with h5py.File(f) as h:
        n = len(h["z_gal"]); ntot.append(n); i = np.sort(rng.permutation(n)[: a.per_file]); cl.append(h["lgMBH_mass"][:][i]); cz.append(h["z_gal"][:][i])
cl, cz, ntot = np.concatenate(cl), np.concatenate(cz), np.array(ntot); print("catalogue universes", len(ntot), "mean N_TDE", ntot.mean(), flush=True)
# ---- flow
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); fl, xt, ck = load_flow(f"{SC}/models/flow_tde_best/best.pt", dev)
ctx = LamScaler().forward(torch.tensor(lam_dict_to_array(FIDUCIAL))[None]).to(dev)
with torch.no_grad(): xf = torch.cat([xt.inverse(fl.sample(ctx.repeat(50000, 1))).cpu() for _ in range(a.n_flow // 50000)]).numpy()
res["flow_cols"] = "lgMBH,z (check order)"; print("flow sample", xf.shape, xf[:2], flush=True)
res["flow_vs_cat"] = {k: dict(ks=float(ks_2samp(xf[:, j], c).statistic), wass=float(wasserstein_distance(xf[:, j][:200000], c)), crit=float(1.36 * np.sqrt(1 / len(c) + 1 / 200000)))
                      for j, (k, c) in enumerate((("lgMBH", cl), ("z", cz)))}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for j, (k, c) in enumerate((("lgMBH", cl), ("z", cz))):
    b = np.linspace(min(c.min(), xf[:, j].min()), np.percentile(c, 99.9), 60); ax[j].hist(c, b, density=True, histtype="step", label="catalogue"); ax[j].hist(xf[:, j], b, density=True, histtype="step", label="flow"); ax[j].set_xlabel(k); ax[j].set_yscale("log")
ax[0].legend(); fig.suptitle(f"{a.name}: all TDEs, flow vs catalogue"); fig.tight_layout(); fig.savefig(f"{out}/flow_vs_catalogue.png", dpi=120); plt.close(fig)
# ---- rate
lt = [es.simulate_sources(es.FIDUCIAL, seed=88_000_000 + i, legacy=bool(a.legacy))["Lambda_tot_tde"] for i in range(a.n_sim)]
res["rate"] = dict(catalogue_mean=float(ntot.mean()), catalogue_std=float(ntot.std()), lambda_tot_tde_mean=float(np.mean(lt)), lambda_tot_tde=[float(x) for x in lt])
# ---- detection
res["detect"] = {}
for sv in ("ZTF", "LSST"):
    g = np.load(f"{BASE}/simulations/sf/tde_grid_{sv}.npz"); f = RegularGridInterpolator((g["lgc"], np.log(g["zc"])), g["Psoft"], bounds_error=False, fill_value=None)
    P = np.where(xf[:, 1] < ZMAX[sv], np.clip(f(np.stack([xf[:, 0], np.log(np.maximum(xf[:, 1], 1e-3))], 1)), 0, 1), 0.0)
    npass, ntotal, dl, dz = [], [], [], []
    for fn in sorted(glob.glob(f"{CAT}/TDE_SNRs_{sv}_{tag}_*.h5")):
        with h5py.File(fn) as h:
            npass.append(int(h.attrs["n_events_passed"])); ntotal.append(int(h.attrs["n_events_total"]))
            if "z_gal" in h and len(h["z_gal"]): i = rng.permutation(len(h["z_gal"]))[: a.per_file]; i = np.sort(i); dl.append(h["lgMBH_mass"][:][i]); dz.append(h["z_gal"][:][i])
    npass, ntotal = np.array(npass), np.array(ntotal); dl, dz = np.concatenate(dl), np.concatenate(dz)
    pm = float(P.mean()); mine_cat_tot = ntot.mean() * WIN[sv] * pm; mine_lam = float(np.mean(lt)) * WIN[sv] * pm
    w = P / P.sum(); idx = rng.choice(len(P), 200000, p=w)
    r = dict(cat_passed_mean=float(npass.mean()), cat_passed_std=float(npass.std()), cat_inwindow_total=float(ntotal.mean()), window_frac_expected=WIN[sv], window_frac_cat=float(ntotal.mean() / ntot.mean()),
             mean_P=pm, pred_from_cat_total=float(mine_cat_tot), pred_from_Lambda=mine_lam, ratio_pred_cat_total=float(mine_cat_tot / npass.mean()), ratio_pred_Lambda=float(mine_lam / npass.mean()),
             ks_lgM=float(ks_2samp(xf[idx, 0], dl).statistic), ks_z=float(ks_2samp(xf[idx, 1], dz).statistic), ks_crit=float(1.36 * np.sqrt(1 / len(dl) + 1 / 200000)))
    res["detect"][sv] = r; print(sv, json.dumps(r), flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for j, (k, c) in enumerate((("lgMBH", dl), ("z", dz))):
        b = np.linspace(c.min(), np.percentile(c, 99.9), 50); ax[j].hist(c, b, density=True, histtype="step", label="catalogue detected"); ax[j].hist(xf[idx, j], b, density=True, histtype="step", label="flow x classifier"); ax[j].set_xlabel(k)
    ax[0].legend(); fig.suptitle(f"{a.name} {sv}: detected distributions"); fig.tight_layout(); fig.savefig(f"{out}/detected_{sv}.png", dpi=120); plt.close(fig)
json.dump(res, open(f"{out}/tde_check.json", "w"), indent=1); print("done", out)
