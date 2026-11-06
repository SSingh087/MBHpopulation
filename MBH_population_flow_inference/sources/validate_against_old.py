#!/usr/bin/env python
"""
Source-simulator validation: new simulator at the FIDUCIAL Lambda vs the old pipeline's saved source catalogues
(use_sigma / hannah / varying-occupation / oleary21; old catalogues are the *pre-detection* sources).

Compares
  (1) expected count: Lambda_tot of new runs vs attrs['Lambda_EMRI_tot'] of the old universes, and realised counts
  (2) pooled marginals: lgMBH, z, spin, e_plunge (KS test + overlay plot)
Run:   python validate_against_old.py --n_new 20          (about 7 s per new draw)
"""
import argparse, glob, os, sys, time, json
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import source_sim as es

DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"
OLD_DIR = "/data/wiay/postgrads/shashwat/EMRI_TDE_data/astrophysical_data/100000"

ap = argparse.ArgumentParser()
ap.add_argument("--n_new", type=int, default=20)
ap.add_argument("--n_pool", type=int, default=100_000)
ap.add_argument("--occupation", default="varying", choices=["varying", "uniform"],
                help="old-catalogue occupation to compare to. 'varying' = Zou+2025 at fiducial params. "
                     "'uniform' needs f_occ=1, which Lambda cannot express, so only 'varying' is meaningful for the new sim.")
ap.add_argument("--seed0", type=int, default=1000)
ap.add_argument("--legacy", action="store_true", help="reproduce the old pipeline quirks (double sigma draw, redraw after PPS)")
args = ap.parse_args()

# ---- old catalogues
pat = f"{OLD_DIR}/galaxy_catalog_data_use_sigma_hannah_{args.occupation}_oleary21_*.h5"
files = sorted(glob.glob(pat))
if not files:
    sys.exit(f"no old catalogues match {pat}")
old = {k: [] for k in ["lgMBH", "z", "spin", "e_plunge"]}
old_Lam, old_N = [], []
for f in files:
    with h5py.File(f, "r") as h:
        old_Lam.append(float(h.attrs["Lambda_EMRI_tot"]))
        g = h["EMRI_catalog"]
        old_N.append(len(g["lgMBH"]))
        old["lgMBH"].append(np.array(g["lgMBH"])); old["z"].append(np.array(g["z_gal"]))
        old["spin"].append(np.array(g["initial_MBHspin"])); old["e_plunge"].append(np.array(g["eccentricities_at_plunge"])
                                                                                  if "eccentricities_at_plunge" in g else np.array([]))
old = {k: np.concatenate(v) for k, v in old.items()}
old_Lam, old_N = np.array(old_Lam), np.array(old_N)
oldT = {"lgMBH": [], "z": []}; oldT_Lam = []
for f in files:
    with h5py.File(f, "r") as h:
        oldT_Lam.append(float(h.attrs["Lambda_TDE_tot"]))
        oldT["lgMBH"].append(np.array(h["TDE_catalog"]["lgMBH"])); oldT["z"].append(np.array(h["TDE_catalog"]["z_gal"]))
oldT = {k: np.concatenate(v) for k, v in oldT.items()}; oldT_Lam = np.array(oldT_Lam)

# ---- new sims at fiducial
new = {k: [] for k in ["lgMBH", "z", "spin", "e_plunge"]}
new_Lam, new_N, times = [], [], []
newT = {"lgMBH": [], "z": []}; newT_Lam, newT_N = [], []
for i in range(args.n_new):
    t0 = time.time()
    r = es.simulate_sources(es.FIDUCIAL, seed=args.seed0 + i, n_pool=args.n_pool, legacy=args.legacy)
    times.append(time.time() - t0)
    new_Lam.append(r["Lambda_tot"]); new_N.append(r["n_events"])
    newT_Lam.append(r["Lambda_tot_tde"]); newT_N.append(r["n_tde_total"])
    newT["lgMBH"].append(r["tde_events"]["lgMBH"]); newT["z"].append(r["tde_events"]["z"])
    ev = r["events"]
    new["lgMBH"].append(ev["lgMBH"]); new["z"].append(ev["z"]); new["spin"].append(ev["spin"]); new["e_plunge"].append(ev["e_plunge"])
    print(f"[{i+1}/{args.n_new}] Lambda_tot={r['Lambda_tot']:.2f} N={r['n_events']} n_eff={r['n_eff']:.0f} t={times[-1]:.1f}s", flush=True)
new = {k: np.concatenate(v) for k, v in new.items()}
new_Lam, new_N = np.array(new_Lam), np.array(new_N)
newT = {k: np.concatenate(v) for k, v in newT.items()}; newT_Lam = np.array(newT_Lam); newT_N = np.array(newT_N)

# ---- report
rep = {}
print("\n=== expected count Lambda_tot (mean, std over universes) ===")
print(f"old ({len(old_Lam)} univ): {old_Lam.mean():.2f} +- {old_Lam.std():.2f}   median {np.median(old_Lam):.2f}")
print(f"new ({len(new_Lam)} univ): {new_Lam.mean():.2f} +- {new_Lam.std():.2f}   median {np.median(new_Lam):.2f}")
print(f"ratio of means new/old = {new_Lam.mean()/old_Lam.mean():.3f}")
print(f"realised N: old mean {old_N.mean():.2f}, new mean {new_N.mean():.2f}")
rep["Lambda_tot_old_mean"] = float(old_Lam.mean()); rep["Lambda_tot_new_mean"] = float(new_Lam.mean())
print("\n=== pooled marginals, KS (new vs old) ===")
labels = dict(lgMBH=r"$\log_{10}M_{BH}$", z="z", spin="spin a", e_plunge=r"$e_{plunge}$")
fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
for a, k in zip(ax, ["lgMBH", "z", "spin", "e_plunge"]):
    o, n = old[k], new[k]
    if len(o) == 0 or len(n) == 0:
        print(f"{k}: missing data (old {len(o)}, new {len(n)})"); continue
    ks = ks_2samp(o, n)
    print(f"{k:9s} n_old={len(o):6d} n_new={len(n):6d}  KS={ks.statistic:.3f}  p={ks.pvalue:.3g}  mean old/new {o.mean():.3f}/{n.mean():.3f}")
    rep[f"KS_{k}"] = [float(ks.statistic), float(ks.pvalue)]
    bins = np.linspace(min(o.min(), n.min()), max(o.max(), n.max()), 40)
    a.hist(o, bins, density=True, histtype="step", lw=2, label=f"old ({len(o)})")
    a.hist(n, bins, density=True, histtype="step", lw=2, label=f"new ({len(n)})")
    a.set_xlabel(labels[k]); a.set_title(f"KS={ks.statistic:.3f}")
ax[0].legend()
plt.tight_layout()
os.makedirs(f"{DATA}/plots", exist_ok=True)
out = f"{DATA}/plots/sources_validation_fiducial_{args.occupation}{"_legacy" if args.legacy else ""}.png"
plt.savefig(out, dpi=140); print("saved", out)
print("\n=== TDE: expected count Lambda_tot ===")
print(f"old ({len(oldT_Lam)} univ): {oldT_Lam.mean():.0f} +- {oldT_Lam.std():.0f}")
print(f"new ({len(newT_Lam)} univ): {newT_Lam.mean():.0f} +- {newT_Lam.std():.0f}   ratio {newT_Lam.mean()/oldT_Lam.mean():.3f}")
print(f"realised TDE N (new, before subsampling) mean {newT_N.mean():.0f}")
fig2, ax2 = plt.subplots(1, 2, figsize=(8, 3.6))
for a, k in zip(ax2, ["lgMBH", "z"]):
    o, n = oldT[k], newT[k]
    ks = ks_2samp(o, n)
    print(f"TDE {k:6s} n_old={len(o)} n_new={len(n)} KS={ks.statistic:.3f} p={ks.pvalue:.3g} mean old/new {o.mean():.3f}/{n.mean():.3f}")
    rep[f"TDE_KS_{k}"] = [float(ks.statistic), float(ks.pvalue)]
    bins = np.linspace(min(o.min(), n.min()), max(o.max(), n.max()), 40)
    a.hist(o, bins, density=True, histtype="step", lw=2, label="old"); a.hist(n, bins, density=True, histtype="step", lw=2, label="new")
    a.set_xlabel(labels[k] if k in labels else k); a.set_title(f"TDE KS={ks.statistic:.3f}")
ax2[0].legend(); plt.tight_layout()
out2 = out.replace(".png", "_TDE.png"); plt.savefig(out2, dpi=140); print("saved", out2)
rep["TDE_Lambda_old_mean"] = float(oldT_Lam.mean()); rep["TDE_Lambda_new_mean"] = float(newT_Lam.mean())
rep["time_per_draw_s"] = float(np.mean(times))
with open(f"{DATA}/logs/sources_validation_{args.occupation}{"_legacy" if args.legacy else ""}.json", "w") as f:
    json.dump(rep, f, indent=1)
