#!/usr/bin/env python
"""
sf_brute_mc.py -- one-to-one test of the EMRI selection function: fixed-points estimator vs plain Monte Carlo (the poplar method).

Monte Carlo: for a Lambda draw N sources from the flow, one nuisance draw (Y0 U(0.1,0.7), T U(1,6) yr) each, (e0,p0) emulator -> SNR network -> poplar detection
probability (non-central chi2, df 2, threshold 20), average = alpha.   Fixed points: the same P_det evaluated once on a fixed point set, weighted by p(x|Lambda)/q(x).
Lambda set (40): fiducial, the 4 benchmark corners of the Stage-1 tests, and 35 draws from the prior (fixed seed).
  python sf_brute_mc.py --lam 3 --chunk 1 --n 1e8      one DAG node (seeded per Lambda and chunk)
  python sf_brute_mc.py --report                       merge, compare, figure, JSON
"""
import argparse, glob, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, f"{HERE}/fisher")
from sf_lib import *
ap = argparse.ArgumentParser(); ap.add_argument("--lam", type=int, default=0); ap.add_argument("--chunk", type=int, default=0); ap.add_argument("--n", type=float, default=1e8)
ap.add_argument("--batch", type=int, default=500_000); ap.add_argument("--report", action="store_true"); ap.add_argument("--m", type=int, default=19); ap.add_argument("--tag", default="")
a = ap.parse_args()
SF_DIR = f"{DATA}/simulations/sf"; OUT = f"{DATA}/simulations/sf_mc"; os.makedirs(OUT, exist_ok=True)
CORNERS = [dict(A=8.10, B=3.50, scatter=0.70, logM0=8.70, delta=10.0, theta=1.8), dict(A=7.65, B=5.60, scatter=0.40, logM0=10.2, delta=1.0, theta=0.5),
           dict(A=8.10, B=5.50, scatter=0.40, logM0=8.60, delta=15.0, theta=0.3), dict(A=7.60, B=3.40, scatter=0.70, logM0=9.80, delta=2.0, theta=1.2)]
meta = {k: v for k, v in np.load(f"{DATA}/simulations/consolidated/meta.npz").items()}
rng0 = np.random.default_rng(7); pick = rng0.choice(len(meta["lam"]), 35, replace=False)
LAMS = [np.array([FIDUCIAL[k] for k in LAMBDA_NAMES], np.float32)] + [np.array([c[k] for k in LAMBDA_NAMES], np.float32) for c in CORNERS] + [meta["lam"][j].astype(np.float32) for j in pick]
NAMES = ["fiducial", "heavy_corner", "light_corner", "highA_steep", "lowA_shallow"] + [f"prior_{i:02d}" for i in range(35)]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not a.report:
    sf = EMRISF(str(dev)); ls = lam_scaled(LAMS[a.lam], dev); seed = 1_000_000 * (a.lam + 1) + a.chunk
    torch.manual_seed(seed); gen = torch.Generator(device=dev).manual_seed(seed); s1 = s2 = 0.0; n = 0; t0 = time.time()
    while n < a.n:
        xs = sf.flow_samples(ls, a.batch); p = sf.pdet_points(xs, 1, gen).double(); s1 += p.sum().item(); s2 += (p ** 2).sum().item(); n += a.batch
    np.savez(f"{OUT}/{a.lam:02d}_{a.chunk}{a.tag}.npz", s1=s1, s2=s2, n=n)
    print(f"lam {a.lam} ({NAMES[a.lam]}) chunk {a.chunk}: N={n:.2e} alpha={s1/n:.5e} +- {np.sqrt(max(s2/n-(s1/n)**2,0)/n):.1e}  [{time.time()-t0:.0f}s]"); sys.exit(0)

# ---------------------------------------------------------------- report
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sf = EMRISF(str(dev)); tab = load_table(f"{SF_DIR}/table_m{a.m}.npz", dev); R = []
for k in range(len(LAMS)):
    fs = sorted(glob.glob(f"{OUT}/{k:02d}_*{a.tag}.npz"))
    if not fs: continue
    D = [np.load(f) for f in fs]; n = sum(float(d["n"]) for d in D); s1 = sum(float(d["s1"]) for d in D); s2 = sum(float(d["s2"]) for d in D)
    mc = s1 / n; mce = np.sqrt(max(s2 / n - mc ** 2, 0) / n); fp, fpe, _ = sf.alpha(lam_scaled(LAMS[k], dev), tab, "P_naive")
    R.append(dict(k=k, name=NAMES[k], n=n, mc=mc, mc_err=mce, fp=fp, fp_err=fpe, diff_pct=100 * (fp - mc) / mc, sigma=(fp - mc) / np.hypot(mce, fpe)))
    print(f"{NAMES[k]:14s} N={n:.1e}  MC {mc:.4e} +- {100*mce/mc:.2f}%   fixed points {fp:.4e} +- {100*fpe/fp:.2f}%   diff {R[-1]['diff_pct']:+.2f}%  ({R[-1]['sigma']:+.1f} sigma)")
mc, fp = np.array([r["mc"] for r in R]), np.array([r["fp"] for r in R]); d = np.array([r["diff_pct"] for r in R]); sg = np.array([r["sigma"] for r in R])
de = np.array([100 * np.hypot(r["mc_err"], r["fp_err"]) / r["mc"] for r in R])
print(f"{len(R)} Lambda: mean diff {d.mean():+.2f}%  median |diff| {np.median(np.abs(d)):.2f}%  max |diff| {np.abs(d).max():.2f}%  rms sigma {np.sqrt((sg**2).mean()):.2f}  (1.0 = errors fully explain the scatter)")
json.dump(dict(rows=R, mean_diff_pct=float(d.mean()), median_abs_diff_pct=float(np.median(np.abs(d))), max_abs_diff_pct=float(np.abs(d).max()), rms_sigma=float(np.sqrt((sg ** 2).mean()))), open(f"{DATA}/logs/sf_brute_mc.json", "w"), indent=1, default=float)
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6}); fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax[0].errorbar(mc, fp, yerr=[r["fp_err"] for r in R], fmt="o", ms=3, color="#0072B2", lw=0.8); lim = [mc.min() * 0.8, mc.max() * 1.25]; ax[0].plot(lim, lim, "k-", lw=0.6)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel(r"Monte Carlo $\alpha$"); ax[0].set_ylabel(r"fixed-points $\alpha$")
ax[1].errorbar(np.arange(len(R)), d, de, fmt="o", ms=3, color="#0072B2", lw=0.8); ax[1].axhline(0, color="k", lw=0.6); ax[1].set_xlabel(r"$\Lambda$ (0: fiducial, 1-4: corners, 5+: prior draws)"); ax[1].set_ylabel("(fixed points $-$ Monte Carlo) / Monte Carlo [%]")
fig.tight_layout(); os.makedirs(f"{DATA}/plots/detectability/sf", exist_ok=True); fig.savefig(f"{DATA}/plots/detectability/sf/sf_brute_mc.png", dpi=170); fig.savefig(f"{DATA}/plots/detectability/sf/sf_brute_mc.pdf")
