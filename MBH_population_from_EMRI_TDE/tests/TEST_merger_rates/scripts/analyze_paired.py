"""Analysis + plots of the paired test (outputs/paired_N*_s*.json/.npz). Usage: python analyze_paired.py [N=20000]
Writes plots/paired_*.png and outputs/paired_summary_N*.csv, outputs/paired_mass_N*.csv."""
import os, sys, glob, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
N = sys.argv[1] if len(sys.argv) > 1 else "20000"
H = os.path.dirname(os.path.abspath(__file__)); OUT = f"{H}/../outputs"; PL = f"{H}/../plots"
seeds = sorted(int(f.split("_s")[1][:-5]) for f in glob.glob(f"{OUT}/paired_N{N}_s*.json"))
J = {s: json.load(open(f"{OUT}/paired_N{N}_s{s}.json")) for s in seeds}
Z = {s: np.load(f"{OUT}/paired_N{N}_s{s}.npz") for s in seeds}
COL = {"baseline": "#2a78d6", "mundy17": "#eb6834", "oleary_9.7": "#eda100", "oleary_10.3": "#e87ba4", "oleary_11.0": "#008300", "oleary_mass": "#1baf7a", "none": "#52514e", "duan24_pl": "#4a3aa7", "duan24_plexp": "#e34948"}
STY = {"baseline": "-", "mundy17": "-", "oleary_9.7": "-", "oleary_10.3": "--", "oleary_11.0": ":", "oleary_mass": "--", "none": ":", "duan24_pl": "-", "duan24_plexp": "--"}
LAB = {"baseline": "Baseline (1+z)$^2$", "mundy17": "Mundy+17", "oleary_9.7": "EMERGE ≥9.7", "oleary_10.3": "EMERGE ≥10.3", "oleary_11.0": "EMERGE ≥11.0", "oleary_mass": "EMERGE mass-interp.", "none": "No mergers (floor)", "duan24_pl": "Duan+24 PL", "duan24_plexp": "Duan+24 PL+exp"}
INK, MUT = "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9",
    "grid.linewidth": .6, "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUT, "ytick.color": MUT, "legend.frameon": False, "font.size": 11})

def tde_contrib(z, k):   # per-galaxy expected TDE contribution, and active mask, for variant k
    a = (z[k + "|T_c"] < z["t_rlx"]).astype(float)
    g = np.where(z["mask"], z["rate_TDE"] * float(z["T_obs_TDE"]) * a / (1 + z["z"]), 0.0)
    return z["w"] * g, a

# ---- 1. summary table with paired ratios (also relative to the SAME-settings baseline)
names = list(J[seeds[0]])
rows = []
for n in names:
    t = np.array([J[s][n]["Lambda_TDE"] / J[s]["baseline"]["Lambda_TDE"] for s in seeds])
    e = np.array([J[s][n]["Lambda_EMRI"] / J[s]["baseline"]["Lambda_EMRI"] for s in seeds])
    rows.append((n, t.mean(), t.std(ddof=1), e.mean(), e.std(ddof=1), np.mean([J[s][n]["active_fraction"] for s in seeds]), np.mean([J[s][n]["censored"] for s in seeds])))
with open(f"{OUT}/paired_summary_N{N}.csv", "w") as f:
    f.write("variant,TDE_ratio_vs_baseline,sd,EMRI_ratio,sd,active_frac,censored_frac\n")
    for r in rows: f.write(f"{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f},{r[6]:.3f}\n")
# grid-converged ratio: model@grid2000 / baseline@grid2000
conv = {}
for m in [m for m in ("mundy17", "oleary_9.7", "duan24_plexp") if m + "@grid2000" in J[seeds[0]]]:
    r = np.array([J[s][m + "@grid2000"]["Lambda_TDE"] / J[s]["baseline@grid2000"]["Lambda_TDE"] for s in seeds]); conv[m] = (r.mean(), r.std(ddof=1))
print("grid-converged TDE ratio vs converged baseline:", {k: f"{v[0]:.3f}±{v[1]:.3f}" for k, v in conv.items()})

# ---- 2. mass-resolved TDE contribution + active fraction, per lgMBH bin
edges = np.arange(4.0, 7.01, 0.5); cen = 0.5 * (edges[1:] + edges[:-1]); show = ["baseline", "mundy17", "oleary_9.7", "oleary_11.0", "duan24_plexp", "none"]
ratio = {k: [] for k in show}; actf = {k: [] for k in show}; share = {k: [] for k in show}; tcz = {k: [] for k in show}
for s in seeds:
    z = Z[s]; cb, _ = tde_contrib(z, "baseline"); bins = np.digitize(z["lgMBH"], edges) - 1
    for k in show:
        c, a = tde_contrib(z, k); Tc0 = (z[k + "|T_c"] == 0)
        r, af, sh, tz = [], [], [], []
        for b in range(len(cen)):
            sel = (bins == b) & z["mask"]
            r.append(c[sel].sum() / cb[sel].sum() if cb[sel].sum() > 0 else np.nan)
            af.append(np.average(a[sel], weights=z["w"][sel]) if sel.any() else np.nan)
            sh.append(c[sel].sum() / cb.sum())
            tz.append(np.average(Tc0[sel & (a > 0)], weights=z["w"][sel & (a > 0)]) if (sel & (a > 0)).any() else np.nan)
        ratio[k].append(r); actf[k].append(af); share[k].append(sh); tcz[k].append(tz)
A = lambda d: {k: (np.nanmean(v, 0), np.nanstd(v, 0, ddof=1)) for k, v in d.items()}
ratio, actf, share, tcz = A(ratio), A(actf), A(share), A(tcz)
with open(f"{OUT}/paired_mass_N{N}.csv", "w") as f:
    f.write("variant,lgMBH_bin_centre,TDE_ratio_vs_baseline,active_frac,share_of_total_baseline_TDE,frac_of_active_with_Tc_zero\n")
    for k in show:
        for i, c in enumerate(cen): f.write(f"{k},{c:.2f},{ratio[k][0][i]:.4f},{actf[k][0][i]:.4f},{share[k][0][i]:.4f},{tcz[k][0][i]:.4f}\n")
print("\nper-lgMBH-bin (mean over seeds): active fraction / TDE ratio vs baseline / share of baseline TDEs")
print("lgMBH   " + "  ".join(f"{c:5.2f}" for c in cen))
for k in show:
    print(f"{k:11s} act  " + "  ".join(f"{v:5.2f}" for v in actf[k][0]))
for k in show[1:]:
    print(f"{k:11s} rat  " + "  ".join(f"{v:5.2f}" for v in ratio[k][0]))
print(f"{'baseline':11s} shr  " + "  ".join(f"{v:5.2f}" for v in share['baseline'][0]))
print(f"{'baseline':11s} Tc=0 " + "  ".join(f"{v:5.2f}" for v in tcz['baseline'][0]), "(fraction of 'active' with cusp age clipped to 0)")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for k in show:
    ax[0].errorbar(cen, actf[k][0], actf[k][1], color=COL[k], ls=STY[k], lw=2, marker="o", ms=4, label=LAB[k])
    if k != "baseline": ax[1].errorbar(cen, ratio[k][0], ratio[k][1], color=COL[k], ls=STY[k], lw=2, marker="o", ms=4, label=LAB[k])
ax[0].set_xlabel("log$_{10}$ M$_{BH}$"); ax[0].set_ylabel("fraction of galaxies 'active'"); ax[0].set_title("Active fraction vs black-hole mass", loc="left"); ax[0].legend(fontsize=8.5)
ax[1].set_xlabel("log$_{10}$ M$_{BH}$"); ax[1].set_ylabel("expected TDEs / baseline (same mass bin)"); ax[1].set_ylim(0, 1.05); ax[1].set_title("TDE suppression vs mass", loc="left")
ax[2].bar(cen, share["baseline"][0], width=0.4, color=COL["baseline"]); ax[2].set_xlabel("log$_{10}$ M$_{BH}$"); ax[2].set_ylabel("share of baseline TDEs"); ax[2].set_title("Where the baseline TDEs come from", loc="left")
fig.tight_layout(); fig.savefig(f"{PL}/paired_mass_resolved.png", dpi=150); plt.close(fig)

# ---- 3. headline bar chart: paired ratios with error bars, plus grid/kappa sensitivity
order = ["mundy17", "mundy17_n5e4", "oleary_9.7", "oleary_10.3", "oleary_11.0", "oleary_mass", "duan24_pl", "duan24_plexp", "baseline_docM", "none"]
d = {r[0]: r for r in rows}
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
ax[0].bar(range(len(order)), [d[k][1] for k in order], yerr=[d[k][2] for k in order], capsize=3, color=["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948", "#2a78d6", "#898781"], width=.7)
for i, k in enumerate(order): ax[0].text(i, d[k][1] + d[k][2] + .012, f"{d[k][1]:.2f}", ha="center", fontsize=9)
ax[0].set_xticks(range(len(order))); ax[0].set_xticklabels([LAB.get(k, k).replace(" ", "\n", 1) if k in LAB else k.replace("_", "\n") for k in order], fontsize=8)
ax[0].set_ylabel(f"expected TDEs / baseline  (mean ± sd, {len(seeds)} populations)"); ax[0].set_title("Paired TDE reduction (same galaxies, same random numbers)", loc="left"); ax[0].grid(axis="x", visible=False)
lab = ["baseline\nrepo grid", "baseline\ngrid 2000", "baseline\nκ=0.05", "baseline\nκ=1", "Mundy\nrepo grid", "Mundy\ngrid 2000", "Mundy\nκ=0.05", "Mundy\nκ=1"]
key = ["baseline", "baseline@grid2000", "baseline@k0.05", "baseline@k1", "mundy17", "mundy17@grid2000", "mundy17@k0.05", "mundy17@k1"]
ab = [np.mean([J[s][k]["Lambda_TDE"] for s in seeds]) for k in key]; sd = [np.std([J[s][k]["Lambda_TDE"] for s in seeds], ddof=1) for k in key]
ax[1].bar(range(8), ab, yerr=sd, capsize=3, color=["#2a78d6"] * 4 + ["#eb6834"] * 4, width=.7); ax[1].set_xticks(range(8)); ax[1].set_xticklabels(lab, fontsize=8)
ax[1].set_ylabel("absolute expected TDEs"); ax[1].set_title("Sensitivity: hazard-grid resolution and κ", loc="left"); ax[1].grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(f"{PL}/paired_summary.png", dpi=150); plt.close(fig); print("\nsaved plots + csv")
