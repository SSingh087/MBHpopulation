"""Plots for the time-averaging test: rate curves, cumulative counts, ratios and redshift distributions."""
# Plots for the time-averaging question. Run from repo root.
#  fig1: rate curves and cumulative number of events vs cusp age (shape only, peak-normalised)
#  fig2: does averaging change TDE/EMRI? ratio inst. vs averaged, and boost vs cusp age
#  fig3: population level: cumulative redshift distribution of expected events, time-dependent vs time-averaged
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, 'astrophysical_setup')
from rate import UniversalRate as U
from scipy.integrate import cumulative_trapezoid as cumtrapz

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False})
OUT = "tests/TEST_time_average/plots/"
tau = np.linspace(1e-4, 30, 6000); fE, fT = U.EMRI_rate(tau), U.TDE_rate(tau)
TF = 10.0   # t_f = 10 t_E (as in ConstantRate)

# ---------- fig 1
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
a = ax[0]
a.plot(tau, fE, color=BLUE, lw=2); a.plot(tau, fT, color=ORANGE, lw=2)
for f, c in [(fE, BLUE), (fT, ORANGE)]:
    m = np.trapz(f[tau <= TF], tau[tau <= TF]) / TF
    a.hlines(m, 0, TF, color=c, ls="--", lw=1.4); 
a.axvline(TF, color=MUTED, lw=1, ls=":"); a.text(TF + .3, 1.55, r"$t_f=10\,t_E$", color=MUTED, fontsize=10)
a.text(20, 0.02, "EMRI", color=BLUE, fontweight="bold"); a.text(20, 0.22, "TDE", color=ORANGE, fontweight="bold")
a.text(0.4, 0.5, "dashed = time average\nover $0$–$t_f$", color=MUTED, fontsize=10, va="top") if False else None
a.set_xlabel(r"cusp age  $\tau = T_c / t_E$"); a.set_ylabel(r"rate  $\Gamma/\hat\Gamma$"); a.set_ylim(0, 1.7); a.set_xlim(0, 30)
a.set_title("Rate vs cusp age (dashed: time average to $t_f$)", loc="left", fontsize=11.5)
b = ax[1]
CE, CT = cumtrapz(fE, tau, initial=0), cumtrapz(fT, tau, initial=0)
iF = np.searchsorted(tau, TF)
b.plot(tau, CE / CE[iF], color=BLUE, lw=2); b.plot(tau, CT / CT[iF], color=ORANGE, lw=2)
b.axvline(TF, color=MUTED, lw=1, ls=":"); b.axhline(1, color=MUTED, lw=1, ls=":")
for t in (1, 3):
    b.plot([t], [CT[np.searchsorted(tau, t)] / CT[iF]], "o", color=ORANGE, ms=6); b.plot([t], [CE[np.searchsorted(tau, t)] / CE[iF]], "o", color=BLUE, ms=6)
b.text(1.6, .12, "TDE events arrive earlier:\nat $\\tau=1$ "+f"{CT[np.searchsorted(tau,1)]/CT[iF]*100:.0f}% of the $t_f$ total (EMRI {CE[np.searchsorted(tau,1)]/CE[iF]*100:.0f}%)", color=INK, fontsize=10)
b.set_xlim(0, 15); b.set_ylim(0, 1.12); b.set_xlabel(r"cusp age  $\tau = T_c / t_E$"); b.set_ylabel(r"cumulative events, fraction of $N(<t_f)$")
b.set_title("Cumulative number of events", loc="left", fontsize=11.5)
fig.tight_layout(); fig.savefig(OUT + "fig1_curves_and_cumulative.png", dpi=170); plt.close(fig)

# ---------- fig 2
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
def avg_ratio(tf):
    """Ratio of instantaneous to time-averaged counts for cusp age tf."""
    m = tau <= tf; return np.trapz(fT[m], tau[m]) / np.trapz(fE[m], tau[m])
tt = np.linspace(0.1, 15, 400); inst = U.TDE_rate(tt) / U.EMRI_rate(tt)
a = ax[0]; a.plot(tt, inst, color=INK, lw=2); a.set_yscale("log")
a.axhspan(avg_ratio(3), avg_ratio(30), color=MUTED, alpha=.18, lw=0); a.axhline(avg_ratio(TF), color=MUTED, ls="--", lw=1.5)
a.text(15, avg_ratio(TF) * 0.83, "time-averaged ($t_f$ = 3–30 $t_E$)", color=MUTED, fontsize=10, ha="right", va="top")
a.text(0.5, inst[0] * .8, "instantaneous", color=INK, fontsize=10, fontweight="bold")
a.set_xlabel(r"cusp age  $\tau$"); a.set_ylabel(r"$\Gamma_{TDE}/\Gamma_{EMRI}$  (units of $\hat\Gamma_T/\hat\Gamma_E$)")
a.set_yticks([0.7, 1, 2, 4]); a.set_yticklabels(["0.7", "1", "2", "4"]); a.minorticks_off(); a.set_ylim(0.6, 5.5)
a.set_title("Ratio at a given age vs averaged over the history", loc="left", fontsize=11.5)
b = ax[1]
for tf, ls in [(3, ":"), (10, "-"), (30, "--")]:
    b.plot(tt, avg_ratio(tf) / inst, color=INK, ls=ls, lw=2, label=fr"$t_f={tf}\,t_E$")
b.axhline(1, color=MUTED, lw=1); b.axhspan(1/1.5, 1.5, color=MUTED, alpha=.10, lw=0)
b.set_xlabel(r"cusp age  $\tau$"); b.set_ylabel("boost of TDE/EMRI from averaging"); b.set_ylim(.3, 1.7); b.legend(loc="upper right")
b.set_title("Shape-only boost from averaging: within ~1.5×", loc="left", fontsize=11.5)
fig.tight_layout(); fig.savefig(OUT + "fig2_ratio_and_boost.png", dpi=170); plt.close(fig)

# ---------- fig 3
d = np.load("tests/TEST_time_average/time_avg_200k.npz"); z, w = d["z"], d["w"]
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
def cdf(lam):
    """Cumulative redshift distribution for rate lam."""
    W = w * lam; o = np.argsort(z); c = np.cumsum(W[o]); return z[o], c / c[-1]
for a, kind, col in [(ax[0], "TDE", ORANGE), (ax[1], "EMRI", BLUE)]:
    for key, ls, lab in [("lam_td", "-", "time-dependent"), ("lam_avg", "--", "time-averaged")]:
        x, c = cdf(d[f"{key}_{kind}"]); a.plot(x, c, color=col, ls=ls, lw=2, label=lab)
        a.plot([x[np.searchsorted(c, .5)]], [.5], "o", color=col, ms=6)
    a.set_xlim(0, 6); a.set_ylim(0, 1.02); a.set_xlabel("redshift z"); a.set_ylabel("cumulative fraction of expected events")
    a.set_title(f"{kind}: where the events are (normalised, shape only)", loc="left", fontsize=11.5); a.legend(loc="lower right")
fig.tight_layout(); fig.savefig(OUT + "fig3_redshift_cumulative.png", dpi=170); plt.close(fig)
print("ok")

# ---------- fig 4: is the shift to high z caused by the T_c < t_rlx cutoff?
d = np.load("tests/TEST_time_average/time_avg_200k.npz"); z, w, lgM = d["z"], d["w"], d["lgMBH"]
GREY = "#898781"
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
for a, kind, col in [(ax[0], "TDE", ORANGE), (ax[1], "EMRI", BLUE)]:
    for key, ls, lw, c, lab in [("lam_td", "-", 2, col, "time-dependent"), ("lam_avg", "--", 2, col, "time-averaged, with $T_c<t_{rlx}$ cut"),
                                ("lam_avg_nocut", ":", 2, GREY, "time-averaged, no cut")]:
        x, c_ = cdf(d[f"{key}_{kind}"]); a.plot(x, c_, color=c, ls=ls, lw=lw, label=lab)
    a.set_xlim(0, 6); a.set_ylim(0, 1.02); a.set_xlabel("redshift z"); a.set_ylabel("cumulative fraction of expected events")
    a.set_title(f"{kind}: effect of the top-hat cutoff (shape only)", loc="left", fontsize=11.5); a.legend(loc="lower right", fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT + "fig4_cutoff_test.png", dpi=170); plt.close(fig)

# ---------- fig 5: median redshift of expected events by black-hole mass
edges = [4, 5, 6, 7]; cen = [4.5, 5.5, 6.5]
def wmed(x, W):
    """Weighted median."""
    o = np.argsort(x); c = np.cumsum(W[o]); return x[o][np.searchsorted(c, .5 * c[-1])] if c[-1] > 0 else np.nan
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
for a, kind, col in [(ax[0], "TDE", ORANGE), (ax[1], "EMRI", BLUE)]:
    for key, mk, ls, lab in [("lam_td", "o", "-", "time-dependent"), ("lam_avg", "s", "--", "time-averaged")]:
        y = [wmed(z[(lgM > lo) & (lgM <= hi)], (w * d[f"{key}_{kind}"])[(lgM > lo) & (lgM <= hi)]) for lo, hi in zip(edges[:-1], edges[1:])]
        a.plot(cen, y, color=col, ls=ls, marker=mk, ms=7, lw=2, label=lab)
    a.set_xticks(cen); a.set_xticklabels(["$10^4$–$10^5$", "$10^5$–$10^6$", "$10^6$–$10^7$"]); a.set_xlabel(r"$M_{BH}$ [M$_\odot$]")
    a.set_ylabel("median redshift of expected events"); a.set_ylim(0, None); a.legend(loc="lower right")
    a.set_title(f"{kind}: shift with black-hole mass", loc="left", fontsize=11.5)
fig.tight_layout(); fig.savefig(OUT + "fig5_by_mass.png", dpi=170); plt.close(fig)

# ---------- fig 6: how well do the by-eye fitted curves reproduce Broggi+22 Table? (fit / table, 1 = perfect)
from scipy.integrate import quad
from rate import ConstantRate as C
from config import BROGGI_SIGMA0, BROGGI_MBH_PIVOT, BROGGI_INV_SLOPE
MB = 10 ** C._log10_MBH_grid; t_f = np.minimum(10 * C._t_E, 10.0)
R = {k: np.zeros((7, 4)) for k in ("t_E", "N_E", "N_T")}
for i, M in enumerate(MB):
    for j, r in enumerate(C._sigma_ratio_grid):
        GE = 6.2e-6 * M ** -0.25 * r ** 3.09; GT = 1.6e-4 * M ** -0.22 * r ** 1.16; tE = 6.4e-9 * M ** 1.29 * r ** -2.97
        tf = t_f[i, j] / tE
        R["t_E"][i, j] = tE / C._t_E[i, j]
        R["N_E"][i, j] = GE * tE * 1e9 * quad(lambda x: float(U.EMRI_rate(x)), 0, tf, limit=200)[0] / C._N_E[i, j]
        R["N_T"][i, j] = GT * tE * 1e9 * quad(lambda x: float(U.TDE_rate(x)), 0, tf, limit=200)[0] / C._N_T[i, j]
fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.3))
for a, (k, lab) in zip(ax, [("t_E", "$t_E$ (fit / table)"), ("N_E", "$N_E$ (fit / table)"), ("N_T", "$N_T$ (fit / table)")]):
    im = a.imshow(np.log10(R[k]), origin="lower", cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    for i in range(7):
        for j in range(4): a.text(j, i, f"{R[k][i,j]:.2g}", ha="center", va="center", fontsize=8.5, color=INK)
    a.set_xticks(range(4)); a.set_xticklabels(C._sigma_ratio_grid); a.set_yticks(range(7)); a.set_yticklabels([f"$10^{{{int(np.log10(m))}}}$" if np.log10(m).is_integer() else f"{m:.0e}" for m in MB])
    a.set_xlabel(r"$\sigma/\sigma_0$"); a.set_ylabel(r"$M_{BH}$ [M$_\odot$]"); a.set_title(lab, loc="left", fontsize=11.5); a.grid(False)
fig.tight_layout(); fig.savefig(OUT + "fig6_fit_vs_table.png", dpi=170); plt.close(fig)
print("ok2")
