"""Plots of merger-rate models and their hazard consequences. Output: ../plots/"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from merger_models import make_models
from cosmology import CosmologyModel

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")
os.makedirs(OUT, exist_ok=True)

# validated categorical slots 1-6 (dataviz palette), fixed order; line style = secondary encoding
STYLE = {
    "baseline":     ("#2a78d6", "-",  2.6),
    "mundy17":      ("#eb6834", "-",  2.0),
    "mundy17_n5e4": ("#1baf7a", "--", 2.0),
    "oleary_9.7":   ("#eda100", "-",  2.0),
    "oleary_10.3":  ("#e87ba4", "--", 2.0),
    "oleary_11.0":  ("#008300", ":",  2.2),
    "duan24_pl":    ("#4a3aa7", "-",  2.0),
    "duan24_plexp": ("#e34948", "--", 2.2),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 11, "legend.frameon": False})

cosmo = CosmologyModel()
M = make_models(cosmo)
z = np.linspace(0, 10, 500)


def lines(ax, f, **kw):
    for k, m in M.items():
        if k not in STYLE: continue
        c, ls, lw = STYLE[k]
        ax.plot(z, f(m, k), color=c, ls=ls, lw=lw, label=m.label, **kw)


fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
lines(ax[0], lambda m, k: m.lambda_MM(z)); ax[0].set_yscale("log")
ax[0].set_xlabel("redshift z"); ax[0].set_ylabel(r"major-merger rate $\lambda(z)$ [Gyr$^{-1}$ galaxy$^{-1}$]")
ax[0].set_title("Merger rate (per galaxy)", loc="left"); ax[0].axvspan(3.5, 10, color="#898781", alpha=0.08, lw=0)
ax[0].text(3.6, ax[0].get_ylim()[0] * 1.4, "beyond Mundy+17 data (z<3.5)", color=MUTED, fontsize=9)
ax[0].legend(fontsize=8.5, loc="upper left")
r = lambda m, k: m.lambda_MM(z) / M["baseline"].lambda_MM(z)
lines(ax[1], r); ax[1].set_yscale("log"); ax[1].axhline(1, color=MUTED, lw=0.8)
ax[1].set_xlabel("redshift z"); ax[1].set_ylabel("rate / baseline rate")
ax[1].set_title("Suppression relative to (1+z)$^2$ baseline", loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/merger_rates_vs_z.png", dpi=150); plt.close(fig)

# survival function from z_obs (probability of NO major merger between z_obs and z)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for a, z0 in zip(ax, (0.1, 1.0)):
    zg = np.linspace(z0, 12, 600)
    for k, m in M.items():
        if k not in STYLE: continue
        c, ls, lw = STYLE[k]
        g = m.hazard_integrand(zg); Lam = np.concatenate([[0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(zg))])
        a.plot(zg, np.exp(-Lam), color=c, ls=ls, lw=lw, label=m.label)
    a.set_xlabel("redshift z (looking back from observation)"); a.set_ylabel("S = P(no major merger since z$_{obs}$)")
    a.set_title(f"Survival, galaxy observed at z$_{{obs}}$={z0}", loc="left"); a.set_ylim(0, 1.02)
ax[0].legend(fontsize=8.5, loc="lower left")
fig.tight_layout(); fig.savefig(f"{OUT}/survival_functions.png", dpi=150); plt.close(fig)

# LMM lookback distributions (sampled), z_obs = 0.1 and 1.0; censored (no merger < z=12) shown as text
np.random.seed(7)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for a, z0 in zip(ax, (0.1, 1.0)):
    bins = np.linspace(z0, 12, 60)
    txt = []
    for k, m in M.items():
        if k not in STYLE: continue
        c, ls, lw = STYLE[k]
        zs = m.sample_z_LMM(np.full(20000, z0))
        cens = np.mean(zs >= m.z_max - 1e-9)
        h, e = np.histogram(zs[zs < m.z_max - 1e-9], bins=bins, weights=np.full((zs < m.z_max - 1e-9).sum(), 1 / len(zs)))
        a.step(0.5 * (e[1:] + e[:-1]), h, where="mid", color=c, ls=ls, lw=lw, label=f"{m.label}  (no merger: {cens:.0%})")
    a.set_xlabel("$z_{LMM}$"); a.set_ylabel("fraction of galaxies per bin"); a.set_yscale("log")
    a.set_title(f"Last-major-merger redshift, z$_{{obs}}$={z0}", loc="left")
ax[0].legend(fontsize=7.8, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/zLMM_distributions.png", dpi=150); plt.close(fig)
print("saved to", os.path.abspath(OUT))
