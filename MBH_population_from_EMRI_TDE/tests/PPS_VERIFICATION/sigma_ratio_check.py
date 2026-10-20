#!/usr/bin/env python
"""
Checks how galaxy velocity dispersion (sigma_km_s) and the derived
sigma/sigma_broggi(MBH) ratio -- the quantity ConstantRate.constant_rate()
uses to pick a bin on Broggi et al. 2022's {0.01,0.1,1,10} rho_0 density
grid -- actually vary across the real, already-generated universes in
this repo's DAG output (data_generation/sources_num.py's
galaxy_catalog_data_*.h5 files, one per (mbh_mass_method, nucleation_method,
occupation_method, UNIVERSE_MODEL) combination).

Reads sigma_km_s and lgMBH straight from the resampled EMRI+TDE event
catalogs (the same values sources_num.py wrote), recomputes
broggi_sigma0(MBH) with the repo's own constants (config.py), and plots:

  1. sigma_km_s distribution, one curve per universe file (colored by
     scenario), to see how much sigma actually varies run to run.
  2. sigma / sigma_broggi(MBH) distribution, with the calibration grid's
     4 levels {0.464, 0.681, 1.0, 1.468} marked -- to see where the real
     simulated population actually lands relative to Broggi's grid.

Usage:
    python tests/PPS_VERIFICATION/sigma_ratio_check.py
"""
import os, sys, glob
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('./astrophysical_setup'))
sys.path.insert(0, os.path.abspath('./'))
from utils import ASTROPHYSICAL_DATA_DIR, MOCK_GALAXY_CATALOG_DATA
from config import BROGGI_SIGMA0, BROGGI_MBH_PIVOT, BROGGI_INV_SLOPE
from galaxy import Galaxy
from nsc import NSC
from rate import ConstantRate

RATE_LO, RATE_HI = 1e-5, 1e-4  # target range, events/galaxy/year (same as tde_rate_check.py)

GALAXIES = 100000  # matches EXECUTABLES/config.sh's GALAXIES, i.e. the DAG output directory

SIGMA_RATIO_GRID = np.array([0.464, 0.681, 1.0, 1.468])  # ConstantRate's own grid (rate.py)


def broggi_sigma0(MBH):
    return BROGGI_SIGMA0 * (MBH / BROGGI_MBH_PIVOT) ** BROGGI_INV_SLOPE


pattern = f"{ASTROPHYSICAL_DATA_DIR}/{GALAXIES}/{MOCK_GALAXY_CATALOG_DATA}_*.h5"
files = sorted(glob.glob(pattern))
print(f"Found {len(files)} universe catalog files matching {pattern}")

if not files:
    raise RuntimeError(f"No files found at {pattern} -- check ASTROPHYSICAL_DATA_DIR/GALAXIES.")

# ---- collect sigma_km_s, lgMBH, and a scenario label per file -------------
per_file = []
for fpath in files:
    base = os.path.basename(fpath)
    # strip prefix/suffix to get the "<mbh_method>_<nucleation>_<occupation>_<universe>" tag
    tag = base[len(MOCK_GALAXY_CATALOG_DATA) + 1: -len(".h5")]
    scenario = "_".join(tag.split("_")[:-1])  # drop trailing universe index

    with h5py.File(fpath, "r") as f:
        sigma_list, lgMBH_list = [], []
        for group in ("EMRI_catalog", "TDE_catalog"):
            if group in f and len(f[group]["sigma_km_s"]) > 0:
                sigma_list.append(f[group]["sigma_km_s"][:])
                lgMBH_list.append(f[group]["lgMBH"][:])
        if sigma_list:
            sigma = np.concatenate(sigma_list)
            lgMBH = np.concatenate(lgMBH_list)
        else:
            sigma = np.array([])
            lgMBH = np.array([])

    per_file.append((scenario, base, sigma, lgMBH))

scenarios = sorted(set(s for s, _, _, _ in per_file))
cmap = plt.get_cmap("tab10")
scenario_color = {s: cmap(i % 10) for i, s in enumerate(scenarios)}

print(f"Scenarios: {scenarios}")

# ---- Plot 1: sigma_km_s distribution, one curve per universe file ---------
fig, ax = plt.subplots(figsize=(8, 6))
bins_sigma = np.logspace(0, 3.5, 60)
seen_scenario = set()
for scenario, base, sigma, lgMBH in per_file:
    if sigma.size == 0:
        continue
    label = scenario if scenario not in seen_scenario else None
    seen_scenario.add(scenario)
    counts, edges = np.histogram(sigma, bins=bins_sigma)
    centers = 0.5 * (edges[1:] + edges[:-1])
    ax.plot(centers, counts, color=scenario_color[scenario], alpha=0.35, lw=1, label=label)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\sigma$ [km/s]")
ax.set_ylabel("count per universe (event catalog)")
ax.set_title(f"$\\sigma_{{\\rm km/s}}$ across all {len(files)} universes\n"
             f"(EMRI+TDE resampled event catalogs, {GALAXIES} MC galaxies each)")
ax.legend(fontsize=6, ncol=2)
plt.tight_layout()
out1 = "tests/PPS_VERIFICATION/sigma_distribution_all_universes.png"
plt.savefig(out1, dpi=150)
print(f"Saved {out1}")
plt.close(fig)

# ---- Plot 2: sigma / sigma_broggi(MBH) distribution ------------------------
all_ratios = []
all_lgMBH = []
fig, ax = plt.subplots(figsize=(8, 6))
bins_ratio = np.logspace(np.log10(0.05), np.log10(20), 80)
seen_scenario = set()
for scenario, base, sigma, lgMBH in per_file:
    if sigma.size == 0:
        continue
    MBH = 10.0 ** lgMBH
    ratio = sigma / broggi_sigma0(MBH)
    all_ratios.append(ratio)
    all_lgMBH.append(lgMBH)

    label = scenario if scenario not in seen_scenario else None
    seen_scenario.add(scenario)
    counts, edges = np.histogram(ratio, bins=bins_ratio)
    centers = 0.5 * (edges[1:] + edges[:-1])
    ax.plot(centers, counts, color=scenario_color[scenario], alpha=0.35, lw=1, label=label)

for level in SIGMA_RATIO_GRID:
    ax.axvline(level, color='k', ls='--', lw=1, alpha=0.7)
    ax.text(level, ax.get_ylim()[1], f"{level}", rotation=90, va='top', ha='right', fontsize=7)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\sigma / \sigma_{\rm broggi}(M_{\rm BH})$")
ax.set_ylabel("count per universe (event catalog)")
ax.set_title("Where simulated galaxies land on Broggi's {0.01,0.1,1,10}"
             r"$\rho_0$ grid (dashed lines = grid's $\sigma/\sigma_0$ levels)")
ax.legend(fontsize=6, ncol=2, loc='upper left')

# ---- overlay: intrinsic TDE rate vs sigma_ratio at 4 representative MBH values ----
# (rate depends on BOTH MBH and sigma_ratio -- one curve per MBH, matching
# Broggi's own grid endpoints/decades, mirrors tde_rate_check.py's approach
# of showing one curve per sigma_ratio across MBH, just transposed.)
REPRESENTATIVE_MBH = [1e4, 1e5, 1e6, 1e7]
rate_colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']

ratio_grid = np.logspace(np.log10(0.05), np.log10(20), 200)

ax2 = ax.twinx()
for MBH_rep, rcolor in zip(REPRESENTATIVE_MBH, rate_colors):
    lgMBH_rep = np.log10(MBH_rep)
    sigma_km_s_grid = ratio_grid * broggi_sigma0(MBH_rep)

    gal_r = Galaxy(lgMBH=np.full_like(ratio_grid, lgMBH_rep),
                   sigma_pc_yr=sigma_km_s_grid * 1e5 * 3.154e7 / 3.085677581491367e18,
                   z_gal=np.zeros_like(ratio_grid), mbh_mass_method='use_sigma')
    nsc_r = NSC(gal_r)
    cr_r = ConstantRate(nsc=nsc_r)
    rate_per_yr = cr_r.constant_rate(kind='TDE') / 1e9

    ax2.plot(ratio_grid, rate_per_yr, color=rcolor, lw=2, ls='-',
              label=fr"rate at $M_{{\rm BH}}$={MBH_rep:.0e} $M_\odot$")

ax2.axhspan(RATE_LO, RATE_HI, color='grey', alpha=0.15,
             label=f"target [{RATE_LO:.0e},{RATE_HI:.0e}] /gal/yr")
ax2.set_yscale("log")
ax2.set_ylabel("TDE rate [events/galaxy/year] (solid curves)")
ax2.legend(fontsize=6, loc='upper right')

plt.tight_layout()
out2 = "tests/PPS_VERIFICATION/sigma_ratio_vs_broggi_grid.png"
plt.savefig(out2, dpi=150)
print(f"Saved {out2}")
print(f"(rate overlay computed at MBH = {REPRESENTATIVE_MBH} Msun)")
plt.close(fig)

# ---- Plot 3: tde_rate_check.py's own reference curves (rate vs MBH, one
#      curve per sigma_ratio grid level), with the REAL simulated population's
#      actual (MBH, rate) scattered on top -- ties the two diagnostics together. ----
rng = np.random.default_rng(0)
N_SUBSAMPLE = 200_000
sigma_all = np.concatenate([sigma for _, _, sigma, _ in per_file if sigma.size > 0])
lgMBH_all_pts = np.concatenate([lgMBH for _, _, sigma, lgMBH in per_file if sigma.size > 0])

idx = rng.choice(len(sigma_all), size=min(N_SUBSAMPLE, len(sigma_all)), replace=False)
sigma_sub = sigma_all[idx]
lgMBH_sub = lgMBH_all_pts[idx]
MBH_sub = 10.0 ** lgMBH_sub

gal_sub = Galaxy(lgMBH=lgMBH_sub, sigma_pc_yr=sigma_sub * 1e5 * 3.154e7 / 3.085677581491367e18,
                  z_gal=np.zeros_like(lgMBH_sub), mbh_mass_method='use_sigma')
nsc_sub = NSC(gal_sub)
cr_sub = ConstantRate(nsc=nsc_sub)
rate_sub_per_yr = cr_sub.constant_rate(kind='TDE') / 1e9

fig3, ax3 = plt.subplots(figsize=(8, 6))

# tde_rate_check.py's own 4 reference curves (rate vs MBH, one per sigma_ratio grid level)
MBH_grid_ref = np.logspace(4.0, 7.0, 25)
for sr, rcolor in zip(SIGMA_RATIO_GRID, ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']):
    rates_ref = []
    for MBH_g in MBH_grid_ref:
        lgMBH_g = np.log10(MBH_g)
        sigma_km_s_g = sr * broggi_sigma0(MBH_g)
        gal_g = Galaxy(lgMBH=np.array([lgMBH_g]),
                        sigma_pc_yr=np.array([sigma_km_s_g * 1e5 * 3.154e7 / 3.085677581491367e18]),
                        z_gal=np.array([0.0]), mbh_mass_method='use_sigma')
        nsc_g = NSC(gal_g)
        cr_g = ConstantRate(nsc=nsc_g)
        rates_ref.append(cr_g.constant_rate(kind='TDE')[0] / 1e9)
    ax3.plot(MBH_grid_ref, rates_ref, color=rcolor, lw=2, marker='o', ms=3,
              label=fr"$\sigma/\sigma_0={sr}$ (reference grid)")

# real simulated population scattered on top
sc = ax3.scatter(MBH_sub, rate_sub_per_yr, c=sigma_sub / broggi_sigma0(MBH_sub),
                   cmap='viridis', s=2, alpha=0.15, norm=matplotlib.colors.LogNorm(vmin=0.1, vmax=5),
                   label=f'real population ({len(idx):,} sampled events)')
cbar = plt.colorbar(sc, ax=ax3)
cbar.set_label(r"$\sigma/\sigma_{\rm broggi}(M_{\rm BH})$ of that event")

ax3.axhspan(RATE_LO, RATE_HI, color='grey', alpha=0.2, label=f"target [{RATE_LO:.0e},{RATE_HI:.0e}] /gal/yr")
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_xlabel(r"$M_{\rm BH}$ [$M_\odot$]")
ax3.set_ylabel("TDE rate [events/galaxy/year]")
ax3.legend(fontsize=7, loc='upper right')
plt.tight_layout()
out3 = "tests/PPS_VERIFICATION/rate_check_with_real_population.png"
plt.savefig(out3, dpi=150)
print(f"Saved {out3}")
plt.close(fig3)

# ---- Summary stats ----------------------------------------------------------
all_ratios = np.concatenate(all_ratios)
print(f"\nsigma/sigma_broggi(MBH) over {len(all_ratios)} events, all universes combined:")
print(f"  min={all_ratios.min():.3f}  5%={np.percentile(all_ratios,5):.3f}  "
      f"median={np.median(all_ratios):.3f}  95%={np.percentile(all_ratios,95):.3f}  max={all_ratios.max():.3f}")

frac_near_1 = np.mean((all_ratios > 0.7) & (all_ratios < 1.5))
frac_underdense = np.mean(all_ratios <= 0.7)
print(f"  fraction landing near rho_0 or denser (ratio in [0.7,1.5]): {frac_near_1:.1%}")
print(f"  fraction landing in the underdense (correctly-calibrated) bins (ratio <= 0.7): {frac_underdense:.1%}")
