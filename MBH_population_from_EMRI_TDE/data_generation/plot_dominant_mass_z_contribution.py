#!/usr/bin/env python
"""
Which (galaxy stellar mass, redshift) region actually dominates the
simulated EMRI/TDE source counts, AFTER nucleation, MBH occupation, and
the merger-rate-driven "active cusp" cut?

This reuses the exact production weighting pipeline from
data_generation/sources_num.py (same GSMF importance sampling, same
Galaxy/NSC/CuspEvolution physics, same combined_EMRI_weight /
combined_TDE_weight construction -- see that file for the full derivation)
but instead of collapsing the per-galaxy weights to a single scalar count,
bins them on a 2D (z, lgMgal) grid and plots the result as a filled
contour, so you can read off which mass/redshift region contributes most.

"After nucleation and rate cut" here means: after
  (1) nucleation (Galaxy.check_nucleation),
  (2) MBH occupation + the calibrated lgMBH in [4,7] mass window,
  (3) the merger-rate-derived "active" mask (cusp_age < t_relax, which
      depends on the sampled last-major-merger time under the chosen
      --merger_model),
exactly as sources_num.py applies them before PPS-resampling the final
event catalog.

Usage:
    python data_generation/plot_dominant_mass_z_contribution.py \\
        --merger_model baseline --mbh_mass_method use_lgMgal \\
        --nucleation_method hannah --occupation_method uniform   # numeric summary only

    # make the M1-M8 scenario-grid figures (EMRI + TDE, all 4 merger models):
    python data_generation/plot_dominant_mass_z_contribution.py --scenario_grid [--grid_models oleary21]

Plots are made with --scenario_grid and saved (PNG + PDF) in the shared PLOTS_DIR
defined in plots/paper/plot_style.py; without it the script only prints
a numeric summary.
"""
import os
import sys

sys.path.insert(0, os.path.abspath("./astrophysical_setup"))
sys.path.insert(0, os.path.abspath("./"))
sys.path.insert(0, os.path.abspath("./plots/paper"))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from utils import MERGER_MODEL_CHOICES
from plot_style import PLOTS_DIR, savefig_all, setup_style, fig_size, TEXT_W, SCEN_LABEL

setup_style()
from galaxy import Galaxy
from nsc import NSC, CompactObject, MBH_properties
from density import DehnenProfile
from relaxation import RelaxationModel
from rate import RateModel, ConstantRate
from evolution import CuspEvolution
from cosmology import LastMajorMerger, CosmologyModel, GalaxyStellarMassFunction
from scipy.interpolate import interp1d

# Stated validity range of each merger-rate fit, from
# data_generation/README_merger_models.md's model table + "Assumptions /
# caveats" section. `z_min`/`z_max`/`mass_min`/`mass_max` (log10 Msun) are
# omitted where the paper places no such bound. `baseline` isn't from a
# paper and has no stated validity range at all.
#
# mundy17's z_max=2.5 (not the paper's overall z<3.5 title range) is
# verified directly against Mundy+2017 Sec. 4.2, which states that the
# specific n=1e-4 Mpc^-3 selection used here (mass_min=11.2 at 0<z<0.2)
# traces its progenitors "out to a higher redshift of z=2.5" -- the
# paper's z<3.5 range applies to its other, mass-selected samples.
MERGER_MODEL_VALIDITY = {
    "baseline": {},
    "mundy17": {"z_max": 2.5, "mass_min": 11.2},
    "oleary21": {"z_max": 6.0, "mass_min": 9.7},
    "duan24": {"z_min": 4.5, "z_max": 11.5, "mass_min": 8.0, "mass_max": 10.0},
}

# Same 8 (mbh_mass_method, nucleation_method, occupation_method) scenarios as
# MODEL_NAME_MAP in plots/paper/plot_predict_EMRI_counts.py.
SCENARIOS_M1_M8 = [
    ("M1", "use_lgMgal", "hannah",   "uniform"),
    ("M2", "use_sigma",  "hannah",   "uniform"),
    ("M3", "use_lgMgal", "neumayer", "uniform"),
    ("M4", "use_sigma",  "neumayer", "uniform"),
    ("M5", "use_lgMgal", "hannah",   "varying"),
    ("M6", "use_sigma",  "hannah",   "varying"),
    ("M7", "use_lgMgal", "neumayer", "varying"),
    ("M8", "use_sigma",  "neumayer", "varying"),
]


def run_weighted_sample(merger_model, mbh_mass_method, nucleation_method, occupation_method,
                         N_objs_sim=200_000, Z_MAX=8.0, T_OBS_EMRI=6.0, T_OBS_TDE=17.25,
                         seed=42):
    """
    Reproduces sources_num.py's weighting pipeline (same GSMF importance
    sampling + nucleation/occupation/active-cusp cuts) for one
    (merger_model, scenario) combination, returning the per-sampled-galaxy
    (z, lgMgal, combined_EMRI_weight, combined_TDE_weight) arrays -- i.e.
    each nucleated sample's contribution to the final EMRI/TDE counts,
    AFTER nucleation and the merger-rate-driven active-cusp cut.
    """
    rng = np.random.default_rng(seed=seed)
    cosmo_model = CosmologyModel()

    Z_MIN = 1e-5
    T_obs_EMRI_Gyr = T_OBS_EMRI / 1e9
    T_obs_TDE_Gyr = T_OBS_TDE / 1e9

    Rmin = cosmo_model.comoving_distance(Z_MIN)
    Rmax = cosmo_model.comoving_distance(Z_MAX)

    GSMF = GalaxyStellarMassFunction()

    def build_density_tables(gsmf, z_min, z_max, n_z=600, n_points_mass=5000):
        z_table = np.linspace(z_min, z_max, n_z)
        lgM_grid, log10phi_grid = gsmf.get_gsmf(z_table, n_points_mass=n_points_mass)
        phi_grid = np.nan_to_num(10.0 ** log10phi_grid, nan=0.0, posinf=0.0, neginf=0.0)
        n_density_table = np.trapz(phi_grid, lgM_grid, axis=1)
        n_density_interp = interp1d(z_table, n_density_table, kind='linear',
                                     bounds_error=False,
                                     fill_value=(n_density_table[0], n_density_table[-1]))
        return z_table, n_density_table, n_density_interp

    def find_lgMgal_rate_range(lgMBH_min=4.0, lgMBH_max=7.0, N_pilot=100_000):
        lgM_min_full = GSMF.lgMgal_data.min()
        lgM_max_full = GSMF.lgMgal_data.max()

        u_p = rng.random(N_pilot)
        r_p = (Rmin ** 3 + (Rmax ** 3 - Rmin ** 3) * u_p) ** (1 / 3)
        z_p = cosmo_model.z_from_comoving_distance_fast(r_p)

        lgMgal_p = rng.uniform(lgM_min_full, lgM_max_full, size=N_pilot)
        nuc_idx_p = Galaxy.check_nucleation(lgMgal_p, nucleation_method=nucleation_method)
        gal_p = Galaxy(lgMgal=lgMgal_p, z_gal=z_p, nucleation_occurs=nuc_idx_p,
                        mbh_mass_method=mbh_mass_method, nucleation_method=nucleation_method,
                        occupation_method=occupation_method)

        lgMBH_p = gal_p.lgMBH_mass
        lgMgal_nuc_p = lgMgal_p[nuc_idx_p]
        in_range = (lgMBH_p >= lgMBH_min) & (lgMBH_p <= lgMBH_max) & gal_p.occupation_occurs

        if in_range.sum() < 50:
            raise RuntimeError(
                f"Only {in_range.sum()} pilot galaxies in lgMBH [{lgMBH_min},{lgMBH_max}]. "
                f"Increase N_pilot (currently {N_pilot}).")

        lgM_low = np.percentile(lgMgal_nuc_p[in_range], 0.01)
        lgM_high = np.percentile(lgMgal_nuc_p[in_range], 99.99)
        return lgM_low, lgM_high

    z_table, n_density_table, _ = build_density_tables(GSMF, Z_MIN, Z_MAX)
    dVc_dz_table = cosmo_model.dVc_dz(z_table)

    u = rng.random(N_objs_sim)
    r_samples = (Rmin ** 3 + (Rmax ** 3 - Rmin ** 3) * u) ** (1 / 3)
    z_grid = cosmo_model.z_from_comoving_distance_fast(r_samples)

    lgM_low, lgM_high = find_lgMgal_rate_range()
    lgMgal_samples = rng.uniform(lgM_low, lgM_high, size=N_objs_sim)

    log10phi_pointwise = GSMF.interp(np.column_stack([z_grid, lgMgal_samples]))
    phi_pointwise = np.nan_to_num(10.0 ** log10phi_pointwise, nan=0.0, posinf=0.0, neginf=0.0)

    V_tot = np.trapz(dVc_dz_table, z_table)
    mc_weight_i = phi_pointwise * V_tot * (lgM_high - lgM_low) / N_objs_sim

    nucleation_indices = Galaxy.check_nucleation(lgMgal_samples, nucleation_method=nucleation_method)
    gal_obj = Galaxy(lgMgal=lgMgal_samples, z_gal=z_grid, nucleation_occurs=nucleation_indices,
                      mbh_mass_method=mbh_mass_method, nucleation_method=nucleation_method,
                      occupation_method=occupation_method)

    NSC_obj = NSC(gal_obj)
    MBH_properties(nsc=NSC_obj)

    CO_objs = CompactObject(nsc=NSC_obj, masses={'sBH': 10.0, 'star': 1.0},
                             total_mass_in_MBH_mass_units={'sBH': 0.2, 'star': 20.0},
                             types_CO=['sBH', 'star'], types_masses='same_mass', type_CO_limits=None)

    dehnen_obj = DehnenProfile(nsc=NSC_obj, compact_object=CO_objs)
    relax_obj = RelaxationModel(nsc=NSC_obj, compact_object=CO_objs, profile=dehnen_obj)
    rate_obj = RateModel(nsc=NSC_obj)
    const_rate_obj = ConstantRate(nsc=NSC_obj)

    cusp_evolution_object = CuspEvolution(
        nsc=NSC_obj, compact_object=CO_objs, relaxation=relax_obj, rate_model=rate_obj,
        LastMajorMerger=LastMajorMerger(cosmo_model, merger_model=merger_model))

    cusp_evolution_object.cusp_turn_on_time(kvir=1.0, force_resample=True)
    T_c_Gyr = np.atleast_1d(cusp_evolution_object.cusp_age(kvir=1.0))
    t_rlx_Gyr = relax_obj.t_relax_at_rinfl(kvir=1.0, unit='Gyr')
    active = (T_c_Gyr < t_rlx_Gyr).astype(float)

    MBH_MASK = (gal_obj.lgMBH_mass > 4.0) & (gal_obj.lgMBH_mass <= 7.0) & gal_obj.occupation_occurs

    rate_EMRI_Gyr = const_rate_obj.constant_rate(kind='EMRI')
    rate_TDE_Gyr = const_rate_obj.constant_rate(kind='TDE')

    z_gal_nuc = NSC_obj.gal.z_gal
    Gamma_EMRI_i = np.where(MBH_MASK, rate_EMRI_Gyr * T_obs_EMRI_Gyr * active / (1.0 + z_gal_nuc), 0.0)
    Gamma_TDE_i = np.where(MBH_MASK, rate_TDE_Gyr * T_obs_TDE_Gyr * active / (1.0 + z_gal_nuc), 0.0)

    mc_weight_i_nuc = mc_weight_i[nucleation_indices]
    lgMgal_nuc = lgMgal_samples[nucleation_indices]

    combined_EMRI_weight = mc_weight_i_nuc * Gamma_EMRI_i
    combined_TDE_weight = mc_weight_i_nuc * Gamma_TDE_i

    return z_gal_nuc, lgMgal_nuc, combined_EMRI_weight, combined_TDE_weight


def dominant_region_grid(z, lgMgal, weight, n_z_bins=60, n_mass_bins=60,
                          z_range=None, mass_range=None):
    """2D-histogram `weight` over (z, lgMgal), returning bin edges/centers
    and the summed-weight grid, shape (n_mass_bins, n_z_bins)."""
    if z_range is None:
        z_range = (z.min(), z.max())
    if mass_range is None:
        mass_range = (lgMgal.min(), lgMgal.max())

    H, z_edges, m_edges = np.histogram2d(
        z, lgMgal, bins=[n_z_bins, n_mass_bins],
        range=[z_range, mass_range], weights=weight)
    return z_edges, m_edges, H.T  # H.T so rows=mass, cols=z


def _plot_weight_panel(ax, z, lgMgal, weight, label, smooth_sigma=1.2, validity=None):
    """
    Bins+smooths+contours one (z, lgMgal, weight) sample onto `ax`, marks
    the peak bin, and (if `validity` is given) draws dashed v/h lines at
    the merger model's stated z/mass validity bounds (see
    MERGER_MODEL_VALIDITY) so the reader can see how much of the
    dominant-contribution region falls outside where the rate was
    actually fit.

    smooth_sigma: Gaussian-smoothing width (in grid cells) applied to the
    2D weight histogram before contouring/peak-finding. Raw per-bin sums
    are Monte-Carlo shot noise -- a handful of samples can land a single
    large Gamma draw in an otherwise sparse bin and fake a "peak" there.
    Smoothing turns the plot (and the marked peak) into a read of the
    dominant *region*, not a single noisy bin.
    """
    z_edges, m_edges, H = dominant_region_grid(z, lgMgal, weight)
    Z_c = 0.5 * (z_edges[:-1] + z_edges[1:])
    M_c = 0.5 * (m_edges[:-1] + m_edges[1:])

    H = np.nan_to_num(H, nan=0.0)
    H = H / H.sum() if H.sum() > 0 else H  # fraction of total count per (z, lgMgal) bin
    H_smooth = gaussian_filter(H, sigma=smooth_sigma)
    floor = 1e-4 * H_smooth.max() if H_smooth.max() > 0 else 0.0
    H_plot = np.where(H_smooth > floor, H_smooth, np.nan)

    peak_idx = np.unravel_index(np.nanargmax(H_plot), H_plot.shape)
    z_peak, m_peak = Z_c[peak_idx[1]], M_c[peak_idx[0]]

    finite = H_plot[np.isfinite(H_plot)]
    use_lognorm = finite.max() / finite.min() > 50 if finite.size else False
    cf = ax.contourf(Z_c, M_c, H_plot, levels=20, cmap="viridis",
                      norm=plt.matplotlib.colors.LogNorm() if use_lognorm else None)
    ax.contour(Z_c, M_c, H_plot, levels=6, colors="white", linewidths=0.25, alpha=0.5)
    ax.plot(z_peak, m_peak, marker="*", color="crimson", markersize=6,
            markeredgecolor="black", markeredgewidth=0.5, zorder=5)

    if validity:
        xlo, xhi = z_edges[0], z_edges[-1]
        ylo, yhi = m_edges[0], m_edges[-1]
        shade_kwargs = dict(facecolor="none", edgecolor="red", hatch="//", alpha=0.45, zorder=1, lw=0)

        z_lo = validity.get("z_min", xlo)
        z_hi = validity.get("z_max", xhi)
        if "z_min" in validity:
            ax.axvspan(xlo, z_lo, **shade_kwargs)
            ax.axvline(z_lo, color="red", ls="--", lw=0.9, zorder=6)
        if "z_max" in validity:
            ax.axvspan(z_hi, xhi, **shade_kwargs)
            ax.axvline(z_hi, color="red", ls="--", lw=0.9, zorder=6)

        m_lo = validity.get("mass_min", ylo)
        m_hi = validity.get("mass_max", yhi)
        if "mass_min" in validity:
            ax.axhspan(ylo, m_lo, **shade_kwargs)
            ax.axhline(m_lo, color="red", ls="--", lw=0.9, zorder=6)
        if "mass_max" in validity:
            ax.axhspan(m_hi, yhi, **shade_kwargs)
            ax.axhline(m_hi, color="red", ls="--", lw=0.9, zorder=6)

        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)

    return cf, z_peak, m_peak


import matplotlib as _mpl
_mpl.rcParams['hatch.linewidth'] = 0.4


def plot_scenario_grid(merger_model, results_by_scenario, weight_idx, label, out_path,
                        smooth_sigma=1.2):
    """
    results_by_scenario: dict {scenario_name (M1-M8): (z, lgMgal, weight_emri, weight_tde)}.
    weight_idx: 2 for weight_emri, 3 for weight_tde (positions in that tuple).

    One figure per merger model, 2x4 subplots (one per M1-M8 universe
    type), each showing `merger_model`'s dominant (z, lgMgal) contribution
    region for that scenario, with dashed red lines marking the merger
    model's stated z/mass validity range (MERGER_MODEL_VALIDITY) -- i.e.
    how much of that scenario's dominant-contribution region relies on
    extrapolating the merger rate outside where it was actually measured.
    """
    validity = MERGER_MODEL_VALIDITY.get(merger_model, {})
    scenarios = list(results_by_scenario.keys())

    fig, axes = plt.subplots(2, 4, figsize=fig_size(TEXT_W, 0.62), sharex=True, sharey=True)
    cf = None
    for ax, scen in zip(axes.flat, scenarios):
        z, lgMgal, w_emri, w_tde = results_by_scenario[scen]
        weight = w_emri if weight_idx == 2 else w_tde
        if weight.sum() <= 0:
            ax.set_title(f"{SCEN_LABEL.get(scen, scen)} -- zero weight")
            ax.axis("off")
            continue
        cf, _, _ = _plot_weight_panel(ax, z, lgMgal, weight, label, smooth_sigma=smooth_sigma,
                                       validity=validity)
        ax.set_title(SCEN_LABEL.get(scen, scen))

    for ax in axes[-1, :]:
        ax.set_xlabel("$z$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\log_{10}(M_{\mathrm{gal},\star}/M_\odot)$")

    handles = [plt.Line2D([], [], marker="*", color="crimson", markeredgecolor="black",
                           linestyle="", markersize=6, label="peak bin")]
    if validity:
        handles.append(plt.Rectangle((0, 0), 0, 0, facecolor="none", edgecolor="red", hatch="xx",
                                      label="outside merger model's stated validity range"))
    if cf is not None:
        cbar = fig.colorbar(cf, ax=axes, pad=0.01, shrink=0.9, aspect=30)
        cbar.set_label(f"Fraction of {label} count per bin")
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.45, -0.07))
    # fig.suptitle(f"{label} dominant-contribution region across M1-M8 -- merger_model={merger_model}")
    savefig_all(fig, out_path, tight=True)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--merger_model", type=str, choices=MERGER_MODEL_CHOICES, default="baseline",
                   help="Galaxy merger-rate model (default: baseline).")
    p.add_argument("--all", action="store_true",
                   help="Loop over all merger models (mundy17, oleary21, duan24, baseline), "
                        "one figure per model, instead of just --merger_model.")
    p.add_argument("--mbh_mass_method", type=str, choices=["use_sigma", "use_lgMgal"], default="use_lgMgal")
    p.add_argument("--nucleation_method", type=str, choices=["hannah", "neumayer"], default="hannah")
    p.add_argument("--occupation_method", type=str, choices=["uniform", "varying"], default="uniform")
    p.add_argument("--scenario_grid", action="store_true",
                   help="For each of the 4 merger models, produce two figures (EMRI, TDE), each "
                        "with 2x4 subplots -- one per M1-M8 universe-type scenario -- showing that "
                        "merger model's dominant (z, lgMgal) contribution region per scenario, with "
                        "dashed red lines marking the merger model's stated z/mass validity range "
                        "from README_merger_models.md (MERGER_MODEL_VALIDITY). Saves "
                        "combined_dominant_scenario_grid_{merger_model}_{EMRI,TDE}.{png,pdf} in PLOTS_DIR, "
                        "ignoring --merger_model/--mbh_mass_method/--nucleation_method/"
                        "--occupation_method/--all.")
    p.add_argument("--grid_models", type=str, nargs="+", choices=MERGER_MODEL_CHOICES,
                   default=None,
                   help="With --scenario_grid: merger model(s) to sweep M1-M8 for "
                        "(default: all), e.g. --grid_models oleary21.")
    p.add_argument("--GALAXIES", type=int, default=200_000,
                   help="Monte Carlo sample size for the weighting pipeline (default: 200000).")
    p.add_argument("--Z_MAX", type=float, default=8.0)
    p.add_argument("--T_OBS_EMRI", type=float, default=6.0, help="years")
    p.add_argument("--T_OBS_TDE", type=float, default=17.25, help="years")
    
    args = p.parse_args()

    if args.scenario_grid:
        for mm in (args.grid_models or MERGER_MODEL_CHOICES):
            print(f"\n### merger_model={mm} ###")
            results_by_scenario = {}
            for scen_name, mbh_mass_method, nucleation_method, occupation_method in SCENARIOS_M1_M8:
                z, lgMgal, w_emri, w_tde = run_weighted_sample(
                    merger_model=mm, mbh_mass_method=mbh_mass_method,
                    nucleation_method=nucleation_method, occupation_method=occupation_method,
                    N_objs_sim=args.GALAXIES, Z_MAX=args.Z_MAX,
                    T_OBS_EMRI=args.T_OBS_EMRI, T_OBS_TDE=args.T_OBS_TDE)
                results_by_scenario[scen_name] = (z, lgMgal, w_emri, w_tde)
                if w_emri.sum() <= 0 and w_tde.sum() <= 0:
                    print(f"[SKIP] {mm}/{scen_name}: zero total weight.")

            for weight_idx, label in [(2, "EMRI"), (3, "TDE")]:
                out_path = f"combined_dominant_scenario_grid_{mm}_{label}"
                plot_scenario_grid(mm, results_by_scenario, weight_idx, label, out_path)
        sys.exit(0)

    scenario_tag = f"{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}"
    merger_models = MERGER_MODEL_CHOICES if args.all else [args.merger_model]

    results_by_model = {}
    for mm in merger_models:
        print(f"\n=== merger_model={mm}, scenario={scenario_tag} ===")
        z, lgMgal, w_emri, w_tde = run_weighted_sample(
            merger_model=mm, mbh_mass_method=args.mbh_mass_method,
            nucleation_method=args.nucleation_method, occupation_method=args.occupation_method,
            N_objs_sim=args.GALAXIES, Z_MAX=args.Z_MAX,
            T_OBS_EMRI=args.T_OBS_EMRI, T_OBS_TDE=args.T_OBS_TDE)
        results_by_model[mm] = (z, lgMgal, w_emri, w_tde)

        if w_emri.sum() <= 0 and w_tde.sum() <= 0:
            print(f"[SKIP] {mm}/{scenario_tag}: zero total weight, nothing to plot.")
            continue

        # Quick numeric summary alongside the plot
        for label, w in [("EMRI", w_emri), ("TDE", w_tde)]:
            if w.sum() <= 0:
                print(f"  {label}: zero total weight")
                continue
            order = np.argsort(w)[::-1]
            cum = np.cumsum(w[order]) / w.sum()
            n50 = np.searchsorted(cum, 0.5) + 1
            z_dom = np.average(z, weights=w)
            m_dom = np.average(lgMgal, weights=w)
            print(f"  {label}: weight-mean z={z_dom:.2f}, weight-mean log10(M*)={m_dom:.2f}; "
                  f"{n50}/{len(w)} samples carry 50% of the total weight")
