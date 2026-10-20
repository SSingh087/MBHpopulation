"""
Cumulative TDE (or EMRI) detections across all N universes for a given scenario,
with mean / min-max envelope / percentile bands, and an overlay of real detections.

Adapts your existing single-universe snippet:

    if len(t_detected) > 0:
        year = 2000 + (t_detected - 51544.5) / 365.25
        ...
        plt.step(edges[:-1], cumulative_counts, where='mid')

to aggregate across UNIVERSE_MODEL = 0 .. N_UNIVERSES-1 for one fixed scenario
(mbh_mass_method, nucleation_method, occupation_method, SURVEY).
"""
import sys, os, argparse
sys.path.insert(0, os.path.abspath('./'))
from utils import ASTROPHYSICAL_DATA_DIR, MERGER_MODEL_CHOICES
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
import seaborn as sns

# NOTE: "M1".."M8" below label (mbh_mass_method, nucleation_method,
# occupation_method) SCENARIO combos -- NOT redback TDE models. Kept as
# SCENARIO_NAME_MAP/SCENARIO_ORDER to avoid colliding with the separate
# TDE_MODELS axis (cooling_envelope/stream_stream_tde/tde_analytical/
# tde_fallback) added below.
SCENARIO_NAME_MAP = {
    ("use_lgMgal", "hannah",   "uniform"): "M1",
    ("use_sigma",  "hannah",   "uniform"): "M2",
    ("use_lgMgal", "neumayer", "uniform"): "M3",
    ("use_sigma",  "neumayer", "uniform"): "M4",
    ("use_lgMgal", "hannah",   "varying"): "M5",
    ("use_sigma",  "hannah",   "varying"): "M6",
    ("use_lgMgal", "neumayer", "varying"): "M7",
    ("use_sigma",  "neumayer", "varying"): "M8",
}
# Ordered list so legend/colors come out as M1..M8 consistently
SCENARIO_ORDER = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]

# Must match data_generation/gen_TDE_params.py's MODELS list.
TDE_MODELS = [
    "cooling_envelope",
    "stream_stream_tde",
    "tde_analytical",
    "tde_fallback",
]


MJD_REF = 51544.5  # MJD of 2000-01-01T12:00 (matches your year = 2000 + (t - 51544.5)/365.25)


def mjd_to_year(t_mjd):
    """Convert MJD to decimal year."""
    return 2000 + (np.asarray(t_mjd) - MJD_REF) / 365.25


def plot_combined_models(results, survey, tde_model, ax=None, show_band=(50, 95),
                          palette="colorblind", real_t_mjd=None,
                          real_label="Observed (TDE)", real_source_tags=None):
    """
    Overlay mean cumulative-detection curves for all 8 scenarios (M1-M8) for a
    given (survey, TDE model) on one axis, colored with a seaborn palette.

    Parameters
    ----------
    results : dict
        (mbh_mass_method, nucleation_method, occupation_method, tde_model, survey) ->
        (year_grid, curves, n_missing, max_data_year), as built in the __main__ loop.
    survey : str
        "ZTF" or "LSST".
    tde_model : str
        One of TDE_MODELS -- which redback TDE model's results to plot.
    show_band : float, tuple, or None
        If given, also draw a shaded percentile band around each model's mean,
        using the same color at low alpha. A float draws a single band (e.g.
        50); a tuple/list draws multiple bands in decreasing alpha.
    palette : str
        Any seaborn/matplotlib palette name (e.g. "colorblind", "tab10",
        "husl", "viridis").
    real_t_mjd : array-like or None
        Optional real detection epochs to overlay in the same panel.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6.5))

    if show_band is None:
        band_levels = []
    elif np.isscalar(show_band):
        band_levels = [float(show_band)]
    else:
        band_levels = [float(b) for b in show_band]

    colors = sns.color_palette(palette, n_colors=len(SCENARIO_ORDER))
    color_map = dict(zip(SCENARIO_ORDER, colors))

    plotted_any = False
    for (mbh_mass_method, nucleation_method, occupation_method, model, surv), \
            (year_grid, curves, n_missing, max_data_year) in results.items():
        if surv != survey or model != tde_model:
            continue
        label = SCENARIO_NAME_MAP.get(
            (mbh_mass_method, nucleation_method, occupation_method)
        )
        if label is None:
            continue  # scenario not in the table mapping; skip silently
 
        color = color_map[label]
        mean_curve = curves.mean(axis=0)
 
        ax.step(year_grid, mean_curve, where="post", color=color, lw=2.0,
                label=label)
 
        for i, pct in enumerate(sorted(band_levels, reverse=True)):
            lo = np.percentile(curves, (100 - pct) / 2, axis=0)
            hi = np.percentile(curves, 100 - (100 - pct) / 2, axis=0)
            ax.fill_between(year_grid, lo, hi, step="post",
                            alpha=max(0.05, 0.28 - 0.10 * i), color=color)
        plotted_any = True
 
    if not plotted_any:
        raise RuntimeError(f"No results found for survey={survey!r} to combine-plot.")

    if real_t_mjd is not None and len(real_t_mjd) > 0:
        real_year = np.sort(mjd_to_year(real_t_mjd))
        real_count = np.arange(1, len(real_year) + 1, dtype=float)
        ax.step(real_year, real_count, where="post", color="crimson", lw=2.2,
                linestyle="--", label=real_label)

        if real_source_tags is not None:
            for yr, tag in real_source_tags:
                idx = np.searchsorted(real_year, yr)
                y_val = real_count[min(idx, len(real_count) - 1)]
                ax.annotate(tag, xy=(yr, y_val), xytext=(5, 8),
                            textcoords="offset points", fontsize=7, rotation=45,
                            color="crimson", alpha=0.8)
 
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative detections")
    band_text = ", ".join(f"{pct}%" for pct in sorted(band_levels, reverse=True))
    band_str = f" (CI: {band_text})" if band_levels else ""
    ax.set_title(f"{survey} cumulative detections -- all scenarios ({tde_model}){band_str}")
    ax.grid(alpha=0.3)
    # Sort legend entries M1..M8 regardless of dict iteration order
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: SCENARIO_ORDER.index(labels[i]) if labels[i] in SCENARIO_ORDER else 99)
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="upper left", fontsize=8, framealpha=0.9, ncol=2,
              title="Model")
    return ax


def load_t_detected(astro_data_dir, galaxies, survey, mbh_mass_method, nucleation_method,
                     occupation_method, merger_model, tde_model, universe_model):
    """
    Load the detection-passing t0_mjd_transient array for one (universe, TDE
    model), straight from the SNR/classifier output file (OUT_PATH in
    calculate_SNR_TDE.py):

        {ASTROPHYSICAL_DATA_DIR}/{GALAXIES}/TDE_SNRs_{SURVEY}_{mbh_mass_method}_
        {nucleation_method}_{occupation_method}_{merger_model}_{tde_model}_{UNIVERSE_MODEL}.h5

    That file already stores ONLY the events that passed DETECTION_THRESHOLD
    (see events_idx / OUT_PATH in that script), so t0_mjd_transient there IS
    t_detected -- no extra masking needed here.
    """
    path = Path(
        f"{astro_data_dir}/{galaxies}/TDE_SNRs_{survey}_{mbh_mass_method}_"
        f"{nucleation_method}_{occupation_method}_{merger_model}_{tde_model}_{universe_model}.h5"
    )
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        if "t0_mjd_transient" not in f or f.attrs.get("n_events_passed", 1) == 0:
            return np.array([])
        return f["t0_mjd_transient"][:]


def build_year_grid(all_years_lists, n_bins_per_year=1):
    """Common year grid spanning the union of all universes' detection epochs."""
    all_years = np.concatenate(all_years_lists) if len(all_years_lists) else np.array([])
    if all_years.size == 0:
        raise ValueError("No detections found across any universe for this scenario.")
    year_min = int(np.floor(all_years.min()))
    year_max = int(np.ceil(all_years.max()))
    # n_bins_per_year=1 -> integer-year bins, matching your original snippet;
    # increase for finer time resolution (e.g. quarterly = 4).
    return np.arange(year_min, year_max + 1, 1.0 / n_bins_per_year)


def cumulative_curve_on_grid(t_mjd, year_grid):
    """Step-function cumulative count for one universe, evaluated at year_grid edges."""
    year = mjd_to_year(t_mjd)
    counts, _ = np.histogram(year, bins=year_grid)
    cum = np.cumsum(counts)
    # cum[i] = detections up to year_grid[i+1]; prepend 0 at year_grid[0]
    return np.concatenate([[0], cum])


def aggregate_scenario(astro_data_dir, galaxies, survey, mbh_mass_method, nucleation_method,
                        occupation_method, merger_model, tde_model, n_universes=100, n_bins_per_year=1):
    """
    Returns:
        year_grid : (n_edges,) array of year grid points
        curves    : (n_curves, n_edges) array, cumulative counts per universe,
                    where n_curves = n_universes - n_missing (missing = no
                    output file at all; zero-detection universes ARE included,
                    as an all-zero row, since that's a real outcome not missing
                    data)
        n_missing : number of universes with no SNR output file found
        max_data_year : the true latest detection epoch across all universes in
                    this scenario (unlike year_grid[-1], which is rounded up to
                    the next bin edge and can overshoot this by up to one full
                    bin -- use max_data_year, not year_grid[-1], when setting
                    plot x-limits)
    """
    per_universe_years = []
    per_universe_t = []
    n_missing = 0
    for u in range(n_universes):
        try:
            t_mjd = load_t_detected(astro_data_dir, galaxies, survey, mbh_mass_method,
                                     nucleation_method, occupation_method, merger_model, tde_model, u)
        except FileNotFoundError:
            n_missing += 1
            continue
        if len(t_mjd) == 0:
            # Universe ran but had 0 detections -- this is a REAL zero, not missing
            # data. It still needs to appear in the ensemble (as an all-zero curve)
            # once we know the shared year_grid; handled after the grid is built.
            per_universe_t.append(t_mjd)
            continue
        per_universe_t.append(t_mjd)
        per_universe_years.append(mjd_to_year(t_mjd))

    if n_missing:
        print(f"[WARN] {n_missing}/{n_universes} universes had no SNR output file "
              f"for {mbh_mass_method}_{nucleation_method}_{occupation_method}_{tde_model} "
              f"({survey}) -- excluded from the ensemble. If this is a large fraction, the "
              f"mean/CI band below is based on fewer than {n_universes} universes.")

    if not per_universe_years:
        raise RuntimeError(f"No usable universes with >=1 detection found for scenario "
                            f"{mbh_mass_method}_{nucleation_method}_{occupation_method}_{tde_model} "
                            f"({survey}) -- cannot build a year grid. (n_missing={n_missing}, "
                            f"n_zero_detections={len(per_universe_t) - len(per_universe_years)})")

    year_grid = build_year_grid(per_universe_years, n_bins_per_year=n_bins_per_year)
    max_data_year = max(y.max() for y in per_universe_years)

    # per_universe_t includes both nonzero- and zero-detection universes;
    # cumulative_curve_on_grid handles an empty t array fine (all-zero curve).
    curves = np.stack([cumulative_curve_on_grid(t, year_grid) for t in per_universe_t])
    return year_grid, curves, n_missing, max_data_year


def plot_scenario_envelope(year_grid, curves, survey, scenario_label,
                            real_t_mjd=None, real_label="Observed (ZTF)",
                            real_source_tags=None, ax=None, show_ci=(50, 95)):
    """
    curves : (n_universes, n_edges) cumulative-count array from aggregate_scenario()
    real_t_mjd : optional array of MJD discovery times for the real, observed sample
    real_source_tags : optional list of (year, label) tuples to annotate individual
                        real detections with their literature source (e.g. paper name)
    """
    x = year_grid  # left edges of each bin, plus the trailing edge from build_year_grid
    x_mid = x  # already the grid points corresponding to curves' columns

    mean_curve = curves.mean(axis=0)
    min_curve = curves.min(axis=0)
    max_curve = curves.max(axis=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6.5))

    # Envelope: absolute min-max across all universes
    ax.fill_between(x_mid, min_curve, max_curve, step="post", alpha=0.15,
                     color="C0", label=f"Min--max across {curves.shape[0]} universes")

    # Percentile bands (e.g. 50% and 95% credible intervals)
    for pct, alpha in zip(sorted(show_ci), (0.35, 0.20)):
        lo = np.percentile(curves, (100 - pct) / 2, axis=0)
        hi = np.percentile(curves, 100 - (100 - pct) / 2, axis=0)
        ax.fill_between(x_mid, lo, hi, step="post", alpha=alpha, color="C0",
                         label=f"{pct}% interval")

    # Mean, solid line
    ax.step(x_mid, mean_curve, where="post", color="C0", lw=2.2, label="Mean (all universes)")

    # Real detections overlay -- plotted at full event resolution (one step per
    # detection), matching plot_combined_models, rather than binned onto the
    # coarse simulated year_grid (which artificially blocks same-year events
    # together into a single jump).
    if real_t_mjd is not None and len(real_t_mjd) > 0:
        real_year = np.sort(mjd_to_year(real_t_mjd))
        real_cum = np.arange(1, len(real_year) + 1, dtype=float)
        ax.step(real_year, real_cum, where="post", color="crimson", lw=2.2,
                linestyle="--", label=real_label)

        if real_source_tags is not None:
            for yr, tag in real_source_tags:
                idx = np.searchsorted(real_year, yr)
                y_val = real_cum[min(idx, len(real_cum) - 1)]
                ax.annotate(tag, xy=(yr, y_val), xytext=(5, 8),
                            textcoords="offset points", fontsize=7, rotation=45,
                            color="crimson", alpha=0.8)

    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative detections")
    ax.set_title(f"{survey} cumulative detections -- {scenario_label}")
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1)  # log scale can't show 0; 1 is the effective floor for count data
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    return ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare per-universe TDE detection "
                                      "ensembles across scenarios, TDE models, and surveys, "
                                      "with an optional real-catalog overlay.")
    parser.add_argument("--GALAXIES", type=int, default=100000,
                         help="Must match --GALAXIES from gen_events.sub etc.")
    parser.add_argument("--N_UNIVERSES", type=int, default=100,
                         help="Must match make_data_gen_dag.py's N_UNIVERSES.")
    parser.add_argument("--merger_model", type=str, required=True, choices=MERGER_MODEL_CHOICES,
                         help="Galaxy merger-rate model whose universes are compared (tags input files and output plots).")
    args = parser.parse_args()

    GALAXIES = args.GALAXIES
    N_UNIVERSES = args.N_UNIVERSES

    MBH_NUC_SCENARIOS = [
        ("use_lgMgal", "hannah"),
        ("use_sigma",  "hannah"),
        ("use_lgMgal", "neumayer"),
        ("use_sigma",  "neumayer"),
    ]
    OCCUPATION_METHODS = ["uniform", "varying"]
    SURVEYS = ["ZTF", "LSST"]

    # Real detections overlay -- loaded from ASTROPHYSICAL_DATA_DIR/real_tde_detections.npz
    # (see pull_real_tde_catalog.py). Falls back to None (no overlay) if not found yet.
    try:
        _real = np.load(f"{ASTROPHYSICAL_DATA_DIR}/real_tde_detections.npz", allow_pickle=True)
        REAL_T_MJD = {"ZTF": _real["ztf_mjd"], "LSST": _real["lsst_mjd"]}
        REAL_SOURCE_TAGS = {"ZTF": None, "LSST": None}  # add (year, "Hammerstein+23")-style tags if desired
        print(f"Loaded real detections: {len(REAL_T_MJD['ZTF'])} ZTF, "
              f"{len(REAL_T_MJD['LSST'])} LSST")
    except FileNotFoundError:
        print("[WARN] real_tde_detections.npz not found in ASTROPHYSICAL_DATA_DIR -- "
              "run pull_real_tde_catalog.py first. Proceeding without a real-detections overlay.")
        REAL_T_MJD = {"ZTF": None, "LSST": None}
        REAL_SOURCE_TAGS = {"ZTF": None, "LSST": None}

    # Monthly bins (12/year) instead of yearly -- lets the simulated envelope
    # track the real curve's actual cadence instead of only updating once a
    # year, and shrinks the year-grid overshoot from a late-in-year detection
    # (see max_data_year below) down to at most one month instead of one year.
    N_BINS_PER_YEAR = 12

    results = {}     # (mbh_method, nuc_method, occ_method, tde_model, survey) -> (year_grid, curves, n_missing, max_data_year)
    failures = []     # (survey, scenario_label, tde_model) that raised (no usable universes)

    for survey in SURVEYS:
        for tde_model in TDE_MODELS:
            for mbh_mass_method, nucleation_method in MBH_NUC_SCENARIOS:
                for occupation_method in OCCUPATION_METHODS:
                    scenario_label = f"{mbh_mass_method}_{nucleation_method}_{occupation_method}"
                    print(f"\n=== {survey} / {scenario_label} / {tde_model} ===")
                    try:
                        year_grid, curves, n_missing, max_data_year = aggregate_scenario(
                            ASTROPHYSICAL_DATA_DIR, GALAXIES, survey, mbh_mass_method,
                            nucleation_method, occupation_method, args.merger_model, tde_model,
                            n_universes=N_UNIVERSES, n_bins_per_year=N_BINS_PER_YEAR,
                        )
                    except RuntimeError as e:
                        print(f"[SKIP] {survey}/{scenario_label}/{tde_model}: {e}")
                        failures.append((survey, scenario_label, tde_model))
                        continue

                    results[(mbh_mass_method, nucleation_method, occupation_method,
                             tde_model, survey)] = (year_grid, curves, n_missing, max_data_year)

                    ax = plot_scenario_envelope(
                        year_grid, curves, survey, f"{scenario_label} ({tde_model})",
                        real_t_mjd=REAL_T_MJD[survey],
                        real_source_tags=REAL_SOURCE_TAGS[survey],
                    )
                    plt.tight_layout()
                    out_path = (f"{ASTROPHYSICAL_DATA_DIR}/combined_per_model_cumulative_envelope_"
                                f"{survey}_{scenario_label}_{args.merger_model}_{tde_model}.png")
                    plt.savefig(out_path, dpi=200)
                    plt.close(ax.figure)
                    print(f"Saved: {out_path}  "
                          f"({curves.shape[0]}/{N_UNIVERSES} universes used, "
                          f"{n_missing} missing)")

    n_total_combos = len(SURVEYS) * len(TDE_MODELS) * len(MBH_NUC_SCENARIOS) * len(OCCUPATION_METHODS)
    print(f"\n{'='*60}")
    print(f"Done. {len(results)}/{n_total_combos} "
          f"(survey, scenario, TDE model) combinations plotted successfully.")
    if failures:
        print(f"Skipped ({len(failures)}):")
        for survey, scenario_label, tde_model in failures:
            print(f"  - {survey} / {scenario_label} / {tde_model}")

    for survey in SURVEYS:
        for tde_model in TDE_MODELS:
            try:
                ax = plot_combined_models(
                    results,
                    survey,
                    tde_model,
                    palette="colorblind",
                    show_band=(50, 95),
                    real_t_mjd=REAL_T_MJD[survey],
                    real_label="Observed (TDE)",
                    real_source_tags=REAL_SOURCE_TAGS[survey],
                )
            except RuntimeError as e:
                print(f"[SKIP combined] {survey}/{tde_model}: {e}")
                continue
            survey_entries = [
                (year_grid, max_data_year)
                for (mbh_mass_method, nucleation_method, occupation_method, model, surv),
                (year_grid, curves, n_missing, max_data_year) in results.items()
                if surv == survey and model == tde_model
            ]
            if survey_entries:
                x_min = min(grid.min() for grid, _ in survey_entries)
                # Use the true latest detection epoch, not year_grid[-1] -- the
                # grid's trailing edge is rounded up to the next bin boundary
                # and can overshoot the real data by up to one bin.
                x_max = max(mdy for _, mdy in survey_entries)
                if REAL_T_MJD[survey] is not None and len(REAL_T_MJD[survey]) > 0:
                    real_years = mjd_to_year(REAL_T_MJD[survey])
                    x_min = min(x_min, float(np.min(real_years)))
                    x_max = max(x_max, float(np.max(real_years)))
                ax.set_xlim(x_min, x_max)
            ax.set_yscale("log")
            ax.set_ylim(bottom=1)  # log scale can't show 0; 1 is the effective floor for count data
            plt.tight_layout()
            out_path = f"{ASTROPHYSICAL_DATA_DIR}/combined_cumulative_{survey}_{args.merger_model}_{tde_model}.png"
            plt.savefig(out_path, dpi=200)
            plt.close(ax.figure)
            print(f"Saved combined plot: {out_path}")
