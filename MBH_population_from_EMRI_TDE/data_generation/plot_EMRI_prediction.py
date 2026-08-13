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
import sys, os 
sys.path.insert(0, os.path.abspath('../'))

from utils import ASTROPHYSICAL_DATA_DIR

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

MJD_REF = 51544.5  # MJD of 2000-01-01T12:00 (matches your year = 2000 + (t - 51544.5)/365.25)


def mjd_to_year(t_mjd):
    return 2000 + (np.asarray(t_mjd) - MJD_REF) / 365.25


def load_t_detected(astro_data_dir, galaxies, survey, mbh_mass_method, nucleation_method,
                     occupation_method, universe_model):
    """
    Load the detection-passing t0_mjd_transient array for one universe, straight
    from the SNR/classifier output file (OUT_PATH in your snr/classifier script):

        {ASTROPHYSICAL_DATA_DIR}/{GALAXIES}/TDE_SNRs_{SURVEY}_{mbh_mass_method}_
        {nucleation_method}_{occupation_method}_{UNIVERSE_MODEL}.h5

    That file already stores ONLY the events that passed DETECTION_THRESHOLD
    (see events_idx / OUT_PATH in that script), so t0_mjd_transient there IS
    t_detected -- no extra masking needed here.
    """
    path = Path(
        f"{astro_data_dir}/{galaxies}/TDE_SNRs_{survey}_{mbh_mass_method}_"
        f"{nucleation_method}_{occupation_method}_{universe_model}.h5"
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
                        occupation_method, n_universes=100, n_bins_per_year=1):
    """
    Returns:
        year_grid : (n_edges,) array of year grid points
        curves    : (n_curves, n_edges) array, cumulative counts per universe,
                    where n_curves = n_universes - n_missing (missing = no
                    output file at all; zero-detection universes ARE included,
                    as an all-zero row, since that's a real outcome not missing
                    data)
        n_missing : number of universes with no SNR output file found
    """
    per_universe_years = []
    per_universe_t = []
    n_missing = 0
    for u in range(n_universes):
        try:
            t_mjd = load_t_detected(astro_data_dir, galaxies, survey, mbh_mass_method,
                                     nucleation_method, occupation_method, u)
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
              f"for {mbh_mass_method}_{nucleation_method}_{occupation_method} ({survey}) "
              f"-- excluded from the ensemble. If this is a large fraction, the "
              f"mean/CI band below is based on fewer than {n_universes} universes.")

    if not per_universe_years:
        raise RuntimeError(f"No usable universes with >=1 detection found for scenario "
                            f"{mbh_mass_method}_{nucleation_method}_{occupation_method} "
                            f"({survey}) -- cannot build a year grid. (n_missing={n_missing}, "
                            f"n_zero_detections={len(per_universe_t) - len(per_universe_years)})")

    year_grid = build_year_grid(per_universe_years, n_bins_per_year=n_bins_per_year)

    # per_universe_t includes both nonzero- and zero-detection universes;
    # cumulative_curve_on_grid handles an empty t array fine (all-zero curve).
    curves = np.stack([cumulative_curve_on_grid(t, year_grid) for t in per_universe_t])
    return year_grid, curves, n_missing


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

    # Real detections overlay
    if real_t_mjd is not None and len(real_t_mjd) > 0:
        real_year = mjd_to_year(real_t_mjd)
        real_grid = build_year_grid([real_year], n_bins_per_year=1)
        # Re-evaluate on the *same* x grid as the simulated curves for direct comparison
        real_cum = cumulative_curve_on_grid(real_t_mjd, x_mid)
        ax.step(x_mid, real_cum, where="post", color="crimson", lw=2.2,
                linestyle="--", label=real_label)

        if real_source_tags is not None:
            for yr, tag in real_source_tags:
                idx = np.searchsorted(x_mid, yr)
                y_val = real_cum[min(idx, len(real_cum) - 1)]
                ax.annotate(tag, xy=(yr, y_val), xytext=(5, 8),
                            textcoords="offset points", fontsize=7, rotation=45,
                            color="crimson", alpha=0.8)

    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative detections")
    ax.set_title(f"{survey} cumulative detections -- {scenario_label}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    return ax


if __name__ == "__main__":


    GALAXIES = 100000 
    N_UNIVERSES = 100

    MBH_NUC_SCENARIOS = [
        ("use_lgMgal", "hannah"),
        ("use_sigma",  "hannah"),
        ("use_lgMgal", "neumayer"),
        ("use_sigma",  "neumayer"),
    ]
    OCCUPATION_METHODS = ["uniform", "varying"]
    SURVEYS = ["ZTF", "LSST"]

    # Placeholder -- replace with the real, compiled TDE discovery-time array(s),
    # split by survey once compiled (see the separate catalogue-compilation step).
    REAL_T_MJD = {"ZTF": None, "LSST": None}
    REAL_SOURCE_TAGS = {"ZTF": None, "LSST": None}

    results = {}     # (mbh_method, nuc_method, occ_method, survey) -> (year_grid, curves, n_missing)
    failures = []     # scenarios that raised (no usable universes) -- reported at the end

    for survey in SURVEYS:
        for mbh_mass_method, nucleation_method in MBH_NUC_SCENARIOS:
            for occupation_method in OCCUPATION_METHODS:
                scenario_label = f"{mbh_mass_method}_{nucleation_method}_{occupation_method}"
                print(f"\n=== {survey} / {scenario_label} ===")
                try:
                    year_grid, curves, n_missing = aggregate_scenario(
                        ASTROPHYSICAL_DATA_DIR, GALAXIES, survey, mbh_mass_method,
                        nucleation_method, occupation_method, n_universes=N_UNIVERSES,
                    )
                except RuntimeError as e:
                    print(f"[SKIP] {survey}/{scenario_label}: {e}")
                    failures.append((survey, scenario_label))
                    continue

                results[(mbh_mass_method, nucleation_method, occupation_method, survey)] = (
                    year_grid, curves, n_missing
                )

                ax = plot_scenario_envelope(
                    year_grid, curves, survey, scenario_label,
                    real_t_mjd=REAL_T_MJD[survey],
                    real_source_tags=REAL_SOURCE_TAGS[survey],
                )
                plt.tight_layout()
                out_path = (f"{ASTROPHYSICAL_DATA_DIR}/cumulative_envelope_"
                            f"{survey}_{scenario_label}.png")
                plt.savefig(out_path, dpi=200)
                plt.close(ax.figure)
                print(f"Saved: {out_path}  "
                      f"({curves.shape[0]}/{N_UNIVERSES} universes used, "
                      f"{n_missing} missing)")

    print(f"\n{'='*60}")
    print(f"Done. {len(results)}/{len(SURVEYS)*len(MBH_NUC_SCENARIOS)*len(OCCUPATION_METHODS)} "
          f"(survey, scenario) combinations plotted successfully.")
    if failures:
        print(f"Skipped ({len(failures)}):")
        for survey, scenario_label in failures:
            print(f"  - {survey} / {scenario_label}")