#!/usr/bin/env python
"""
Scans the EMRI/TDE SNR-detection output for EVERY (mbh_mass_method,
nucleation_method, occupation_method, universe) combo -- ALL scenarios
AND ALL occupation methods together, in ONE combined scan -- and picks
ONE winning UNIVERSE_MODEL PER TDE MODEL that maximizes the worst
individual (scenario x channel) cell (maximin), where channel is
{EMRI, TDE-ZTF-<model>, TDE-LSST-<model>}.

THIS IS INTENTIONALLY PER-TDE-MODEL: each redback TDE model
(cooling_envelope, stream_stream_tde, tde_analytical, tde_fallback) has
its own detection statistics, and a single aggregated winner can hide a
model with poor coverage behind one with lots of detections. So the
selection is run once per TDE model, each time picking the best
UNIVERSE_MODEL for that model specifically -- EMRI is included in every
model's selection (it doesn't depend on TDE model, but a winner still
needs healthy EMRI counts). Different TDE models can therefore end up
with DIFFERENT winning UNIVERSE_MODEL values; downstream comparisons
that care about a specific TDE model should use that model's own
max_detected_<model>_* files.

Selection is still COMBINED across occupation methods and mbh/nucleation
scenarios within each model's run (not split further per occupation
method): the winning universe for a given TDE model must be the same
UNIVERSE_MODEL used for every downstream (mbh, nuc, occupation)
comparison of that model, or the "same realization for every fair
comparison" property breaks down.

ASSUMPTION (single point of change if wrong): every stage's output
filename gets "_{occupation_method}_{U}" appended after the existing
"_{mbh_mass_method}_{nucleation_method}" scenario suffix, matching
what make_data_gen_dag.py passes via OCCUPATION_METHODS/UNIVERSE_MODEL.
If your actual suffix pattern differs, only emri_snr_path()/
tde_snr_path() below need to change -- nothing else in this script
assumes a particular pattern.

Reads (per mbh/nuc scenario, per occupation method, per TDE model, per universe):
    EMRI_SNRs_data_{mbh}_{nuc}_{occ}_{mm}_{U}.h5           .attrs['n_events_total'/'n_events_passed']
    TDE_SNRs_ZTF_{mbh}_{nuc}_{occ}_{mm}_{model}_{U}.h5     .attrs['n_events_total'/'n_events_passed']
    TDE_SNRs_LSST_{mbh}_{nuc}_{occ}_{mm}_{model}_{U}.h5    .attrs['n_events_total'/'n_events_passed']

Writes (under {outdir}/):
    universe_report_{mbh}_{nuc}_{occ}.csv, universe_report_all.csv,
    universe_comparison_{mbh}_{nuc}_{occ}.png, combined_totals_by_universe.png,
    all_scenarios_all_channels_scatter.png, emri_tde_ratio_scatter.png,
    summary_table.tex, ratio_table.tex, full_report.txt,
    model_selection.txt (only if --OBSERVED_TDE_ZTF/LSST given),
    best_universe.txt (index of every TDE model's winning universe),
    best_universe_<model>.txt (one per TDE model, full detail)
    {base}/max_detected_{model}_..._{mbh}_{nuc}_{occ}_{merger_model}_*.h5   (that TDE model's own
                                                    winning universe's files,
                                                    every scenario x occupation
                                                    combo, copied with a
                                                    max_detected_<model>_ prefix)

Usage:
    python report_universes.py --GALAXIES 100000 --N_UNIVERSES 100
"""

import os, sys, argparse, csv, shutil, math
sys.path.insert(0, os.path.abspath('./astrophysical_setup'))
sys.path.insert(0, os.path.abspath('./'))
from utils import ASTROPHYSICAL_DATA_DIR, MERGER_MODEL_CHOICES

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MBH_NUC_SCENARIOS = [
    ("use_lgMgal", "hannah"),
    ("use_sigma",  "hannah"),
    ("use_lgMgal", "neumayer"),
    ("use_sigma",  "neumayer"),
]
OCCUPATION_METHODS = ["uniform", "varying"]

# Full combined scenario list: every (mbh, nuc) x every occupation
# method. This IS the scenario axis for the maximin selection below --
# it now has len(MBH_NUC_SCENARIOS) * len(OCCUPATION_METHODS) entries
# (default 4 x 2 = 8), not 4.
SCENARIOS = [(mbh, nuc, occ) for (mbh, nuc) in MBH_NUC_SCENARIOS for occ in OCCUPATION_METHODS]

SURVEYS = ["ZTF", "LSST"]

# Model-name key used in the paper's tables (Table~\ref{tab:model_names}):
# M1..M8, matching data_generation/compare_TDE_detection.py's SCENARIO_NAME_MAP
# so the same label means the same scenario everywhere in the project.
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
MODEL_ORDER = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]
NAME_TO_SCENARIO = {v: k for k, v in SCENARIO_NAME_MAP.items()}


def model_name(mbh, nuc, occ):
    """Paper model name of a scenario."""
    return SCENARIO_NAME_MAP[(mbh, nuc, occ)]


def mbh_mass_latex(mbh):
    """LaTeX label of the MBH-mass relation."""
    return r"$M_\bullet$--$M_\star$" if mbh == "use_lgMgal" else r"$M_\bullet$--$\sigma$"


def nucleation_latex(nuc):
    """LaTeX label of the nucleation model."""
    return (r"Hannah et al. \citep{Hannah:2024dmr}" if nuc == "hannah"
            else r"Neumayer et al. \citep{Neumayer:2020gno}")


def occupation_latex(occ):
    """LaTeX label of the occupation model."""
    return (r"Uniform ($f_\mathrm{occ}=1$)" if occ == "uniform"
            else r"Varying \citep{2025ApJ...992..176Z}")

# Must match data_generation/gen_TDE_params.py's MODELS list -- the
# per-model breakdown attrs (n_total_<model>/n_passed_<model>) written
# by calculate_SNR_TDE.py.
TDE_MODELS = ["cooling_envelope", "stream_stream_tde", "tde_analytical", "tde_fallback"]

# Table~\ref{tab:counts_summary} (summary_table.tex, the paper's Table 3) reports
# a single TDE light-curve model's counts, not a sum across all four redback
# models -- tde_fallback is the fiducial choice for that table.
TABLE_TDE_MODEL = "tde_fallback"

# One color per (mbh, nuc, occ) combo, one marker per channel -- used
# consistently across every plot. tab10/tab20-based so it scales past
# 4 scenarios without hand-picking every color.
_CMAP = plt.get_cmap("tab10") if len(SCENARIOS) <= 10 else plt.get_cmap("tab20")
SCENARIO_COLORS = {s: _CMAP(i / max(len(SCENARIOS) - 1, 1)) for i, s in enumerate(SCENARIOS)}

CHANNEL_MARKERS = {
    "EMRI":     "o",
    "TDE-ZTF":  "s",
    "TDE-LSST": "^",
}


def scenario_tag(mbh, nuc, occ):
    """Short tag identifying a scenario."""
    return f"{mbh}_{nuc}_{occ}"


def sci_latex(x, sig=3):
    """Format a float as LaTeX scientific notation, e.g. 6.898e-02 ->
    '6.90 \\times 10^{-2}'. Used for the ratio table, whose values span
    several orders of magnitude (EMRI/TDE-LSST ~1e-4-1e-3)."""
    if x != x:  # NaN check without importing math here
        return "\\text{NaN}"
    mantissa, exp = f"{x:.{sig}e}".split("e")
    exp = int(exp)
    return f"{mantissa} \\times 10^{{{exp}}}"

# Single source of truth for the universe-tagged filenames -- change
# these if your actual naming pattern differs.
def emri_snr_path(base, mbh, nuc, occ, mm, u):
    """Path of the EMRI SNR file of a scenario."""
    return f"{base}/EMRI_SNRs_data_{mbh}_{nuc}_{occ}_{mm}_{u}.h5"

def tde_snr_path(base, survey, mbh, nuc, occ, mm, model, u):
    """Path of the TDE SNR file of a scenario."""
    return f"{base}/TDE_SNRs_{survey}_{mbh}_{nuc}_{occ}_{mm}_{model}_{u}.h5"

def zscore(value, mean, var):
    """(value - mean) / std, using this scenario's own distribution.
    Returns NaN if var is 0 (can't standardize a spike)."""
    std = math.sqrt(var)
    if std == 0:
        return float("nan")
    return (value - mean) / std

def read_counts(path):
    """Returns (n_total, n_passed) or (None, None) if the file is
    missing/unreadable (job not yet finished, or failed)."""
    if not os.path.exists(path):
        return None, None
    try:
        with h5py.File(path, "r") as f:
            return int(f.attrs["n_events_total"]), int(f.attrs["n_events_passed"])
    except (OSError, KeyError) as exc:
        print(f"  [warn] could not read {path}: {exc}")
        return None, None


def read_model_breakdown(base, survey, mbh, nuc, occ, mm, u):
    """Returns {model: (n_total, n_passed)} by reading each TDE model's
    OWN TDE_SNRs file for this (scenario, survey, universe) -- one file
    per model now (see tde_snr_path()), not attrs sliced out of one
    combined file. Missing/unreadable files are simply absent from the
    returned dict."""
    breakdown = {}
    for m in TDE_MODELS:
        n_total, n_passed = read_counts(tde_snr_path(base, survey, mbh, nuc, occ, mm, m, u))
        if n_total is not None:
            breakdown[m] = (n_total, n_passed)
    return breakdown


def main():
    parser = argparse.ArgumentParser(description="Pick one UNIVERSE_MODEL, maximizing the worst "
                                      "(mbh x nuc x occupation x channel) cell, across EVERY "
                                      "scenario AND occupation method combined.")
    parser.add_argument("--GALAXIES", type=int, required=True)
    parser.add_argument("--merger_model", type=str, required=True, choices=MERGER_MODEL_CHOICES,
                         help="Galaxy merger-rate model whose universes are scanned. One report per model; "
                              "outputs go to PLOTS/universe_report/<merger_model>/.")
    parser.add_argument("--N_UNIVERSES", type=int, required=True,
                         help="Number of universes to scan (0..N_UNIVERSES-1), "
                              "matching make_data_gen_dag.py's N_UNIVERSES.")
    parser.add_argument("--REQUIRE_ALL_SCENARIOS", action="store_true", default=True,
                         help="Only consider universes with readable output for ALL "
                              f"{len(SCENARIOS)} (mbh, nuc, occupation) scenarios when picking "
                              "the winner (default: on). Use --no-REQUIRE_ALL_SCENARIOS to "
                              "allow partial universes to compete (their total will just be "
                              "lower, which biases the selection -- off by default for that "
                              "reason).")
    parser.add_argument("--no-REQUIRE_ALL_SCENARIOS", dest="REQUIRE_ALL_SCENARIOS",
                         action="store_false")
    parser.add_argument("--OBSERVED_TDE_ZTF", type=int, default=None,
                         help="Optional: an observed/hypothetical TDE-ZTF detection "
                              "count. If given (with --OBSERVED_TDE_LSST), runs a "
                              "Poisson-likelihood model-selection comparison across "
                              "every scenario and predicts the corresponding EMRI "
                              "count under the favored model.")
    parser.add_argument("--OBSERVED_TDE_LSST", type=int, default=None,
                         help="Optional: paired with --OBSERVED_TDE_ZTF, see above.")
    args = parser.parse_args()

    base = f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}"
    outdir = f"{base}/PLOTS/universe_report/{args.merger_model}"
    os.makedirs(outdir, exist_ok=True)

    print(f"=== Combined scan: {len(MBH_NUC_SCENARIOS)} (mbh, nuc) scenarios x "
          f"{len(OCCUPATION_METHODS)} occupation methods = {len(SCENARIOS)} total scenarios ===")

    # per_scenario_rows[tag] = {universe: row_dict}
    per_scenario_rows = {}
    all_rows = []
    scenario_stats = {}  # tag -> {mean/std for emri, ztf, lsst, n}

    for mbh, nuc, occ in SCENARIOS:
        tag = scenario_tag(mbh, nuc, occ)
        print(f"\n=== scanning scenario: {tag} ===")

        rows_by_universe = {}
        for u in range(args.N_UNIVERSES):
            n_emri_total, n_emri_passed = read_counts(emri_snr_path(base, mbh, nuc, occ, args.merger_model, u))

            # Per-model breakdown IS the source of truth now (one TDE_SNRs
            # file per model) -- the channel total is the SUM across all 4
            # models (diagnostic-only axis; feeds the same (scenario x
            # channel) maximin selection as before, unchanged).
            ztf_by_model = read_model_breakdown(base, "ZTF", mbh, nuc, occ, args.merger_model, u)
            lsst_by_model = read_model_breakdown(base, "LSST", mbh, nuc, occ, args.merger_model, u)

            if n_emri_passed is None and not ztf_by_model and not lsst_by_model:
                # Nothing at all for this universe -- likely still running
                # or failed. Skip rather than fabricate a zero.
                continue

            n_emri_passed = n_emri_passed or 0
            n_ztf_passed = sum(p for _, p in ztf_by_model.values())
            n_lsst_passed = sum(p for _, p in lsst_by_model.values())
            n_total_detected = n_emri_passed + n_ztf_passed + n_lsst_passed

            row = {
                "universe": u,
                "mbh_mass_method": mbh,
                "nucleation_method": nuc,
                "occupation_method": occ,
                "merger_model": args.merger_model,
                "n_emri_passed": n_emri_passed,
                "n_tde_ztf_passed": n_ztf_passed,
                "n_tde_lsst_passed": n_lsst_passed,
                "n_total_detected": n_total_detected,
            }
            for m in TDE_MODELS:
                zt, zp = ztf_by_model.get(m, (0, 0))
                lt, lp = lsst_by_model.get(m, (0, 0))
                row[f"n_tde_ztf_{m}_total"] = zt
                row[f"n_tde_ztf_{m}_passed"] = zp
                row[f"n_tde_lsst_{m}_total"] = lt
                row[f"n_tde_lsst_{m}_passed"] = lp
            rows_by_universe[u] = row
            all_rows.append(row)

        if not rows_by_universe:
            print(f"  [warn] no readable universe output for {tag} -- this scenario "
                  f"will contribute nothing to the combined total (are the "
                  f"corresponding DAG jobs finished?)")
            per_scenario_rows[tag] = {}
            continue

        n_found = len(rows_by_universe)
        n_missing = args.N_UNIVERSES - n_found
        if n_missing > 0:
            print(f"  {n_found}/{args.N_UNIVERSES} universes have readable output "
                  f"({n_missing} missing -- still running, or failed)")

        # ── per-scenario CSV (diagnostic) ──
        rows_list = list(rows_by_universe.values())
        csv_path = f"{outdir}/universe_report_{tag}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_list[0].keys()))
            writer.writeheader()
            writer.writerows(rows_list)
        print(f"  Saved: {csv_path}")

        median_total = np.median([r["n_total_detected"] for r in rows_list])
        local_best = max(rows_list, key=lambda r: r["n_total_detected"])
        print(f"  (diagnostic only) this scenario's own best universe would be "
              f"{local_best['universe']} (total={local_best['n_total_detected']}); "
              f"median across universes: {median_total:.1f}")

        emri_arr = np.array([r["n_emri_passed"] for r in rows_list], dtype=float)
        # Table 3 (summary_table.tex) is per-TDE-model, not summed across all 4
        # redback models -- use TABLE_TDE_MODEL's own passed-count fields here.
        ztf_arr = np.array([r[f"n_tde_ztf_{TABLE_TDE_MODEL}_passed"] for r in rows_list], dtype=float)
        lsst_arr = np.array([r[f"n_tde_lsst_{TABLE_TDE_MODEL}_passed"] for r in rows_list], dtype=float)
        scenario_stats[tag] = {
            "n": len(rows_list),
            "mean_emri": emri_arr.mean(), "var_emri": emri_arr.var(ddof=1) if len(emri_arr) > 1 else 0.0,
            "mean_ztf": ztf_arr.mean(), "var_ztf": ztf_arr.var(ddof=1) if len(ztf_arr) > 1 else 0.0,
            "mean_lsst": lsst_arr.mean(), "var_lsst": lsst_arr.var(ddof=1) if len(lsst_arr) > 1 else 0.0,
        }

        per_scenario_rows[tag] = rows_by_universe

    if not all_rows:
        print("\n[ERROR] No readable universe output for ANY scenario -- nothing to report.")
        return

    # ── combined per-universe totals, summed across ALL scenarios,
    #    kept SEPARATE per channel (EMRI / TDE-ZTF / TDE-LSST). These
    #    are used for the diagnostic plots/printouts, NOT the winner
    #    selection below -- see the note on why.
    combined_emri = {}
    combined_ztf = {}
    combined_lsst = {}
    scenarios_present = {}  # universe -> list of scenario tags that had data
    for u in range(args.N_UNIVERSES):
        e_total, z_total, l_total = 0, 0, 0
        present = []
        for tag, rows_by_universe in per_scenario_rows.items():
            if u in rows_by_universe:
                row = rows_by_universe[u]
                e_total += row["n_emri_passed"]
                z_total += row["n_tde_ztf_passed"]
                l_total += row["n_tde_lsst_passed"]
                present.append(tag)
        if present:
            combined_emri[u] = e_total
            combined_ztf[u] = z_total
            combined_lsst[u] = l_total
            scenarios_present[u] = present

    n_scenarios = len(SCENARIOS)
    if args.REQUIRE_ALL_SCENARIOS:
        complete_universes = [u for u in scenarios_present
                               if len(scenarios_present[u]) == n_scenarios]
        if not complete_universes:
            print(f"\n[ERROR] --REQUIRE_ALL_SCENARIOS is on but no universe has "
                  f"readable output for all {n_scenarios} (mbh, nuc, occupation) "
                  f"scenarios. Either wait for more jobs to finish, or rerun with "
                  f"--no-REQUIRE_ALL_SCENARIOS to allow partial universes "
                  f"(not recommended -- biases toward universes missing the "
                  f"hardest scenario).")
            return
        candidate_universes = complete_universes
    else:
        candidate_universes = list(scenarios_present.keys())

    # ── ONE overall winning universe, combined across TDE models (channel
    #    values here are n_tde_ztf_passed/n_tde_lsst_passed, i.e. summed
    #    over all 4 redback models already) -- used only for the "Best"
    #    column of the paper-facing summary/ratio LaTeX tables below, so
    #    those tables can quote a single shared UNIVERSE_MODEL across every
    #    scenario and channel. Same maximin-over-cells, sum-as-tiebreak
    #    logic as the per-TDE-model selection, just without the per-model
    #    split. ──
    def overall_scenario_channel_cells(u):
        """Detected counts per scenario and channel for universe u."""
        cells = []
        for tag, rows_by_universe in per_scenario_rows.items():
            row = rows_by_universe.get(u)
            if row is None:
                cells.append((-1, tag, "MISSING"))
                continue
            cells.append((row["n_emri_passed"], tag, "EMRI"))
            cells.append((row["n_tde_ztf_passed"], tag, "TDE-ZTF"))
            cells.append((row["n_tde_lsst_passed"], tag, "TDE-LSST"))
        return cells

    def overall_score(u):
        """Maximin score of universe u."""
        cells = overall_scenario_channel_cells(u)
        values = [c[0] for c in cells]
        bottleneck = min(cells, key=lambda c: c[0])
        return (bottleneck[0], sum(values), bottleneck)

    overall_best_u = max(candidate_universes, key=overall_score)
    overall_best_min, overall_best_total, overall_best_bottleneck = overall_score(overall_best_u)
    print(f"\n=== OVERALL winner (combined across TDE models, used for the 'Best' "
          f"column of summary_table.tex/ratio_table.tex): universe {overall_best_u} ===")
    print(f"  Bottleneck cell: scenario={overall_best_bottleneck[1]}, "
          f"channel={overall_best_bottleneck[2]}, value={overall_best_min}  "
          f"(sum over all cells = {overall_best_total})")

    # ── MAXIMIN selection, done SEPARATELY PER TDE MODEL --
    #    each TDE model (cooling_envelope, stream_stream_tde,
    #    tde_analytical, tde_fallback) gets its OWN winning universe,
    #    at (scenario x {EMRI, TDE-ZTF-<model>, TDE-LSST-<model>})
    #    granularity -- i.e. over n_scenarios*3 cells per model (e.g.
    #    8 scenarios x 3 = 24 cells), EMRI included in every model's
    #    selection since a winner still needs healthy EMRI counts.
    #
    #    Aggregating first (summing a channel across every scenario
    #    before taking the min) lets a big scenario mask a total
    #    outage in a small one. Fix: find the single worst
    #    (scenario, channel) cell for each universe FIRST, then
    #    maximize that -- guarantees every (mbh, nuc, occupation)
    #    scenario has nonzero EMRI, TDE-ZTF, and TDE-LSST individually
    #    for the winner (when any such universe exists).
    #
    #    This deliberately means EACH TDE model may pick a DIFFERENT
    #    winning UNIVERSE_MODEL -- unlike EMRI (one seed shared across
    #    all scenarios/occupation methods), the TDE side is scored and
    #    selected independently per redback model.
    def scenario_channel_cells(u, model):
        """Detected counts per scenario and channel for universe u and TDE model."""
        cells = []
        for tag, rows_by_universe in per_scenario_rows.items():
            row = rows_by_universe.get(u)
            if row is None:
                cells.append((-1, tag, "MISSING"))
                continue
            cells.append((row["n_emri_passed"], tag, "EMRI"))
            cells.append((row.get(f"n_tde_ztf_{model}_passed", 0), tag, f"TDE-ZTF-{model}"))
            cells.append((row.get(f"n_tde_lsst_{model}_passed", 0), tag, f"TDE-LSST-{model}"))
        return cells

    def score(u, model):
        """Maximin score of universe u for the TDE model."""
        cells = scenario_channel_cells(u, model)
        values = [c[0] for c in cells]
        bottleneck = min(cells, key=lambda c: c[0])
        return (bottleneck[0], sum(values), bottleneck)

    # winners[model] = (best_universe, best_min, best_total, best_bottleneck)
    winners = {}
    for m in TDE_MODELS:
        best_u = max(candidate_universes, key=lambda u: score(u, m))
        winners[m] = (best_u, *score(best_u, m))

    for m in TDE_MODELS:
        best_u, best_min, best_total, best_bottleneck = winners[m]
        if best_min <= 0:
            _, bn_tag, bn_channel = best_bottleneck
            print(f"\n[WARN][{m}] Even the best available universe has a ZERO/MISSING cell: "
                  f"scenario={bn_tag}, channel={bn_channel}. No candidate universe has "
                  f"nonzero counts in EVERY (scenario, channel) combination for this "
                  f"model -- consider raising --SNR_THRESHOLD/--DETECTION_THRESHOLD's "
                  f"leniency, raising T_OBS, or running more universes (larger "
                  f"N_UNIVERSES) to find one that does.")

        print(f"\n=== WINNER for TDE model '{m}' (maximin over {n_scenarios * 3} "
              f"scenario x channel cells): universe {best_u} ===")
        print(f"  Bottleneck cell: scenario={best_bottleneck[1]}, "
              f"channel={best_bottleneck[2]}, value={best_min}  (sum over all cells = {best_total})")
        for tag in per_scenario_rows:
            row = per_scenario_rows[tag].get(best_u)
            if row is None:
                print(f"  {tag:35s} -> NO DATA for this universe")
            else:
                print(f"  {tag:35s} -> EMRI={row['n_emri_passed']}, "
                      f"TDE-ZTF-{m}={row.get(f'n_tde_ztf_{m}_passed', 0)}, "
                      f"TDE-LSST-{m}={row.get(f'n_tde_lsst_{m}_passed', 0)}")

    print(f"\n  Winning universe per TDE model vs. each scenario's own distribution "
        f"(z = (value - mean) / std, computed from that scenario's {args.N_UNIVERSES}-universe scan):")
    winner_zscores = {}  # model -> tag -> {emri, ztf, lsst}
    for m in TDE_MODELS:
        best_u = winners[m][0]
        winner_zscores[m] = {}
        for tag in per_scenario_rows:
            row = per_scenario_rows[tag].get(best_u)
            s = scenario_stats.get(tag)
            if row is None or s is None:
                continue
            z_emri = zscore(row["n_emri_passed"], s["mean_emri"], s["var_emri"])
            z_ztf = zscore(row.get(f"n_tde_ztf_{m}_passed", 0), s["mean_ztf"], s["var_ztf"])
            z_lsst = zscore(row.get(f"n_tde_lsst_{m}_passed", 0), s["mean_lsst"], s["var_lsst"])
            winner_zscores[m][tag] = {"z_emri": z_emri, "z_ztf": z_ztf, "z_lsst": z_lsst}
        print(f"  [{m}] (universe {best_u}):")
        for tag, z in winner_zscores[m].items():
            print(f"    {tag:35s} -> z(EMRI)={z['z_emri']:+6.2f}  z(TDE-ZTF)={z['z_ztf']:+6.2f}  "
                f"z(TDE-LSST)={z['z_lsst']:+6.2f}")

    # ── per-scenario comparison plots (diagnostic), marking each TDE
    #    model's own winner ──
    _model_line_styles = ["--", ":", "-.", (0, (3, 1, 1, 1))]
    _model_colors = plt.get_cmap("Set1")(np.linspace(0, 1, len(TDE_MODELS)))
    for tag, rows_by_universe in per_scenario_rows.items():
        if not rows_by_universe:
            continue
        universes_sorted = sorted(rows_by_universe.values(), key=lambda r: r["universe"])
        u_vals = [r["universe"] for r in universes_sorted]
        emri_vals = [r["n_emri_passed"] for r in universes_sorted]
        ztf_vals = [r["n_tde_ztf_passed"] for r in universes_sorted]
        lsst_vals = [r["n_tde_lsst_passed"] for r in universes_sorted]

        fig, ax = plt.subplots(figsize=(max(8, len(u_vals) * 0.3), 5))
        width = 0.27
        x = np.arange(len(u_vals))
        ax.bar(x - width, emri_vals, width, label="EMRI", alpha=0.8)
        ax.bar(x, ztf_vals, width, label="TDE (ZTF, all models)", alpha=0.8)
        ax.bar(x + width, lsst_vals, width, label="TDE (LSST, all models)", alpha=0.8)
        for i, m in enumerate(TDE_MODELS):
            best_u = winners[m][0]
            if best_u in u_vals:
                best_idx = u_vals.index(best_u)
                ax.axvline(best_idx, color=_model_colors[i], ls=_model_line_styles[i % len(_model_line_styles)],
                           lw=1.5, label=f"{m} winner = universe {best_u}")
        ax.set_xticks(x)
        ax.set_xticklabels(u_vals, rotation=90 if len(u_vals) > 20 else 0)
        ax.set_xlabel("Universe (UNIVERSE_MODEL)")
        ax.set_ylabel("Detected sources")
        ax.set_title(f"Detected sources by universe -- {tag}\n"
                     f"(vertical lines = each TDE model's own winning universe)")
        ax.legend(fontsize=7)
        plt.tight_layout()
        plot_path = f"{outdir}/universe_comparison_{tag}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Saved: {plot_path}")

    # ── the actual selection plot: per-CHANNEL totals per universe,
    #    so it's visible why a universe with a huge sum but a zero in
    #    one channel does NOT win. ──
    u_sorted = sorted(scenarios_present.keys())
    emri_sorted = [combined_emri[u] for u in u_sorted]
    ztf_sorted = [combined_ztf[u] for u in u_sorted]
    lsst_sorted = [combined_lsst[u] for u in u_sorted]
    complete_mask = [len(scenarios_present[u]) == n_scenarios for u in u_sorted]

    fig, ax = plt.subplots(figsize=(max(8, len(u_sorted) * 0.35), 5))
    width = 0.27
    x = np.arange(len(u_sorted))
    ax.bar(x - width, emri_sorted, width, label="EMRI", alpha=0.85)
    ax.bar(x, ztf_sorted, width, label="TDE (ZTF, all models)", alpha=0.85)
    ax.bar(x + width, lsst_sorted, width, label="TDE (LSST, all models)", alpha=0.85)
    # gray out incomplete universes (missing >=1 scenario) so it's clear
    # why they weren't eligible to win
    for i, complete in enumerate(complete_mask):
        if not complete:
            ax.axvspan(i - 0.5, i + 0.5, color="lightgray", alpha=0.4, zorder=0)
    for i, m in enumerate(TDE_MODELS):
        best_u = winners[m][0]
        if best_u in u_sorted:
            best_idx = u_sorted.index(best_u)
            ax.axvline(best_idx, color=_model_colors[i], ls=_model_line_styles[i % len(_model_line_styles)],
                       lw=1.5, label=f"{m} winner = universe {best_u}")
    ax.set_xticks(x)
    ax.set_xticklabels(u_sorted, rotation=90 if len(u_sorted) > 20 else 0)
    ax.set_xlabel("Universe (UNIVERSE_MODEL)")
    ax.set_ylabel(f"Combined total per channel, summed across all {n_scenarios} scenarios")
    ax.set_title("Per-channel detections by universe -- each TDE model's winner maximizes "
                 "the WORST\nper-model channel, not the sum (shaded = missing >=1 scenario, excluded)")
    ax.legend(fontsize=7)
    plt.tight_layout()
    combined_plot_path = f"{outdir}/combined_totals_by_universe.png"
    plt.savefig(combined_plot_path, dpi=150)
    plt.close()
    print(f"\nSaved: {combined_plot_path}")

    # ── copy each TDE model's OWN winning universe's files for every
    #    scenario, with a max_detected_<model>_ prefix. EMRI files are
    #    copied per model too (same source across models when the
    #    winning universe happens to coincide, harmless duplication
    #    otherwise) so each model's max_detected_ set is self-contained. ──
    for m in TDE_MODELS:
        best_u = winners[m][0]
        print(f"\nCopying universe {best_u}'s files for every scenario, TDE model '{m}' "
              f"(max_detected_{m}_ prefix):")
        for mbh, nuc, occ in SCENARIOS:
            tag = scenario_tag(mbh, nuc, occ)
            u = best_u
            src_paths = {
                "EMRI_SNRs": emri_snr_path(base, mbh, nuc, occ, args.merger_model, u),
                f"TDE_SNR_ZTF_{m}": tde_snr_path(base, "ZTF", mbh, nuc, occ, args.merger_model, m, u),
                f"TDE_SNR_LSST_{m}": tde_snr_path(base, "LSST", mbh, nuc, occ, args.merger_model, m, u),
            }
            for label, src in src_paths.items():
                if not os.path.exists(src):
                    print(f"  [warn] {tag}/{label} missing for winning universe {u}, "
                          f"can't copy: {src}")
                    continue
                fname = os.path.basename(src)
                dst = f"{base}/max_detected_{m}_{fname}"
                shutil.copy2(src, dst)
                print(f"  {tag}/{label:15s} {src} -> {dst}")

    # ── combined CSV across all scenarios (diagnostic) ──
    combined_csv_path = f"{outdir}/universe_report_all.csv"
    with open(combined_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved: {combined_csv_path}")

    # ══════════════════════════════════════════════════════════════════
    # Per-scenario, per-channel mean and variance -- printed here;
    # written to the single consolidated report file further below,
    # alongside the ratio stats (built after they're computed).
    # ══════════════════════════════════════════════════════════════════
    print("\n=== Per-scenario, per-channel detected counts: mean, variance, "
          "CV% (std/mean x 100 -- spread relative to the mean) ===")
    counts_table_lines = [
        f"{'scenario':35s} {'N':>4s}  "
        f"{'EMRI mean':>10s} {'EMRI var':>10s} {'EMRI CV%':>10s}  "
        f"{'TDE-ZTF mean':>12s} {'TDE-ZTF var':>12s} {'TDE-ZTF CV%':>12s}  "
        f"{'TDE-LSST mean':>13s} {'TDE-LSST var':>13s} {'TDE-LSST CV%':>13s}"
    ]
    # ── model-name key table (Table~\ref{tab:model_names}) -- paste once,
    #    referenced by both summary_table.tex and ratio_table.tex below. ──
    model_names_lines = [
        "% Auto-generated by report_universes.py -- paste into the paper.",
        "\\begin{table*}",
        "\\centering",
        "\\footnotesize",
        "\\caption{Mapping between the compact model labels used in "
        "Tables~\\ref{tab:counts_summary} and~\\ref{tab:ratio_summary} and the "
        "eight population scenarios defined in Sec.~\\ref{subsec:initial_conditions}, "
        "given by the cross product of the \\ac{MBH}-mass assignment "
        "($M_\\bullet$--$\\sigma$ vs.\\ $M_\\bullet$--$M_\\star$), the "
        "nucleation-fraction model \\citep{Hannah:2024dmr, Neumayer:2020gno}, and the "
        "occupation-fraction model (uniform, $f_\\mathrm{occ}=1$, vs.\\ varying, "
        "Eq.~\\ref{eq:focc}).}",
        "\\label{tab:model_names}",
        "\\begin{tabular}{l l l l}",
        "\\hline",
        "Model name & MBH-mass relation & Nucleation model & Occupation model \\\\",
        "\\hline",
    ]
    for name in MODEL_ORDER:
        (mbh, nuc, occ), = [k for k, v in SCENARIO_NAME_MAP.items() if v == name]
        model_names_lines.append(
            f"{name} & {mbh_mass_latex(mbh)} & {nucleation_latex(nuc)} & "
            f"{occupation_latex(occ)} \\\\"
        )
    model_names_lines += ["\\hline", "\\end{tabular}", "\\end{table*}"]

    model_names_tex_path = f"{outdir}/model_names.tex"
    with open(model_names_tex_path, "w") as f:
        f.write("\n".join(model_names_lines) + "\n")
    print(f"Saved: {model_names_tex_path}")

    # ── counts summary table (Table~\ref{tab:counts_summary}): one row
    #    per model (M1..M8), one {mean, std, CV%, Best(z)} block per
    #    channel. "Best" is this scenario's own value in the single
    #    overall_best_u universe picked above -- the SAME universe is
    #    quoted in every row/channel, matching the paper's "one shared
    #    realization for every comparison" framing. ──
    latex_lines = [
        "% Auto-generated by report_universes.py -- paste into a LaTeX table body.",
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\caption{Mean detected source count across $N=" + str(args.N_UNIVERSES) +
        "$ independent realisations for each of the eight population scenarios "
        "(see Table~\\ref{tab:model_names} for the model-name key), by detection "
        f"channel. TDE counts use the \\texttt{{{TABLE_TDE_MODEL}}} light-curve model "
        "only. For each channel, we report the realisation mean $\\mu$, standard "
        "deviation $\\sigma$, and coefficient of variation (CV), the spread relative "
        "to the mean. The \\emph{Best} column gives the detected count in the single "
        f"realisation (UNIVERSE\\_MODEL = {overall_best_u}) selected for all "
        "downstream cross-scenario comparisons, together with its offset from the "
        "scenario mean in units of $\\sigma$, $z = (\\mathrm{Best}-\\mu)/\\sigma$. "
        "$|z|\\lesssim1$ indicates the selected realisation is statistically "
        "unremarkable for that channel; $|z|\\gtrsim2$ flags a channel where the "
        "shared realisation is a comparative outlier and downstream constraints from "
        "that channel should be read with this in mind.}",
        "\\label{tab:counts_summary}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{l c r ccc r  ccc r  ccc r}",
        "\\hline",
        "& & & \\multicolumn{4}{c}{EMRI} & \\multicolumn{4}{c}{TDE (ZTF)} & "
        "\\multicolumn{4}{c}{TDE (LSST)} \\\\",
        "Model & $\\mu$ & $\\sigma$ & CV\\% & Best ($z$) & $\\mu$ & $\\sigma$ & "
        "CV\\% & Best ($z$) & $\\mu$ & $\\sigma$ & CV\\% & Best ($z$) \\\\",
        "\\hline",
    ]

    def best_and_z(u, key, mean, var):
        """Best value and its z-score."""
        row = per_scenario_rows.get(tag, {}).get(u)
        if row is None:
            return None, float("nan")
        val = row[key]
        return val, zscore(val, mean, var)

    for name in MODEL_ORDER:
        mbh, nuc, occ = NAME_TO_SCENARIO[name]
        tag = scenario_tag(mbh, nuc, occ)
        s = scenario_stats.get(tag)
        if s is None:
            print(f"  {tag:35s}  NO DATA")
            continue
        std_emri = math.sqrt(s["var_emri"])
        std_ztf = math.sqrt(s["var_ztf"])
        std_lsst = math.sqrt(s["var_lsst"])
        cv_emri = 100 * std_emri / s["mean_emri"] if s["mean_emri"] else float("nan")
        cv_ztf = 100 * std_ztf / s["mean_ztf"] if s["mean_ztf"] else float("nan")
        cv_lsst = 100 * std_lsst / s["mean_lsst"] if s["mean_lsst"] else float("nan")

        best_emri, z_emri = best_and_z(overall_best_u, "n_emri_passed", s["mean_emri"], s["var_emri"])
        best_ztf, z_ztf = best_and_z(overall_best_u, f"n_tde_ztf_{TABLE_TDE_MODEL}_passed", s["mean_ztf"], s["var_ztf"])
        best_lsst, z_lsst = best_and_z(overall_best_u, f"n_tde_lsst_{TABLE_TDE_MODEL}_passed", s["mean_lsst"], s["var_lsst"])

        line = (f"  {tag:35s} {s['n']:>4d}  "
                f"{s['mean_emri']:>10.3f} {std_emri:>10.3f} {cv_emri:>9.2f}%  "
                f"{s['mean_ztf']:>12.3f} {std_ztf:>12.3f} {cv_ztf:>11.2f}%  "
                f"{s['mean_lsst']:>13.3f} {std_lsst:>13.3f} {cv_lsst:>12.2f}%")
        print(line)
        counts_table_lines.append(line)

        def fmt_best(best, z):
            """Format the best value with its z-score."""
            if best is None:
                return "\\text{--}"
            zs = "\\text{NaN}" if z != z else f"{z:+.2f}"
            return f"{best:g} ({zs})"

        latex_lines.append(
            f"{name} & {s['mean_emri']:.2f} & {std_emri:.2f} & {cv_emri:.1f}\\% & "
            f"{fmt_best(best_emri, z_emri)} & "
            f"{s['mean_ztf']:.2f} & {std_ztf:.2f} & {cv_ztf:.1f}\\% & "
            f"{fmt_best(best_ztf, z_ztf)} & "
            f"{s['mean_lsst']:.1f} & {std_lsst:.1f} & {cv_lsst:.1f}\\% & "
            f"{fmt_best(best_lsst, z_lsst)} \\\\"
        )
    latex_lines += ["\\hline", "\\end{tabular}", "\\end{table*}"]

    table_tex_path = f"{outdir}/summary_table.tex"
    with open(table_tex_path, "w") as f:
        f.write("\n".join(latex_lines) + "\n")
    print(f"Saved: {table_tex_path}")

    # ══════════════════════════════════════════════════════════════════
    # combined scatter -- every scenario (color) x 3 channels
    # (marker) x all realizations, in one plot, log-y.
    # ══════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(14, 8))
    handles = []
    for mbh, nuc, occ in SCENARIOS:
        tag = scenario_tag(mbh, nuc, occ)
        rows_by_universe = per_scenario_rows.get(tag, {})
        if not rows_by_universe:
            continue
        color = SCENARIO_COLORS[(mbh, nuc, occ)]
        us = sorted(rows_by_universe.keys())
        for channel, key in [("EMRI", "n_emri_passed"), ("TDE-ZTF", "n_tde_ztf_passed"),
                              ("TDE-LSST", "n_tde_lsst_passed")]:
            vals = np.array([rows_by_universe[u][key] for u in us], dtype=float)
            marker = CHANNEL_MARKERS[channel]
            h = ax.scatter(us, vals, color=color, marker=marker, s=28, alpha=0.75,
                           label=f"{tag} -- {channel}")
            handles.append(h)
            pos_vals = vals[vals > 0]
            if len(pos_vals) > 0:
                ax.axhline(pos_vals.mean(), color=color, linestyle="--", linewidth=1,
                          alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("Universe (UNIVERSE_MODEL)")
    ax.set_ylabel("Detected sources (log scale)")
    ax.set_title("Every scenario (mbh x nucleation x occupation) x all channels x all realizations\n"
                 "(color = scenario, marker = channel; dashed lines = per-series mean)")
    color_handles = [plt.Line2D([], [], color=SCENARIO_COLORS[(mbh, nuc, occ)], marker='o',
                                linestyle='', markersize=8, label=scenario_tag(mbh, nuc, occ))
                     for mbh, nuc, occ in SCENARIOS]
    marker_handles = [plt.Line2D([], [], color='black', marker=CHANNEL_MARKERS[ch],
                                 linestyle='', markersize=8, label=ch)
                      for ch in ["EMRI", "TDE-ZTF", "TDE-LSST"]]
    leg1 = ax.legend(handles=color_handles, title="Scenario (color)", loc="upper left",
                     bbox_to_anchor=(1.01, 1.0), fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=marker_handles, title="Channel (marker)", loc="upper left",
             bbox_to_anchor=(1.01, 0.4))
    plt.tight_layout()
    combined_scatter_path = f"{outdir}/all_scenarios_all_channels_scatter.png"
    plt.savefig(combined_scatter_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {combined_scatter_path}")

    # ══════════════════════════════════════════════════════════════════
    # ratio scatter -- EMRI/TDE-ZTF and EMRI/TDE-LSST per scenario,
    # per realization. Points where the TDE denominator is 0 are
    # dropped (would be inf/undefined), counted and reported.
    # ══════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(14, 8))
    n_skipped_ztf_total = 0
    n_skipped_lsst_total = 0
    ratio_stats = {}  # tag -> {mean_ratio_ztf, mean_ratio_lsst, n_ztf, n_lsst}

    for mbh, nuc, occ in SCENARIOS:
        tag = scenario_tag(mbh, nuc, occ)
        rows_by_universe = per_scenario_rows.get(tag, {})
        if not rows_by_universe:
            continue
        color = SCENARIO_COLORS[(mbh, nuc, occ)]
        us = sorted(rows_by_universe.keys())

        emri_vals = np.array([rows_by_universe[u]["n_emri_passed"] for u in us], dtype=float)
        ztf_vals = np.array([rows_by_universe[u]["n_tde_ztf_passed"] for u in us], dtype=float)
        lsst_vals = np.array([rows_by_universe[u]["n_tde_lsst_passed"] for u in us], dtype=float)

        ztf_ok = ztf_vals > 0
        lsst_ok = lsst_vals > 0
        n_skipped_ztf = (~ztf_ok).sum()
        n_skipped_lsst = (~lsst_ok).sum()
        n_skipped_ztf_total += n_skipped_ztf
        n_skipped_lsst_total += n_skipped_lsst

        us_arr = np.array(us)
        ratio_ztf = emri_vals[ztf_ok] / ztf_vals[ztf_ok]
        ratio_lsst = emri_vals[lsst_ok] / lsst_vals[lsst_ok]

        ax.scatter(us_arr[ztf_ok], ratio_ztf, color=color, marker=CHANNEL_MARKERS["TDE-ZTF"],
                  s=28, alpha=0.75, label=f"{tag} -- EMRI/TDE-ZTF")
        ax.scatter(us_arr[lsst_ok], ratio_lsst, color=color, marker=CHANNEL_MARKERS["TDE-LSST"],
                  s=28, alpha=0.75, label=f"{tag} -- EMRI/TDE-LSST")

        if len(ratio_ztf) > 0:
            ax.axhline(ratio_ztf.mean(), color=color, linestyle="--", linewidth=1, alpha=0.6)
        if len(ratio_lsst) > 0:
            ax.axhline(ratio_lsst.mean(), color=color, linestyle=":", linewidth=1, alpha=0.6)

        ratio_stats[tag] = {
            "mean_ratio_ztf": ratio_ztf.mean() if len(ratio_ztf) > 0 else float("nan"),
            "var_ratio_ztf": ratio_ztf.var(ddof=1) if len(ratio_ztf) > 1 else 0.0,
            "n_ztf": len(ratio_ztf),
            "mean_ratio_lsst": ratio_lsst.mean() if len(ratio_lsst) > 0 else float("nan"),
            "var_ratio_lsst": ratio_lsst.var(ddof=1) if len(ratio_lsst) > 1 else 0.0,
            "n_lsst": len(ratio_lsst),
        }

    ax.set_xlabel("Universe (UNIVERSE_MODEL)")
    ax.set_ylabel("EMRI / TDE ratio")
    ax.set_title("EMRI:TDE ratio by realization, every scenario\n"
                 "(square marker = EMRI/TDE-ZTF, triangle = EMRI/TDE-LSST; "
                 "dashed/dotted lines = per-series mean)")
    ax.set_yscale("log")
    color_handles = [plt.Line2D([], [], color=SCENARIO_COLORS[(mbh, nuc, occ)], marker='o',
                                linestyle='', markersize=8, label=scenario_tag(mbh, nuc, occ))
                     for mbh, nuc, occ in SCENARIOS]
    marker_handles = [
        plt.Line2D([], [], color='black', marker=CHANNEL_MARKERS["TDE-ZTF"], linestyle='',
                  markersize=8, label="EMRI/TDE-ZTF"),
        plt.Line2D([], [], color='black', marker=CHANNEL_MARKERS["TDE-LSST"], linestyle='',
                  markersize=8, label="EMRI/TDE-LSST"),
    ]
    leg1 = ax.legend(handles=color_handles, title="Scenario (color)", loc="upper left",
                     bbox_to_anchor=(1.01, 1.0), fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=marker_handles, title="Ratio (marker)", loc="upper left",
             bbox_to_anchor=(1.01, 0.4))
    plt.tight_layout()
    ratio_plot_path = f"{outdir}/emri_tde_ratio_scatter.png"
    plt.savefig(ratio_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {ratio_plot_path}")
    if n_skipped_ztf_total > 0 or n_skipped_lsst_total > 0:
        print(f"  (dropped {n_skipped_ztf_total} EMRI/TDE-ZTF points and "
              f"{n_skipped_lsst_total} EMRI/TDE-LSST points where the TDE "
              f"denominator was 0 -- these would be inf/undefined)")

    # ══════════════════════════════════════════════════════════════════
    # Per-scenario EMRI/TDE ratio: mean, variance. No interpretation.
    # ══════════════════════════════════════════════════════════════════
    print("\n=== Per-scenario EMRI/TDE ratio: mean, variance ===")
    ratio_table_lines = [
        f"{'scenario':35s} {'EMRI/ZTF mean':>14s} {'EMRI/ZTF var':>14s} {'n':>4s}  "
        f"{'EMRI/LSST mean':>14s} {'EMRI/LSST var':>14s} {'n':>4s}"
    ]
    def best_ratio_and_z(tag, num_key, den_key, mean, var):
        """EMRI/TDE ratio realised in the single shared overall_best_u
        universe for this scenario, plus its z-score vs. that scenario's
        own ratio distribution. Returns (None, nan) if the TDE
        denominator is 0 there (ratio undefined) or the row is missing."""
        row = per_scenario_rows.get(tag, {}).get(overall_best_u)
        if row is None or row[den_key] == 0:
            return None, float("nan")
        val = row[num_key] / row[den_key]
        return val, zscore(val, mean, var)

    ratio_latex_lines = [
        "% Auto-generated by report_universes.py -- paste into a LaTeX table body.",
        "% <.> denotes a mean taken across realizations (universes), not a single value.",
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\caption{Mean EMRI-to-TDE detection ratio, $\\langle \\mathrm{EMRI/TDE}\\rangle$, "
        "across the same realizations as Table~\\ref{tab:counts_summary} (see "
        "Table~\\ref{tab:model_names} for the model-name key). Realisations with zero "
        "detected TDEs in the relevant survey are excluded from that ratio's mean "
        "(denominator undefined), which is why $N$ is reduced for some entries. The "
        "\\emph{Best} column is the ratio realised in "
        f"UNIVERSE\\_MODEL~=~{overall_best_u}, with its offset from the scenario mean "
        "in units of $\\sigma=\\sqrt{\\mathrm{Var}}$.}",
        "\\label{tab:ratio_summary}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{l ccc r  ccc r}",
        "\\hline",
        "& \\multicolumn{4}{c}{EMRI/TDE-ZTF} & \\multicolumn{4}{c}{EMRI/TDE-LSST} \\\\",
        "Model & $\\mu$ & $\\sigma$ & $N$ & Best ($z$) & $\\mu$ & $\\sigma$ & $N$ & "
        "Best ($z$) \\\\",
        "\\hline",
    ]

    def fmt_best_ratio(best, z):
        """Format the best ratio with its z-score."""
        if best is None:
            return "\\text{--}"
        zs = "\\text{NaN}" if z != z else f"{z:+.2f}"
        return f"${sci_latex(best)}$ ({zs})"

    for name in MODEL_ORDER:
        mbh, nuc, occ = NAME_TO_SCENARIO[name]
        tag = scenario_tag(mbh, nuc, occ)
        rs = ratio_stats.get(tag)
        if rs is None:
            print(f"  {tag:35s}  NO DATA")
            continue
        line = (f"  {tag:35s} "
                f"{rs['mean_ratio_ztf']:>14.6e} {rs['var_ratio_ztf']:>14.6e} {rs['n_ztf']:>4d}  "
                f"{rs['mean_ratio_lsst']:>14.6e} {rs['var_ratio_lsst']:>14.6e} {rs['n_lsst']:>4d}")
        print(line)
        ratio_table_lines.append(line)

        best_ztf, z_ztf = best_ratio_and_z(tag, "n_emri_passed", "n_tde_ztf_passed",
                                            rs["mean_ratio_ztf"], rs["var_ratio_ztf"])
        best_lsst, z_lsst = best_ratio_and_z(tag, "n_emri_passed", "n_tde_lsst_passed",
                                              rs["mean_ratio_lsst"], rs["var_ratio_lsst"])

        ratio_latex_lines.append(
            f"{name} & "
            f"${sci_latex(rs['mean_ratio_ztf'])}$ & ${sci_latex(math.sqrt(rs['var_ratio_ztf']))}$ & "
            f"{rs['n_ztf']} & {fmt_best_ratio(best_ztf, z_ztf)} & "
            f"${sci_latex(rs['mean_ratio_lsst'])}$ & ${sci_latex(math.sqrt(rs['var_ratio_lsst']))}$ & "
            f"{rs['n_lsst']} & {fmt_best_ratio(best_lsst, z_lsst)} \\\\"
        )
    ratio_latex_lines += ["\\hline", "\\end{tabular}", "\\end{table*}"]

    ratio_tex_path = f"{outdir}/ratio_table.tex"
    with open(ratio_tex_path, "w") as f:
        f.write("\n".join(ratio_latex_lines) + "\n")
    print(f"Saved: {ratio_tex_path}")

    # ── single consolidated text file: counts table + ratio table ──
    full_report_lines = [
        "=== Per-scenario, per-channel detected counts: mean, variance ===",
        "",
    ] + counts_table_lines + [
        "",
        "=== Per-scenario EMRI/TDE ratio: mean, variance ===",
        "(points with 0 TDE in the denominator excluded from the ratio, not treated as inf)",
        "",
    ] + ratio_table_lines
    full_report_path = f"{outdir}/full_report.txt"
    with open(full_report_path, "w") as f:
        f.write("\n".join(full_report_lines) + "\n")
    print(f"\nSaved: {full_report_path}")

    # ══════════════════════════════════════════════════════════════════
    # optional model selection given an observed TDE count.
    # ══════════════════════════════════════════════════════════════════
    if args.OBSERVED_TDE_ZTF is not None and args.OBSERVED_TDE_LSST is not None:
        print(f"\n=== MODEL SELECTION given observed TDE-ZTF={args.OBSERVED_TDE_ZTF}, "
              f"TDE-LSST={args.OBSERVED_TDE_LSST} ===")

        def poisson_logpmf(k, lam):
            """Poisson log pmf."""
            if lam <= 0:
                return float("-inf") if k > 0 else 0.0
            return k * math.log(lam) - lam - math.lgamma(k + 1)

        model_scores = {}
        for mbh, nuc, occ in SCENARIOS:
            tag = scenario_tag(mbh, nuc, occ)
            s = scenario_stats.get(tag)
            if s is None:
                continue
            ll_ztf = poisson_logpmf(args.OBSERVED_TDE_ZTF, s["mean_ztf"])
            ll_lsst = poisson_logpmf(args.OBSERVED_TDE_LSST, s["mean_lsst"])
            model_scores[tag] = ll_ztf + ll_lsst
            print(f"  {tag:35s}: logL(TDE-ZTF)={ll_ztf:8.3f}  logL(TDE-LSST)={ll_lsst:8.3f}  "
                  f"joint logL={model_scores[tag]:8.3f}")

        favored_tag = max(model_scores, key=model_scores.get)
        rs = ratio_stats[favored_tag]
        pred_emri_ztf = rs["mean_ratio_ztf"] * args.OBSERVED_TDE_ZTF if not math.isnan(rs["mean_ratio_ztf"]) else float("nan")
        pred_emri_lsst = rs["mean_ratio_lsst"] * args.OBSERVED_TDE_LSST if not math.isnan(rs["mean_ratio_lsst"]) else float("nan")

        print(f"\n  FAVORED MODEL: {favored_tag}  (joint logL = {model_scores[favored_tag]:.3f})")
        print(f"  Predicted EMRI count (scaling this model's mean EMRI/TDE ratio to "
              f"the observed TDE counts):")
        print(f"    from TDE-ZTF ratio:  {pred_emri_ztf:.2f}")
        print(f"    from TDE-LSST ratio: {pred_emri_lsst:.2f}")
        print(f"  NOTE: this is a simple ratio-scaling point estimate, not a full "
              f"posterior -- treat as a rough prediction, not a calibrated interval. "
              f"For a real interval, use scenario_stats/ratio_stats' std values or "
              f"refit properly.")

        model_selection_path = f"{outdir}/model_selection.txt"
        with open(model_selection_path, "w") as f:
            f.write(f"Observed: TDE-ZTF={args.OBSERVED_TDE_ZTF}, TDE-LSST={args.OBSERVED_TDE_LSST}\n\n")
            for tag, score in model_scores.items():
                f.write(f"{tag:35s} joint logL = {score:.3f}\n")
            f.write(f"\nFavored model: {favored_tag}\n")
            f.write(f"Predicted EMRI (from TDE-ZTF ratio):  {pred_emri_ztf:.2f}\n")
            f.write(f"Predicted EMRI (from TDE-LSST ratio): {pred_emri_lsst:.2f}\n")
        print(f"  Saved: {model_selection_path}")
    else:
        print("\n(Skipping model-selection step -- pass both --OBSERVED_TDE_ZTF and "
              "--OBSERVED_TDE_LSST to run it.)")

    # ── final summary, one section per TDE model (each with its own
    #    winning universe) plus one combined index file ──
    index_lines = [
        "=== WINNING UNIVERSE PER TDE MODEL ===",
        "Each TDE model (cooling_envelope, stream_stream_tde, tde_analytical, "
        "tde_fallback) is selected independently: maximin over all "
        f"{n_scenarios * 3} (scenario x {{EMRI, TDE-ZTF-<model>, TDE-LSST-<model>}}) "
        "cells, sum as tiebreaker. EMRI is included in every model's selection "
        "criterion (it doesn't depend on TDE model, but a winner still needs "
        "healthy EMRI counts). Different models may end up with DIFFERENT "
        "winning UNIVERSE_MODEL values -- see model_selection_report/<model>.",
        "",
    ]
    for m in TDE_MODELS:
        best_u, best_min, best_total, best_bottleneck = winners[m]
        model_lines = [f"=== WINNING UNIVERSE for TDE model '{m}': {best_u} ==="]
        model_lines.append(
            f"Selection criterion: maximin over all {n_scenarios * 3} "
            f"(scenario x channel) cells ({n_scenarios} scenarios [= "
            f"{len(MBH_NUC_SCENARIOS)} mbh/nuc combos x {len(OCCUPATION_METHODS)} "
            f"occupation methods] x {{EMRI, TDE-ZTF-{m}, TDE-LSST-{m}}}), sum as tiebreaker."
        )
        model_lines.append(
            f"Bottleneck cell: scenario={best_bottleneck[1]}, channel={best_bottleneck[2]}, "
            f"value={best_min}  (sum over all cells = {best_total})"
        )
        model_lines.append("")
        for tag in per_scenario_rows:
            row = per_scenario_rows[tag].get(best_u)
            if row is None:
                model_lines.append(f"{tag:35s} -> NO DATA for this universe")
            else:
                model_lines.append(
                    f"{tag:35s} -> EMRI={row['n_emri_passed']}, "
                    f"TDE-ZTF-{m}={row.get(f'n_tde_ztf_{m}_passed', 0)}, "
                    f"TDE-LSST-{m}={row.get(f'n_tde_lsst_{m}_passed', 0)}"
                )
        model_lines.append("")
        model_lines.append("Winner's z-score vs. each scenario's own distribution "
                            "(diagnostic; NOT part of the selection criterion):")
        for tag, z in winner_zscores[m].items():
            model_lines.append(
                f"  {tag:35s} z(EMRI)={z['z_emri']:+.2f}  z(TDE-ZTF)={z['z_ztf']:+.2f}  "
                f"z(TDE-LSST)={z['z_lsst']:+.2f}"
            )
        model_text = "\n".join(model_lines)
        print("\n" + model_text)

        model_summary_path = f"{outdir}/best_universe_{m}.txt"
        with open(model_summary_path, "w") as f:
            f.write(model_text + "\n")
        print(f"\nSaved: {model_summary_path}")

        index_lines.append(f"{m:20s} -> universe {best_u}  (bottleneck={best_min}, "
                            f"see best_universe_{m}.txt)")

    index_text = "\n".join(index_lines)
    print("\n" + index_text)
    summary_path = f"{outdir}/best_universe.txt"
    with open(summary_path, "w") as f:
        f.write(index_text + "\n")
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()