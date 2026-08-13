#!/usr/bin/env python
"""
Scans the EMRI/TDE SNR-detection output for EVERY (mbh_mass_method,
nucleation_method, occupation_method, universe) combo -- ALL scenarios
AND ALL occupation methods together, in ONE combined scan -- and picks
ONE winning UNIVERSE_MODEL that maximizes the worst individual
(scenario x occupation x channel) cell (maximin).

THIS IS INTENTIONALLY COMBINED, NOT PER-OCCUPATION-METHOD: the winning
universe must be the same UNIVERSE_MODEL used for every downstream
comparison -- across mbh/nucleation scenarios AND across occupation
methods. Running this once per occupation method (as an earlier
version of this script did) could pick a DIFFERENT winning universe
per occupation method, defeating the entire "same realization for
every fair comparison" design. So there is exactly one report, one
winner, matching the single combined REPORT_UNIVERSES job in
make_data_gen_dag.py.

This is deliberately NOT "best universe per scenario" either:
UNIVERSE_MODEL is the seed for the whole realization, so a fair
downstream comparison across all (mbh_mass_method, nucleation_method,
occupation_method) combinations needs them all run on the SAME
universe, not cherry-picked independently per combination.

ASSUMPTION (single point of change if wrong): every stage's output
filename gets "_{occupation_method}_{U}" appended after the existing
"_{mbh_mass_method}_{nucleation_method}" scenario suffix, matching
what make_data_gen_dag.py passes via OCCUPATION_METHODS/UNIVERSE_MODEL.
If your actual suffix pattern differs, only emri_snr_path()/
tde_snr_path() below need to change -- nothing else in this script
assumes a particular pattern.

Reads (per mbh/nuc scenario, per occupation method, per universe):
    EMRI_SNRs_data_{mbh}_{nuc}_{occ}_{U}.h5      .attrs['n_events_total'/'n_events_passed']
    TDE_SNRs_ZTF_{mbh}_{nuc}_{occ}_{U}.h5        .attrs['n_events_total'/'n_events_passed']
    TDE_SNRs_LSST_{mbh}_{nuc}_{occ}_{U}.h5       .attrs['n_events_total'/'n_events_passed']

Writes (single combined set, under {outdir}/):
    universe_report_{mbh}_{nuc}_{occ}.csv, universe_report_all.csv,
    universe_comparison_{mbh}_{nuc}_{occ}.png, combined_totals_by_universe.png,
    all_scenarios_all_channels_scatter.png, emri_tde_ratio_scatter.png,
    summary_table.tex, ratio_table.tex, full_report.txt,
    model_selection.txt (only if --OBSERVED_TDE_ZTF/LSST given),
    best_universe.txt
    {base}/max_detected_{mbh}_{nuc}_{occ}_*.h5   (winning universe's files,
                                                    every scenario x occupation
                                                    combo, copied with a
                                                    max_detected_ prefix)

Usage:
    python report_universes.py --GALAXIES 100000 --N_UNIVERSES 100
"""

import os, sys, argparse, csv, shutil, math
sys.path.insert(0, os.path.abspath('./astrophysical_setup'))
sys.path.insert(0, os.path.abspath('./'))
from utils import ASTROPHYSICAL_DATA_DIR

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
def emri_snr_path(base, mbh, nuc, occ, u):
    return f"{base}/EMRI_SNRs_data_{mbh}_{nuc}_{occ}_{u}.h5"

def tde_snr_path(base, survey, mbh, nuc, occ, u):
    return f"{base}/TDE_SNRs_{survey}_{mbh}_{nuc}_{occ}_{u}.h5"

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


def main():
    parser = argparse.ArgumentParser(description="Pick one UNIVERSE_MODEL, maximizing the worst "
                                      "(mbh x nuc x occupation x channel) cell, across EVERY "
                                      "scenario AND occupation method combined.")
    parser.add_argument("--GALAXIES", type=int, required=True)
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
    outdir = f"{base}/PLOTS/universe_report"
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
            n_emri_total, n_emri_passed = read_counts(emri_snr_path(base, mbh, nuc, occ, u))
            n_ztf_total, n_ztf_passed = read_counts(tde_snr_path(base, "ZTF", mbh, nuc, occ, u))
            n_lsst_total, n_lsst_passed = read_counts(tde_snr_path(base, "LSST", mbh, nuc, occ, u))

            if n_emri_passed is None and n_ztf_passed is None and n_lsst_passed is None:
                # Nothing at all for this universe -- likely still running
                # or failed. Skip rather than fabricate a zero.
                continue

            n_emri_passed = n_emri_passed or 0
            n_ztf_passed = n_ztf_passed or 0
            n_lsst_passed = n_lsst_passed or 0
            n_total_detected = n_emri_passed + n_ztf_passed + n_lsst_passed

            row = {
                "universe": u,
                "mbh_mass_method": mbh,
                "nucleation_method": nuc,
                "occupation_method": occ,
                "n_emri_passed": n_emri_passed,
                "n_tde_ztf_passed": n_ztf_passed,
                "n_tde_lsst_passed": n_lsst_passed,
                "n_total_detected": n_total_detected,
            }
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
        ztf_arr = np.array([r["n_tde_ztf_passed"] for r in rows_list], dtype=float)
        lsst_arr = np.array([r["n_tde_lsst_passed"] for r in rows_list], dtype=float)
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

    # ── MAXIMIN selection, at (scenario x channel) granularity --
    #    i.e. over all n_scenarios*3 cells (e.g. 8 scenarios x 3
    #    channels = 24 cells), not 3 channels aggregated across
    #    scenarios.
    #
    #    Aggregating first (summing a channel across every scenario
    #    before taking the min) lets a big scenario mask a total
    #    outage in a small one. Fix: find the single worst
    #    (scenario, channel) cell for each universe FIRST, then
    #    maximize that -- guarantees every (mbh, nuc, occupation)
    #    scenario has nonzero EMRI, TDE-ZTF, and TDE-LSST individually
    #    for the winner (when any such universe exists).
    def scenario_channel_cells(u):
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

    def score(u):
        cells = scenario_channel_cells(u)
        values = [c[0] for c in cells]
        bottleneck = min(cells, key=lambda c: c[0])
        return (bottleneck[0], sum(values), bottleneck)

    global_best_universe = max(candidate_universes, key=score)
    best_min, best_total, best_bottleneck = score(global_best_universe)

    if best_min <= 0:
        _, bn_tag, bn_channel = best_bottleneck
        print(f"\n[WARN] Even the best available universe has a ZERO/MISSING cell: "
              f"scenario={bn_tag}, channel={bn_channel}. No candidate universe has "
              f"nonzero counts in EVERY (scenario, channel) combination -- consider "
              f"raising --SNR_THRESHOLD/--DETECTION_THRESHOLD's leniency, raising "
              f"T_OBS, or running more universes (larger N_UNIVERSES) to find one "
              f"that does.")

    print(f"\n=== GLOBAL WINNER (maximin over all {n_scenarios * 3} scenario x channel cells, "
          f"across every mbh/nucleation/occupation combination): universe {global_best_universe} ===")
    print(f"  Bottleneck cell: scenario={best_bottleneck[1]}, "
          f"channel={best_bottleneck[2]}, value={best_min}  (sum over all cells = {best_total})")
    print(f"  Combined per-channel totals (all {len(scenarios_present[global_best_universe])} "
          f"scenarios, diagnostic only): EMRI={combined_emri[global_best_universe]}, "
          f"TDE-ZTF={combined_ztf[global_best_universe]}, "
          f"TDE-LSST={combined_lsst[global_best_universe]}")
    for tag in per_scenario_rows:
        row = per_scenario_rows[tag].get(global_best_universe)
        if row is None:
            print(f"  {tag:35s} -> NO DATA for this universe")
        else:
            print(f"  {tag:35s} -> EMRI={row['n_emri_passed']}, "
                  f"TDE-ZTF={row['n_tde_ztf_passed']}, TDE-LSST={row['n_tde_lsst_passed']}, "
                  f"total={row['n_total_detected']}")

    print(f"\n  Winning universe {global_best_universe} vs. each scenario's own distribution "
        f"(z = (value - mean) / std, computed from that scenario's {args.N_UNIVERSES}-universe scan):")
    winner_zscores = {}  # tag -> {emri, ztf, lsst}
    for tag in per_scenario_rows:
        row = per_scenario_rows[tag].get(global_best_universe)
        s = scenario_stats.get(tag)
        if row is None or s is None:
            print(f"  {tag:35s} -> NO DATA / NO STATS")
            continue
        z_emri = zscore(row["n_emri_passed"], s["mean_emri"], s["var_emri"])
        z_ztf = zscore(row["n_tde_ztf_passed"], s["mean_ztf"], s["var_ztf"])
        z_lsst = zscore(row["n_tde_lsst_passed"], s["mean_lsst"], s["var_lsst"])
        winner_zscores[tag] = {"z_emri": z_emri, "z_ztf": z_ztf, "z_lsst": z_lsst}
        print(f"  {tag:35s} -> z(EMRI)={z_emri:+6.2f}  z(TDE-ZTF)={z_ztf:+6.2f}  "
            f"z(TDE-LSST)={z_lsst:+6.2f}")

    # ── per-scenario comparison plots (diagnostic), marking the GLOBAL winner ──
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
        ax.bar(x, ztf_vals, width, label="TDE (ZTF)", alpha=0.8)
        ax.bar(x + width, lsst_vals, width, label="TDE (LSST)", alpha=0.8)
        if global_best_universe in u_vals:
            best_idx = u_vals.index(global_best_universe)
            ax.axvline(best_idx, color="red", ls="--", lw=1.5,
                       label=f"GLOBAL winner = universe {global_best_universe}")
        ax.set_xticks(x)
        ax.set_xticklabels(u_vals, rotation=90 if len(u_vals) > 20 else 0)
        ax.set_xlabel("Universe (UNIVERSE_MODEL)")
        ax.set_ylabel("Detected sources")
        ax.set_title(f"Detected sources by universe -- {tag}\n"
                     f"(red line = the SAME winning universe used for every scenario)")
        ax.legend()
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
    ax.bar(x, ztf_sorted, width, label="TDE (ZTF)", alpha=0.85)
    ax.bar(x + width, lsst_sorted, width, label="TDE (LSST)", alpha=0.85)
    # gray out incomplete universes (missing >=1 scenario) so it's clear
    # why they weren't eligible to win
    for i, complete in enumerate(complete_mask):
        if not complete:
            ax.axvspan(i - 0.5, i + 0.5, color="lightgray", alpha=0.4, zorder=0)
    if global_best_universe in u_sorted:
        best_idx = u_sorted.index(global_best_universe)
        ax.axvline(best_idx, color="red", ls="--", lw=1.5,
                   label=f"winner = universe {global_best_universe} (maximin)")
    ax.set_xticks(x)
    ax.set_xticklabels(u_sorted, rotation=90 if len(u_sorted) > 20 else 0)
    ax.set_xlabel("Universe (UNIVERSE_MODEL)")
    ax.set_ylabel(f"Combined total per channel, summed across all {n_scenarios} scenarios")
    ax.set_title("Per-channel detections by universe -- winner maximizes the WORST "
                 "channel, not the sum\n(shaded = missing >=1 scenario, excluded)")
    ax.legend()
    plt.tight_layout()
    combined_plot_path = f"{outdir}/combined_totals_by_universe.png"
    plt.savefig(combined_plot_path, dpi=150)
    plt.close()
    print(f"\nSaved: {combined_plot_path}")

    # ── copy the SAME winning universe's files for EVERY scenario,
    #    with a max_detected_ prefix. ──
    print(f"\nCopying universe {global_best_universe}'s files for every scenario "
          f"(max_detected_ prefix):")
    for mbh, nuc, occ in SCENARIOS:
        tag = scenario_tag(mbh, nuc, occ)
        u = global_best_universe
        src_paths = {
            "EMRI_SNRs": emri_snr_path(base, mbh, nuc, occ, u),
            "TDE_SNR_ZTF": tde_snr_path(base, "ZTF", mbh, nuc, occ, u),
            "TDE_SNR_LSST": tde_snr_path(base, "LSST", mbh, nuc, occ, u),
        }
        for label, src in src_paths.items():
            if not os.path.exists(src):
                print(f"  [warn] {tag}/{label} missing for winning universe {u}, "
                      f"can't copy: {src}")
                continue
            fname = os.path.basename(src)
            dst = f"{base}/max_detected_{fname}"
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
    latex_lines = [
        "% Auto-generated by report_universes.py -- paste into a LaTeX table body.",
        "\\begin{tabular}{lrccccccccc}",
        "\\hline",
        "Scenario & $N$ & EMRI mean & EMRI var & EMRI CV\\% & "
        "TDE-ZTF mean & TDE-ZTF var & TDE-ZTF CV\\% & "
        "TDE-LSST mean & TDE-LSST var & TDE-LSST CV\\% \\\\",
        "\\hline",
    ]
    for mbh, nuc, occ in SCENARIOS:
        tag = scenario_tag(mbh, nuc, occ)
        s = scenario_stats.get(tag)
        if s is None:
            print(f"  {tag:35s}  NO DATA")
            continue
        cv_emri = 100 * math.sqrt(s["var_emri"]) / s["mean_emri"] if s["mean_emri"] else float("nan")
        cv_ztf = 100 * math.sqrt(s["var_ztf"]) / s["mean_ztf"] if s["mean_ztf"] else float("nan")
        cv_lsst = 100 * math.sqrt(s["var_lsst"]) / s["mean_lsst"] if s["mean_lsst"] else float("nan")
        line = (f"  {tag:35s} {s['n']:>4d}  "
                f"{s['mean_emri']:>10.3f} {s['var_emri']:>10.3f} {cv_emri:>9.2f}%  "
                f"{s['mean_ztf']:>12.3f} {s['var_ztf']:>12.3f} {cv_ztf:>11.2f}%  "
                f"{s['mean_lsst']:>13.3f} {s['var_lsst']:>13.3f} {cv_lsst:>12.2f}%")
        print(line)
        counts_table_lines.append(line)
        tag_escaped = tag.replace('_', '\\_')
        latex_lines.append(
            f"{tag_escaped} & {s['n']} & "
            f"{s['mean_emri']:.3f} & {s['var_emri']:.3f} & {cv_emri:.2f}\\% & "
            f"{s['mean_ztf']:.3f} & {s['var_ztf']:.3f} & {cv_ztf:.2f}\\% & "
            f"{s['mean_lsst']:.3f} & {s['var_lsst']:.3f} & {cv_lsst:.2f}\\% \\\\"
        )
    latex_lines += ["\\hline", "\\end{tabular}"]

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
    ratio_latex_lines = [
        "% Auto-generated by report_universes.py -- paste into a LaTeX table body.",
        "% <.> denotes a mean taken across realizations (universes), not a single value.",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "Scenario & $\\langle \\mathrm{EMRI/TDE\\text{-}ZTF} \\rangle$ & "
        "$\\mathrm{Var}$ & $N$ & "
        "$\\langle \\mathrm{EMRI/TDE\\text{-}LSST} \\rangle$ & "
        "$\\mathrm{Var}$ & $N$ \\\\",
        "\\hline",
    ]
    for mbh, nuc, occ in SCENARIOS:
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

        tag_escaped = tag.replace('_', '\\_')
        ratio_latex_lines.append(
            f"{tag_escaped} & "
            f"${sci_latex(rs['mean_ratio_ztf'])}$ & ${sci_latex(rs['var_ratio_ztf'])}$ & {rs['n_ztf']} & "
            f"${sci_latex(rs['mean_ratio_lsst'])}$ & ${sci_latex(rs['var_ratio_lsst'])}$ & {rs['n_lsst']} \\\\"
        )
    ratio_latex_lines += ["\\hline", "\\end{tabular}"]

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

    # ── final summary ──
    summary_lines = [f"=== GLOBAL WINNING UNIVERSE: {global_best_universe} ==="]
    summary_lines.append(
        f"Selection criterion: maximin over all {n_scenarios * 3} (scenario x channel) cells "
        f"({n_scenarios} scenarios [= {len(MBH_NUC_SCENARIOS)} mbh/nuc combos x "
        f"{len(OCCUPATION_METHODS)} occupation methods] x {{EMRI, TDE-ZTF, TDE-LSST}}), "
        f"sum as tiebreaker."
    )
    summary_lines.append(
        f"Bottleneck cell: scenario={best_bottleneck[1]}, channel={best_bottleneck[2]}, "
        f"value={best_min}  (sum over all cells = {best_total})"
    )
    summary_lines.append(
        f"Combined per-channel totals (diagnostic only): "
        f"EMRI={combined_emri[global_best_universe]}, "
        f"TDE-ZTF={combined_ztf[global_best_universe]}, "
        f"TDE-LSST={combined_lsst[global_best_universe]}"
    )
    summary_lines.append("")
    for tag in per_scenario_rows:
        row = per_scenario_rows[tag].get(global_best_universe)
        if row is None:
            summary_lines.append(f"{tag:35s} -> NO DATA for this universe")
        else:
            summary_lines.append(
                f"{tag:35s} -> EMRI={row['n_emri_passed']}, "
                f"TDE-ZTF={row['n_tde_ztf_passed']}, "
                f"TDE-LSST={row['n_tde_lsst_passed']}, total={row['n_total_detected']}"
            )
    summary_lines.append("")
    summary_lines.append(
        "This is ONE universe used for EVERY scenario -- every (mbh_mass_method, "
        "nucleation_method) combination AND every occupation_method -- not a "
        "separate best universe per scenario and not a separate winner per "
        "occupation method. UNIVERSE_MODEL is the realization seed, so a fair "
        "comparison across scenarios AND across occupation methods needs them "
        "all run on the same realization. The winner maximizes the WORST "
        "individual (scenario, channel) cell -- not a channel total summed "
        "across scenarios -- so it cannot have a per-scenario outage hidden by "
        "a larger scenario's numbers."
    )
    summary_lines.append("")
    summary_lines.append("Winner's z-score vs. each scenario's own distribution "
                        "(diagnostic; NOT part of the selection criterion):")
    for tag, z in winner_zscores.items():
        summary_lines.append(
            f"  {tag:35s} z(EMRI)={z['z_emri']:+.2f}  z(TDE-ZTF)={z['z_ztf']:+.2f}  "
            f"z(TDE-LSST)={z['z_lsst']:+.2f}"
        )
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_path = f"{outdir}/best_universe.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()