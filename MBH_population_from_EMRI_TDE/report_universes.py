#!/usr/bin/env python
"""
Scans the EMRI/TDE SNR-detection output for every (scenario, universe)
combo, tabulates detected-source counts, and picks ONE winning
UNIVERSE_MODEL -- the same number for all 4 scenarios -- that
maximizes TOTAL detected sources summed across all 4 scenarios.

This is deliberately NOT "best universe per scenario": UNIVERSE_MODEL
is the seed for the whole realization, so a fair downstream comparison
across the 4 (mbh_mass_method, nucleation_method) scenarios needs them
all run on the SAME universe, not 4 different ones cherry-picked
independently.

ASSUMPTION (single point of change if wrong): every stage's output
filename gets "_{U}" appended after the existing
"_{mbh_mass_method}_{nucleation_method}" scenario suffix, matching
what generate_universes_dag.py passes via --UNIVERSE_MODEL. If your
actual suffix pattern differs, only emri_snr_path()/tde_snr_path()
below need to change -- nothing else in this script assumes a
particular pattern.

Reads (per scenario, per universe):
    EMRI_SNRs_data_{mbh}_{nuc}_{U}.h5      .attrs['n_events_total'/'n_events_passed']
    TDE_SNRs_ZTF_{mbh}_{nuc}_{U}.h5        .attrs['n_events_total'/'n_events_passed']
    TDE_SNRs_LSST_{mbh}_{nuc}_{U}.h5       .attrs['n_events_total'/'n_events_passed']

Writes:
    {outdir}/universe_report_{mbh}_{nuc}.csv   (per-scenario table, diagnostic)
    {outdir}/universe_report_all.csv           (all scenarios combined, diagnostic)
    {outdir}/universe_comparison_{mbh}_{nuc}.png   (per-scenario plot, diagnostic;
                                                     marks the GLOBAL winner, not a
                                                     per-scenario local best)
    {outdir}/combined_totals_by_universe.png   (the actual selection plot --
                                                 total detected per universe,
                                                 summed across all 4 scenarios)
    {base}/max_detected_*.h5   (winning universe's files, all 4 scenarios,
                                 copied with a max_detected_ prefix)
    Winner summary printed to stdout AND saved as
    {outdir}/best_universe.txt

Usage:
    python report_universes.py --GALAXIES 100000 --N_UNIVERSES 50
"""

import os, sys, argparse, csv, shutil
sys.path.insert(0, os.path.abspath('./astrophysical_setup'))
sys.path.insert(0, os.path.abspath('./'))
from utils import ASTROPHYSICAL_DATA_DIR

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCENARIOS = [
    ("use_lgMgal", "hannah"),
    ("use_sigma",  "hannah"),
    ("use_lgMgal", "neumayer"),
    ("use_sigma",  "neumayer"),
]
SURVEYS = ["ZTF", "LSST"]


def emri_snr_path(base, mbh, nuc, u):
    return f"{base}/EMRI_SNRs_data_{mbh}_{nuc}_{u}.h5"

def tde_snr_path(base, survey, mbh, nuc, u):
    return f"{base}/TDE_SNRs_{survey}_{mbh}_{nuc}_{u}.h5"


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
    parser = argparse.ArgumentParser(description="Pick one UNIVERSE_MODEL, maximizing total detections across all scenarios.")
    parser.add_argument("--GALAXIES", type=int, required=True)
    parser.add_argument("--N_UNIVERSES", type=int, required=True,
                         help="Number of universes to scan (0..N_UNIVERSES-1), "
                              "matching generate_universes_dag.py's N_UNIVERSES.")
    parser.add_argument("--REQUIRE_ALL_SCENARIOS", action="store_true", default=True,
                         help="Only consider universes with readable output for ALL "
                              "4 scenarios when picking the winner (default: on). "
                              "Use --no-REQUIRE_ALL_SCENARIOS to allow partial "
                              "universes to compete (their total will just be lower, "
                              "which biases the selection -- off by default for that "
                              "reason).")
    parser.add_argument("--no-REQUIRE_ALL_SCENARIOS", dest="REQUIRE_ALL_SCENARIOS",
                         action="store_false")
    args = parser.parse_args()

    base = f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}"
    outdir = f"{base}/PLOTS/universe_report"
    os.makedirs(outdir, exist_ok=True)

    # per_scenario_rows[tag] = {universe: row_dict}
    per_scenario_rows = {}
    all_rows = []

    for mbh, nuc in SCENARIOS:
        tag = f"{mbh}_{nuc}"
        print(f"\n=== scanning scenario: {tag} ===")

        rows_by_universe = {}
        for u in range(args.N_UNIVERSES):
            n_emri_total, n_emri_passed = read_counts(emri_snr_path(base, mbh, nuc, u))
            n_ztf_total, n_ztf_passed = read_counts(tde_snr_path(base, "ZTF", mbh, nuc, u))
            n_lsst_total, n_lsst_passed = read_counts(tde_snr_path(base, "LSST", mbh, nuc, u))

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

        per_scenario_rows[tag] = rows_by_universe

    if not all_rows:
        print("\n[ERROR] No readable universe output for ANY scenario -- nothing to report.")
        return

    # ── combined per-universe totals, summed across ALL 4 scenarios,
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
                  f"readable output for all {n_scenarios} scenarios. Either wait "
                  f"for more jobs to finish, or rerun with "
                  f"--no-REQUIRE_ALL_SCENARIOS to allow partial universes "
                  f"(not recommended -- biases toward universes missing the "
                  f"hardest scenario).")
            return
        candidate_universes = complete_universes
    else:
        candidate_universes = list(scenarios_present.keys())

    # ── MAXIMIN selection, at (scenario x channel) granularity --
    #    i.e. over all 12 cells (4 scenarios x 3 channels), not 3
    #    channels aggregated across scenarios.
    #
    #    Aggregating first (summing a channel across all 4 scenarios
    #    before taking the min) lets a big scenario mask a total
    #    outage in a small one -- e.g. a universe with 0 TDE-ZTF
    #    specifically in the lgMgal/neumayer scenario can still have a
    #    healthy summed TDE-ZTF total because Hannah's scenarios
    #    (much larger counts) carry it. That's exactly the bug: the
    #    previous version could pick a universe with a real per-scenario
    #    outage as long as OTHER scenarios covered for it in the sum.
    #
    #    Fix: find the single worst (scenario, channel) cell for each
    #    universe FIRST, then maximize that. This guarantees every
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

    print(f"\n=== GLOBAL WINNER (maximin over all 12 scenario x channel cells): "
          f"universe {global_best_universe} ===")
    print(f"  Bottleneck cell: scenario={best_bottleneck[1]}, "
          f"channel={best_bottleneck[2]}, value={best_min}  (sum over all cells = {best_total})")
    print(f"  Combined per-channel totals (all {len(scenarios_present[global_best_universe])} "
          f"scenarios, diagnostic only): EMRI={combined_emri[global_best_universe]}, "
          f"TDE-ZTF={combined_ztf[global_best_universe]}, "
          f"TDE-LSST={combined_lsst[global_best_universe]}")
    for tag in per_scenario_rows:
        row = per_scenario_rows[tag].get(global_best_universe)
        if row is None:
            print(f"  {tag:25s} -> NO DATA for this universe")
        else:
            print(f"  {tag:25s} -> EMRI={row['n_emri_passed']}, "
                  f"TDE-ZTF={row['n_tde_ztf_passed']}, TDE-LSST={row['n_tde_lsst_passed']}, "
                  f"total={row['n_total_detected']}")

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
        ax.set_yscale("log")
        ax.set_xlabel("Universe (UNIVERSE_MODEL)")
        ax.set_ylabel("Detected sources")
        ax.set_title(f"Detected sources by universe -- {tag}\n"
                     f"(red line = the SAME winning universe used for all scenarios)")
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
    ax.set_yscale("log")
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

    # ── copy the SAME winning universe's files for ALL 4 scenarios,
    #    with a max_detected_ prefix. ──
    print(f"\nCopying universe {global_best_universe}'s files for all scenarios "
          f"(max_detected_ prefix):")
    for mbh, nuc in SCENARIOS:
        tag = f"{mbh}_{nuc}"
        u = global_best_universe
        src_paths = {
            "EMRI_SNRs": emri_snr_path(base, mbh, nuc, u),
            "TDE_SNR_ZTF": tde_snr_path(base, "ZTF", mbh, nuc, u),
            "TDE_SNR_LSST": tde_snr_path(base, "LSST", mbh, nuc, u),
        }

        fnames = {
            "EMRI_SNRs": f"{base}/EMRI_SNRs_data_{mbh}_{nuc}.h5",
            "TDE_SNR_ZTF": f"{base}/TDE_SNRs_ZTF_{mbh}_{nuc}.h5",
            "TDE_SNR_LSST": f"{base}/TDE_SNRs_LSST_{mbh}_{nuc}.h5",
        }

        for label, src in src_paths.items():
            if not os.path.exists(src):
                print(f"  [warn] {tag}/{label} missing for winning universe {u}, "
                      f"can't copy: {src}")
                continue
            fname = os.path.basename(fnames[label])
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

    # ── final summary ──
    summary_lines = [f"=== GLOBAL WINNING UNIVERSE: {global_best_universe} ==="]
    summary_lines.append(
        f"Selection criterion: maximin over all 12 (scenario x channel) cells "
        f"(4 scenarios x {{EMRI, TDE-ZTF, TDE-LSST}}), sum as tiebreaker."
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
            summary_lines.append(f"{tag:25s} -> NO DATA for this universe")
        else:
            summary_lines.append(
                f"{tag:25s} -> EMRI={row['n_emri_passed']}, "
                f"TDE-ZTF={row['n_tde_ztf_passed']}, "
                f"TDE-LSST={row['n_tde_lsst_passed']}, total={row['n_total_detected']}"
            )
    summary_lines.append("")
    summary_lines.append(
        "This is ONE universe used for all 4 scenarios (not a separate best "
        "universe per scenario) -- UNIVERSE_MODEL is the realization seed, so "
        "a fair cross-scenario comparison downstream needs them all run on the "
        "same realization. The winner maximizes the WORST individual "
        "(scenario, channel) cell -- not a channel total summed across "
        "scenarios -- so it cannot have a per-scenario outage hidden by a "
        "larger scenario's numbers."
    )
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_path = f"{outdir}/best_universe.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()