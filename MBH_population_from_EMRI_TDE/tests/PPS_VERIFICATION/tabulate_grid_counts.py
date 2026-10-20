#!/usr/bin/env python
"""
Reads the outputs of run_verification_grid.sh (a fresh, single-universe
re-run of the pipeline for all 8 UNIVERSE scenarios, written to a NEW
astrophysical_data/{GALAXIES}/ directory distinct from the production
30-universe grid) and prints/saves a table of final EMRI/TDE detection
counts per scenario.

Reuses report_universes.py's own path/attrs-reading conventions
(emri_snr_path, tde_snr_path, read_counts, SCENARIO_NAME_MAP) rather
than re-deriving them, so this table is directly comparable to the
paper's Table 3.

Usage:
    python tests/PPS_VERIFICATION/tabulate_grid_counts.py --GALAXIES 20000 --UNIVERSE_MODEL 9001
"""
import os, sys, argparse, csv
sys.path.insert(0, os.path.abspath('./astrophysical_setup'))
sys.path.insert(0, os.path.abspath('./'))
from utils import ASTROPHYSICAL_DATA_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REAL_ZTF_TDE_COUNT = 33  # Yao et al. 2023, 3-yr flux-limited sample -- reference line only

from report_universes import (
    SCENARIOS, SCENARIO_NAME_MAP, TDE_MODELS, SURVEYS,
    emri_snr_path, tde_snr_path, read_counts,
)

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES", type=int, required=True)
parser.add_argument("--UNIVERSE_MODEL", type=int, required=True)
parser.add_argument("--TDE_MODELS", type=str, nargs="*", default=None,
                     help="Subset of TDE models to look for (default: all 4, missing files are just skipped).")
parser.add_argument("--SURVEYS", type=str, nargs="*", default=None,
                     help="Subset of surveys to look for (default: ZTF, LSST).")
args = parser.parse_args()

models = args.TDE_MODELS or TDE_MODELS
surveys = args.SURVEYS or SURVEYS

base = f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}"
u = args.UNIVERSE_MODEL

rows = []
print(f"Reading verification grid from {base}  (UNIVERSE_MODEL={u})\n")

for (mbh, nuc, occ) in SCENARIOS:
    tag = SCENARIO_NAME_MAP[(mbh, nuc, occ)]

    n_emri_tot, n_emri_pass = read_counts(emri_snr_path(base, mbh, nuc, occ, u))
    rows.append({
        "scenario": tag, "mbh": mbh, "nuc": nuc, "occ": occ,
        "channel": "EMRI", "model": "-", "survey": "-",
        "n_total": n_emri_tot, "n_passed": n_emri_pass,
    })

    for model in models:
        for survey in surveys:
            n_tot, n_pass = read_counts(tde_snr_path(base, survey, mbh, nuc, occ, model, u))
            rows.append({
                "scenario": tag, "mbh": mbh, "nuc": nuc, "occ": occ,
                "channel": "TDE", "model": model, "survey": survey,
                "n_total": n_tot, "n_passed": n_pass,
            })

# ---- print table ----
hdr = f"{'scenario':8s} {'channel':6s} {'model':18s} {'survey':6s} {'n_total':>12s} {'n_passed':>10s}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    nt = "MISSING" if r["n_total"] is None else f"{r['n_total']:,}"
    npass = "MISSING" if r["n_passed"] is None else f"{r['n_passed']:,}"
    print(f"{r['scenario']:8s} {r['channel']:6s} {r['model']:18s} {r['survey']:6s} {nt:>12s} {npass:>10s}")

# ---- save csv ----
out_csv = f"{base}/verification_grid_counts_{u}.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["scenario", "mbh", "nuc", "occ", "channel", "model", "survey", "n_total", "n_passed"])
    w.writeheader()
    w.writerows(rows)
print(f"\nSaved {out_csv}")

# ---- bar plot: final detected counts per scenario, EMRI + TDE(ZTF, first model/survey found), vs real ZTF count ----
tde_rows = [r for r in rows if r["channel"] == "TDE" and r["n_passed"] is not None]
emri_rows = [r for r in rows if r["channel"] == "EMRI" and r["n_passed"] is not None]

if tde_rows:
    first_model = tde_rows[0]["model"]
    first_survey = tde_rows[0]["survey"]
    tde_plot_rows = [r for r in tde_rows if r["model"] == first_model and r["survey"] == first_survey]
    scenarios_order = [r["scenario"] for r in tde_plot_rows]
    tde_passed = [r["n_passed"] for r in tde_plot_rows]
    emri_passed = [next((e["n_passed"] for e in emri_rows if e["scenario"] == s), 0) for s in scenarios_order]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(scenarios_order))
    axs[0].bar(x, tde_passed, color="tab:orange")
    axs[0].axhline(REAL_ZTF_TDE_COUNT, color="black", ls="--", lw=1.5,
                    label=f"real ZTF sample (Yao+23, N={REAL_ZTF_TDE_COUNT})")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(scenarios_order)
    axs[0].set_ylabel("Final detected TDEs")
    axs[0].set_title(f"TDE detections per scenario ({first_model}, {first_survey})")
    axs[0].set_yscale("log")
    axs[0].legend(fontsize=8)

    axs[1].bar(x, emri_passed, color="tab:blue")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(scenarios_order)
    axs[1].set_ylabel("Final detected EMRIs")
    axs[1].set_title("EMRI detections per scenario")

    plt.tight_layout()
    out_png = f"{base}/verification_grid_counts_{u}.png"
    plt.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")
