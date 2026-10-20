#!/usr/bin/env python
"""
Before/after EMRI count comparison per scenario (M1-M8): original
(over-predicted) EMRI_SNRs_data counts vs. the realistic, ZTF-ratio-thinned
counts from sweep_thin_EMRI.py.

Saved next to data_generation/predict_EMRI_counts.py's
scenario_comparison_boxplots.png, with a "_realistic" suffix.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath("./"))
import numpy as np
import matplotlib.pyplot as plt

from common_realistic import (
    MERGER_MODEL_CHOICES, set_merger_model,
    ASTROPHYSICAL_DATA_DIR, GALAXIES_DEFAULT, SCENARIOS, SCENARIO_NAME_MAP,
    discover_universes, emri_snr_path, load_h5_catalog, scenario_tag,
)

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES", type=int, default=GALAXIES_DEFAULT)
parser.add_argument("--merger_model", type=str, required=True, choices=MERGER_MODEL_CHOICES,
                    help="Galaxy merger-rate model whose universes are processed.")
args = parser.parse_args()
set_merger_model(args.merger_model)
GALAXIES = args.GALAXIES

REPORT_DIR = os.path.join(ASTROPHYSICAL_DATA_DIR, str(GALAXIES), "PLOTS", "universe_report", args.merger_model)
os.makedirs(REPORT_DIR, exist_ok=True)

order = [SCENARIO_NAME_MAP[s] for s in SCENARIOS]
original_counts = {label: [] for label in order}
realistic_counts = {label: [] for label in order}

for scenario in SCENARIOS:
    label = SCENARIO_NAME_MAP[scenario]
    universes = discover_universes(emri_snr_path(ASTROPHYSICAL_DATA_DIR, GALAXIES, scenario, "*"))
    for u in universes:
        orig_path = emri_snr_path(ASTROPHYSICAL_DATA_DIR, GALAXIES, scenario, u)
        data, _ = load_h5_catalog(orig_path)
        original_counts[label].append(len(data["z_gal"]))

        real_path = emri_snr_path(ASTROPHYSICAL_DATA_DIR, GALAXIES, scenario, u, realistic=True)
        if os.path.exists(real_path):
            rdata, _ = load_h5_catalog(real_path)
            realistic_counts[label].append(len(rdata["z_gal"]))

present = [m for m in order if original_counts[m]]
if not present:
    raise RuntimeError("No EMRI catalogs found -- run sweep_thin_EMRI.py first.")

positions_orig = np.arange(len(present)) * 2.5
positions_real = positions_orig + 1.0

fig, ax = plt.subplots(figsize=(11, 6.5))

bp1 = ax.boxplot([original_counts[m] for m in present], positions=positions_orig,
                  widths=0.8, patch_artist=True, showfliers=True)
bp2 = ax.boxplot([realistic_counts[m] for m in present], positions=positions_real,
                  widths=0.8, patch_artist=True, showfliers=True)

for patch in bp1["boxes"]:
    patch.set_facecolor("#7FA8C9")
    patch.set_alpha(0.7)
for patch in bp2["boxes"]:
    patch.set_facecolor("#E9967A")
    patch.set_alpha(0.7)

ax.set_xticks((positions_orig + positions_real) / 2)
ax.set_xticklabels(present)
ax.set_yscale("log")
ax.set_xlabel("Model")
ax.set_ylabel("EMRI count (per universe)")
ax.set_title("EMRI counts per scenario -- original vs. realistic (ZTF-ratio-thinned)")
ax.grid(alpha=0.3, axis="y")
ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Original", "Realistic"], loc="upper right")

plt.tight_layout()
out_path = os.path.join(REPORT_DIR, f"scenario_comparison_boxplots_realistic_{args.merger_model}.png")
plt.savefig(out_path, dpi=200)
plt.close(fig)
print(f"Saved: {out_path}")
