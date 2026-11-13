#!/usr/bin/env python
"""Fig.-5-style box plot: detected EMRIs and TDEs (ZTF, LSST) per scenario, catalogue (EMRI_TDE_data, 100 universes, oleary21, use_sigma) vs this pipeline (scenario_counts.py)."""
import glob, os, h5py, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"; CAT = "/data/wiay/postgrads/shashwat/EMRI_TDE_data/astrophysical_data/100000"
SC = [("σHU", "HU", "hannah", "uniform"), ("σNU", "NU", "neumayer", "uniform"), ("σHV", "HV", "hannah", "varying"), ("σNV", "NV", "neumayer", "varying")]
def cat(kind, nuc, occ):
    pat = {"EMRI": f"EMRI_SNRs_data_use_sigma_{nuc}_{occ}_oleary21_*.h5"}.get(kind, f"TDE_SNRs_{kind}_use_sigma_{nuc}_{occ}_oleary21_tde_fallback_*.h5")
    return np.array([h5py.File(f).attrs["n_events_passed"] for f in sorted(glob.glob(f"{CAT}/{pat}"))])
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2)); C1, C2 = "#1f77b4", "#d95f02"
for ax, (kind, key) in zip(axs, (("EMRI", "N_emri"), ("ZTF", "N_ZTF"), ("LSST", "N_LSST"))):
    for i, (lab, s, nuc, occ) in enumerate(SC):
        c = cat(kind, nuc, occ); f = f"{B}/scenarios/counts_check/sigma{s}_legacy_a075.npz"
        m = np.load(f)[key] if os.path.exists(f) else None
        kw = dict(widths=0.32, patch_artist=True, showfliers=True, flierprops=dict(markersize=2.5), medianprops=dict(color="k"))
        b1 = ax.boxplot(c, positions=[i - 0.19], **kw); b1["boxes"][0].set(facecolor=C1, alpha=.7)
        if m is not None: b2 = ax.boxplot(m, positions=[i + 0.19], **kw); b2["boxes"][0].set(facecolor=C2, alpha=.7)
    ax.set_xticks(range(4)); ax.set_xticklabels([x[0] for x in SC]); ax.set_ylabel(f"detected {kind}" + (" (SNR>20)" if kind == "EMRI" else " (p>0.5)")); ax.set_title(kind)
    if kind == "ZTF": ax.axhline(44, color="gray", ls="--", lw=1); ax.text(3.45, 46, "observed ~44", ha="right", fontsize=8, color="gray")
    if kind == "LSST": ax.set_yscale("log")
axs[0].legend([plt.Rectangle((0, 0), 1, 1, fc=C1, alpha=.7), plt.Rectangle((0, 0), 1, 1, fc=C2, alpha=.7)], ["catalogue (old pipeline)", "this work (flows' simulator + SF)"], fontsize=8)
fig.tight_layout(); fig.savefig(f"{B}/plots/scenarios/detected_boxplot.png", dpi=150); print("saved")
