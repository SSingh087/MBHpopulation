#!/usr/bin/env python
"""
Merge the sigma_calibration/{SURVEY}_chunk*.h5 files written by calibrate_sigma_TDE_fallback.py
and turn them into the sigma(z) table that prep_pseudo_data.py reads with --SIGMA_FILE:

    {"ZTF": {"lgMBH_mass": {"z": [...], "sigma": [...]}, "z_gal": {...}}, "LSST": {...}}

sigma(z) is the median Fisher sigma in log-spaced redshift bins (np.interp between bin centres,
clamped outside the calibrated range).  The Fisher sigma is a LOWER bound on the true
measurement error.  Also writes sigma_fit_tde_fallback.png comparing with the old linear
surrogate sigma_ref*(1+z/z_ref).

Usage:  python fisher_analysis/fit_TDE_sigma.py --GALAXIES 100000
"""
import argparse
import glob
import json
import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath('./'))
from utils import FISHER_DATA_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES", type=int, required=True)
parser.add_argument("--N_BINS", type=int, default=8)
args = parser.parse_args()

d = f"{FISHER_DATA_DIR}/{args.GALAXIES}/sigma_calibration"
OLD = {"ZTF": (0.1, 0.3, 5e-3), "LSST": (0.2, 0.3, 5e-3)}   # z_ref, sigma_mass_ref, sigma_z_ref in prep_pseudo_data.py
table, fig_axes = {}, plt.subplots(2, 2, figsize=(11, 8), sharex=True)
fig, axes = fig_axes

for r, survey in enumerate(("ZTF", "LSST")):
    files = sorted(glob.glob(f"{d}/{survey}_chunk*.h5"))
    if not files:
        print(f"[warn] no calibration files for {survey}")
        continue
    z, sig, names = [], [], None
    for p in files:
        with h5py.File(p, "r") as f:
            names = [n.decode() if isinstance(n, bytes) else n for n in f.attrs["param_names"]]
            z.append(f["z"][:]); sig.append(f["sigma"][:])
    z, sig = np.concatenate(z), np.concatenate(sig)
    print(f"{survey}: {len(z)} events from {len(files)} chunks, z {z.min():.3f}-{z.max():.3f}")
    edges = np.geomspace(z.min(), z.max() * 1.0001, args.N_BINS + 1)
    b = np.clip(np.digitize(z, edges) - 1, 0, args.N_BINS - 1)
    table[survey] = {}
    for c, (key, col) in enumerate((("lgMBH_mass", names.index("mbh6")), ("z_gal", names.index("redshift")))):
        s = sig[:, col]
        good = np.isfinite(s) & (s > 0)
        zc, sc = [], []
        for k in range(args.N_BINS):
            m = (b == k) & good
            if m.sum() >= 2:
                zc.append(float(np.median(z[m]))); sc.append(float(np.exp(np.median(np.log(s[m])))))
        table[survey][key] = {"z": zc, "sigma": sc}
        ax = axes[r, c]
        ax.loglog(z[good], s[good], ".", alpha=0.4, label="Fisher events")
        ax.loglog(zc, sc, "o-", color="C3", label="binned median (used)")
        zz = np.geomspace(z.min(), z.max(), 100)
        z_ref, s_m, s_z = OLD[survey]
        ax.loglog(zz, (s_m if c == 0 else s_z) * (1 + zz / z_ref), "k--", label="old linear surrogate")
        ax.set_title(f"{survey}: sigma_{key}")
        ax.set_xlabel("z"); ax.set_ylabel("sigma")
    axes[r, 0].legend(fontsize=8)

json.dump(table, open(f"{d}/sigma_fit_tde_fallback.json", "w"), indent=1)
fig.tight_layout()
fig.savefig(f"{d}/sigma_fit_tde_fallback.png", dpi=130)
print(f"wrote {d}/sigma_fit_tde_fallback.json and .png")
