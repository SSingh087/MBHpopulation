#!/usr/bin/env python
import os, sys, argparse
sys.path.insert(0, os.path.abspath('../../poplar'))
import h5py
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES", type=int, required=True)
parser.add_argument("--POPULATION", type=str, default="A", choices=["A", "B", "MIX"])
args = parser.parse_args()

loc = f'./data/pseudo_inference_data/{args.GALAXIES}'


# ── Load data ─────────────────────────────────────────────────────────────
def load_h5(path):
    with h5py.File(path, 'r') as hf:
        return {k: hf[k][()] for k in hf.keys()}


emri = load_h5(os.path.join(loc, f"true_data_EMRI_pseudo_data_{args.POPULATION}.h5"))
ztf  = load_h5(os.path.join(loc, f"true_data_TDE_ZTF_pseudo_data_{args.POPULATION}.h5"))
lsst = load_h5(os.path.join(loc, f"true_data_TDE_LSST_pseudo_data_{args.POPULATION}.h5"))

print("[INFO] EMRI keys :", [k for k in emri if not k.endswith("_posterior")])
print("[INFO] TDE  keys :", [k for k in ztf if not k.endswith("_posterior")])

# ── Plot config ───────────────────────────────────────────────────────────
# Map each physical quantity to its datasets and axis label
panels = [
    {
        "title": r"Black hole mass  $\log_{10}(M_\mathrm{BH}/M_\odot)$",
        "xlabel": r"$\log_{10}(M_\mathrm{BH}/M_\odot)$",
        "sources": [
            ("EMRI", emri["lgMBH_mass"]),
            ("TDE ZTF", ztf["mbh_6"] + 6),   # convert back for plotting
            ("TDE LSST", lsst["mbh_6"] + 6),
        ],
    },
    {
        "title": r"MBH spin  $a$",
        "xlabel": r"$a$",
        "sources": [
            ("EMRI", emri["MBHspin"]),
            # TDEs have no spin
        ],
    },
    {
        "title": r"Companion / stellar mass  $m$",
        "xlabel": r"$m \ [M_\odot]$",
        "sources": [
            ("EMRI", emri["CO_mass"]),
            ("TDE ZTF", ztf["stellar_mass"]),
            ("TDE LSST", lsst["stellar_mass"]),
        ],
    },
    {
        "title": r"Redshift  $z$",
        "xlabel": r"$z$",
        "sources": [
            ("EMRI", emri["z_gal"]),
            ("TDE ZTF", ztf["redshift"]),
            ("TDE LSST", lsst["redshift"]),
        ],
    },
    {
        "title": r"Eccentricity  $e_\mathrm{pl}$",
        "xlabel": r"$e_\mathrm{pl}$",
        "sources": [
            ("EMRI", emri["e_pl"]),
        ],
    },
]

colors = {"EMRI": "steelblue", "TDE ZTF": "tomato", "TDE LSST": "seagreen"}

# ── Figure ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle(
    f"Data catalogue  |  pop={args.POPULATION}  |  GALAXIES={args.GALAXIES}"
    f"| EMRIs={len(emri['lgMBH_mass'])} | TDEs={len(ztf['mbh_6'])} ",
    fontsize=13
)

for ax, panel in zip(axes, panels):
    for label, values in panel["sources"]:
        ax.hist(values, bins=60, density=True, histtype="stepfilled",
                 alpha=0.4, color=colors[label], label=label)
        ax.hist(values, bins=60, density=True, histtype="step",
                 linewidth=1.4, color=colors[label])
    ax.set_title(panel["title"], fontsize=10)
    ax.set_xlabel(panel["xlabel"], fontsize=11)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()

out = os.path.join(loc, f"catalogue_histograms_pop{args.POPULATION}.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\n[SAVED] {out}")