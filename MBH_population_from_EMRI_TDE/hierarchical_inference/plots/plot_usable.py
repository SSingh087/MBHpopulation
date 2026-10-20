#!/usr/bin/env python
"""How many mock events are (nearly) uninformative?  Light: reads only true values, applies the
sigma model of prep_pseudo_data.py (TDE: 0.3(1+z/z_ref), EMRI: 0.01) analytically.
Usable classes by sigma_lgM: <0.3 dex good, 0.3-1 marginal, >1 dex (>1/3 of the [4,7] box) poor.
Also: fraction whose noisy centre falls outside the mass box [4,7] / z box.  -> usable_events.png/.txt"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, h5py, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from utils import SCENARIO_KEY, MODEL_LABELS
ap = argparse.ArgumentParser(); ap.add_argument("--POP", default="A"); ap.add_argument("--CAP", type=float, default=None)
a = ap.parse_args()
B = "./data/pseudo_inference_data/100000"
ZREF = {"ZTF": 0.1, "LSST": 0.2}; rng = np.random.default_rng(0)
keys = sorted(SCENARIO_KEY, key=lambda t: int(MODEL_LABELS[t][1:]))
cls = {s: [] for s in ("ZTF", "LSST", "EMRI")}; out = {s: [] for s in cls}; zs = {}
for k in keys:
    sc = "_".join(k)
    for s in cls:
        fn = f"{B}/true_data_{s if s=='EMRI' else 'TDE_'+s}_pseudo_data_{a.POP}_{sc}.h5"
        with h5py.File(fn) as f:
            if s == "EMRI": m, z = f["lgMBH_mass"][:], f["z_gal"][:]; sig = np.full(len(m), 0.01)
            else:
                m, z = f["mbh_6"][:] + 6, f["redshift"][:]; sig = 0.3 * (1 + z / ZREF[s])
                if a.CAP: sig = np.minimum(sig, a.CAP)
        obs = m + sig * rng.standard_normal(len(m))
        cls[s].append([(sig < .3).mean(), ((sig >= .3) & (sig < 1)).mean(), (sig >= 1).mean(), len(m)])
        out[s].append(((obs < 4) | (obs > 7)).mean())
        if k == keys[0]: zs[s] = (z, sig)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5)); lines = []
for s, c in zip(("ZTF", "LSST"), ("C1", "C2")):
    z, sig = zs[s]; ax[0].loglog(z, sig, ".", ms=2, alpha=.3, color=c, label=s)
ax[0].axhline(1, color="k", ls="--"); ax[0].axhline(.3, color="k", ls=":"); ax[0].set_xlabel("z"); ax[0].set_ylabel("sigma_lgM (dex)"); ax[0].legend()
ax[0].set_title("per-event mass sigma (M1)")
x = np.arange(len(keys))
for j, s in enumerate(("ZTF", "LSST")):
    c = np.array(cls[s]); w = .38
    bot = np.zeros(len(keys))
    for i, (col, lab) in enumerate((("tab:green", "<0.3 dex"), ("gold", "0.3-1"), ("tab:red", ">1 dex"))):
        ax[1].bar(x + (j - .5) * w, c[:, i], w, bottom=bot, color=col, edgecolor="k", lw=.3, label=lab if j == 0 else None); bot += c[:, i]
ax[1].set_xticks(x); ax[1].set_xticklabels([MODEL_LABELS[k] for k in keys]); ax[1].set_ylabel("fraction of TDEs"); ax[1].legend(); ax[1].set_title("TDE usability (left ZTF, right LSST)")
for s in ("ZTF", "LSST"): ax[2].plot(x, out[s], "o-", label=f"TDE {s}")
ax[2].plot(x, out["EMRI"], "s-", label="EMRI"); ax[2].set_xticks(x); ax[2].set_xticklabels([MODEL_LABELS[k] for k in keys])
ax[2].set_ylabel("fraction with noisy centre outside [4,7]"); ax[2].legend()
fig.tight_layout(); fig.savefig("plots/usable_events.png", dpi=130)
for s in cls:
    for k, c, o in zip(keys, cls[s], out[s]):
        lines.append(f"{s:5s} {MODEL_LABELS[k]} N={int(c[3]):6d} good={c[0]:.2f} marg={c[1]:.2f} poor={c[2]:.2f} outside_box={o:.2f}")
open("plots/usable_events.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
