"""Plot the beta distributions of the different beta sets."""
import sys, numpy as np, h5py, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "../poplar"); sys.path.insert(0, "..")
from utils import ASTROPHYSICAL_DATA_DIR, TDE_DATA_FILE_NAME
BMIN, BMAX, RS, G = 1.001, 30.0, 6.957e5, 1.4766
p = f"{ASTROPHYSICAL_DATA_DIR}/100000/{TDE_DATA_FILE_NAME}_use_sigma_hannah_uniform_oleary21_tde_fallback_0.h5"
with h5py.File(p) as f: lgM, ms = f["lgMBH_mass"][:], f["star_mass"][:]
rng = np.random.default_rng(1); n = len(lgM)
rt = RS*ms**.8*(10**lgM/ms)**(1/3)
bmax = np.minimum(np.clip(rt/(2*G*10**lgM), BMIN*1.0001, None), BMAX)
inv = lambda: 1/rng.uniform(1/bmax, 1/BMIN)
fp = np.clip(.22*(10**lgM/1e8)**-.307, 0, 1)
B = {"uniform": rng.uniform(BMIN, BMAX, n), "stone_metzger": np.where(rng.uniform(size=n) < fp, inv(), BMIN),
     "full_loss_cone": inv(), "empty_loss_cone": np.full(n, BMIN)}
c = dict(zip(B, ["gray", "C0", "C1", "C2"]))
plt.rcParams["text.usetex"] = False
d = np.load("results_fallback.npz")
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
bins = np.linspace(1, 30, 59)
for k, v in B.items():
    if k == "empty_loss_cone": ax[0].axvline(1, c=c[k], label=f"{k} (all β=1)"); continue
    ax[0].hist(v, bins, density=True, histtype="step", lw=2, color=c[k], label=f"{k}  <β>={v.mean():.1f}")
ax[0].set_yscale("log"); ax[0].set_xlabel("β"); ax[0].set_ylabel("pdf"); ax[0].set_title("β sets"); ax[0].legend(fontsize=8)
for a, sv in zip(ax[1:], ["ZTF", "LSST"]):
    for i, k in enumerate(B):
        y = d[f"{sv}|tde_fallback|{k}"]
        a.plot(np.arange(len(y)), y, "o-", ms=3, c=c[k], label=f"{k} ({y.mean():.0f})")
    a.set_xlabel("universe"); a.set_ylabel("detected TDEs"); a.set_title(f"{sv} tde_fallback"); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig("beta_sets.png", dpi=140)
