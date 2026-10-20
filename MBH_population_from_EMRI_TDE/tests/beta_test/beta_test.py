#!/usr/bin/env python
"""
Test of the beta (penetration factor) distribution, following Luca's comment.

Takes ONE universe's existing TDE parameter catalog (gen_TDE_params.py output),
replaces only `beta` according to each scheme below, runs the SAME trained
detectability classifier as data_generation/calculate_SNR_TDE.py, and counts
detections (same z / time-window cuts, threshold 0.5).

Schemes:
  uniform       : current choice, beta ~ U[1.001, 30]
  stone_metzger : Stone & Metzger 2016 Eq. 28-29. With prob f_pinhole(M) beta=1/D,
                  D~U[1/beta_max,1] (i.e. beta^-2); else beta=1 (here 1.001).
  full_loss_cone: all events beta^-2 on [1.001, beta_max(M)]
  empty_loss_cone: all events beta = 1.001
Run with igwn python:
  python tests/beta_test/beta_test.py --universes 0 1 2
"""
import os, sys, pickle, argparse
import numpy as np, h5py, torch

sys.path.insert(0, os.path.abspath('./'))
sys.path.insert(0, os.path.abspath('../poplar'))
from utils import ASTROPHYSICAL_DATA_DIR, TDE_DATA_FILE_NAME, TRAIN_DIR_TDE

ARCH = {  # (neurons, layers) from EXECUTABLES/config.sh
    ("cooling_envelope", "ZTF"): (512, 4), ("cooling_envelope", "LSST"): (128, 4),
    ("stream_stream_tde", "ZTF"): (256, 3), ("stream_stream_tde", "LSST"): (512, 4),
    ("tde_analytical", "ZTF"): (512, 4), ("tde_analytical", "LSST"): (512, 4),
    ("tde_fallback", "ZTF"): (512, 4), ("tde_fallback", "LSST"): (128, 4),
}
SHARED = ["lgMBH", "z_gal", "ra_rad", "dec_rad", "eta", "alpha", "beta", "t0_mjd_transient"]
EXTRA = {
    "cooling_envelope": [], "stream_stream_tde": [("peak_time", "log"), ("sigma_t", "log")],
    "tde_analytical": [("mej", "log"), ("vej", "log"), ("kappa", "linear"), ("kappa_gamma", "log"),
                       ("temperature_floor", "linear"), ("l0", "log"), ("t_0_turn", "log")],
    "tde_fallback": [("tvisc", "log"), ("bb", "linear"), ("rph0", "log"), ("lphoto", "linear")],
}
KEYMAP = {"lgMBH": "lgMBH_mass"}
WINDOWS = {"ZTF": (58200, 61396), "LSST": (60827, 64423)}
ZMAX = {"ZTF": None, "LSST": None}  # filled from config below if given
BETA_MIN, BETA_MAX_TRAIN = 1.001, 30.0
RSUN_KM, GM_C2_KM = 6.957e5, 1.4766  # R_sun; GM_sun/c^2

p = argparse.ArgumentParser()
p.add_argument("--universes", type=int, nargs="+", default=[0])
p.add_argument("--GALAXIES", type=int, default=100000)
p.add_argument("--mbh", default="use_sigma"); p.add_argument("--nuc", default="hannah")
p.add_argument("--occ", default="uniform");   p.add_argument("--merger", default="oleary21")
p.add_argument("--models", nargs="+", default=list(EXTRA))
p.add_argument("--surveys", nargs="+", default=["ZTF", "LSST"])
p.add_argument("--zmin", type=float, default=0.0); p.add_argument("--zmax", type=float, default=10.0)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--out", default="tests/beta_test/results.npz")
a = p.parse_args()


def beta_max(lgM, mstar):
    rstar = RSUN_KM * mstar**0.8
    rt = rstar * (10**lgM / mstar) ** (1 / 3)
    return np.clip(rt / (2 * GM_C2_KM * 10**lgM), BETA_MIN * 1.0001, None)


def f_pinhole(lgM):
    return np.clip(0.22 * (10**lgM / 1e8) ** -0.307, 0, 1)


def draw_beta(scheme, lgM, mstar, rng):
    n = len(lgM); bmax = np.minimum(beta_max(lgM, mstar), BETA_MAX_TRAIN)
    inv = lambda: 1.0 / rng.uniform(1.0 / bmax, 1.0 / BETA_MIN)   # beta^-2 on [BETA_MIN,bmax]
    if scheme == "uniform": return rng.uniform(BETA_MIN, BETA_MAX_TRAIN, n)
    if scheme == "full_loss_cone": return inv()
    if scheme == "empty_loss_cone": return np.full(n, BETA_MIN)
    if scheme == "stone_metzger":
        return np.where(rng.uniform(size=n) < f_pinhole(lgM), inv(), BETA_MIN)
    raise ValueError(scheme)

SCHEMES = ["uniform", "stone_metzger", "full_loss_cone", "empty_loss_cone"]
rng = np.random.default_rng(a.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
res = {}  # (survey, model, scheme) -> list over universes of n_detected

for survey in a.surveys:
    for model in a.models:
        nn, ly = ARCH[(model, survey)]
        mpath = f"{TRAIN_DIR_TDE}/1E6_events/TDE_DETECT_BCE_no_rescaler_{model}_{nn}_{ly}_10_{survey}/best_final_model/model/model.pth"
        keys = SHARED + [k for k, _ in EXTRA[model]]
        scaling = {}
        with h5py.File(f"{TRAIN_DIR_TDE}/1E6_events/TDE_params_1E6_{survey}_{model}.h5") as f:
            for k in keys:
                d = f[k][:]
                kind = "log" if (k in ("eta", "alpha") or dict(EXTRA[model]).get(k) == "log") else "linear"
                if kind == "log": d = np.log10(d); scaling[k] = (kind, d.min(), d.max(), -1.0, 1.0)
                else: scaling[k] = (kind, d.min(), d.max(), 0.0, 1.0)
        with open(mpath, "rb") as f: net = pickle.load(f)
        net.to(dev).eval()
        t0, t1 = WINDOWS[survey]
        for u in a.universes:
            path = (f"{ASTROPHYSICAL_DATA_DIR}/{a.GALAXIES}/{TDE_DATA_FILE_NAME}_{a.mbh}_{a.nuc}_{a.occ}_{a.merger}_tde_fallback_{u}.h5")
            with h5py.File(path) as f: cat = {k: f[k][:] for k in f.keys() if k != "model"}
            n0 = len(cat["z_gal"]); lu = lambda lo, hi: np.exp(rng.uniform(np.log(lo), np.log(hi), n0))
            for k in ("tvisc", "bb", "rph0", "lphoto"): cat.pop(k, None)
            if model == "stream_stream_tde": cat.update(peak_time=lu(5, 40), sigma_t=lu(5, 30))
            elif model == "tde_analytical":
                cat.update(mej=lu(.1, 5), vej=lu(3e3, 1.5e4), kappa=rng.uniform(.1, 1, n0), kappa_gamma=lu(1e-4, 1e4),
                           temperature_floor=rng.uniform(1e4, 5e4, n0), l0=lu(1e51, 1e60), t_0_turn=lu(1e-4, 500))
            elif model == "tde_fallback":
                cat.update(tvisc=lu(1, 30), bb=rng.uniform(0, 2, n0), rph0=lu(1e-4, 1e4), lphoto=rng.uniform(0, 2, n0))
            m = (cat["z_gal"] >= a.zmin) & (cat["z_gal"] <= a.zmax) & (cat["t0_mjd_transient"] >= t0) & (cat["t0_mjd_transient"] <= t1)
            cat = {k: v[m] for k, v in cat.items()}
            for scheme in SCHEMES:
                cat["beta"] = draw_beta(scheme, cat["lgMBH_mass"], cat["star_mass"], rng)
                X = np.zeros((m.sum(), len(keys)), np.float32)
                for c, k in enumerate(keys):
                    kind, lo, hi, tl, th = scaling[k]
                    raw = cat[KEYMAP.get(k, k)].astype(float)
                    v = np.log10(raw) if kind == "log" else raw
                    X[:, c] = (v - lo) / (hi - lo) * (th - tl) + tl
                with torch.no_grad():
                    pr = torch.sigmoid(net(torch.tensor(X).to(dev))).reshape(-1).cpu().numpy()
                nd = int((pr > 0.5).sum())
                res.setdefault((survey, model, scheme), []).append(nd)
                print(f"{survey:4s} {model:18s} u={u:3d} {scheme:16s} detected={nd:7d}/{m.sum()}  <beta>={cat['beta'].mean():.2f}", flush=True)

print("\n=== mean detections over universes (ratio to uniform) ===")
for survey in a.surveys:
    for model in a.models:
        base = np.mean(res[(survey, model, "uniform")])
        row = "  ".join(f"{s}={np.mean(res[(survey, model, s)]):.0f} ({np.mean(res[(survey, model, s)])/max(base,1e-9):.2f}x)" for s in SCHEMES)
        print(f"{survey:4s} {model:18s} {row}")
np.savez(a.out, **{"|".join(k): np.array(v) for k, v in res.items()})
