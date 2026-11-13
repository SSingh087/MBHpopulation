#!/usr/bin/env python
"""
Throughput benchmark of the EMRI selection-function pipeline, per stage, on CPU or GPU.

  for one Lambda:   x ~ flow p(x|Lambda)   (lgM, a, z, e_plunge)
                    nuisance: Y0 ~ U(-1,1), T ~ U(0.9, 6.2) yr, mu = 10 Msun        (BENCHMARK ONLY: check against source_sim before production)
                    (e_plunge, T, lgM, a, Y0) -> (e0, p0)    p0/e0 emulator
                    (M, mu, a, e0, p0, Y0, T) -> SNR*D_L     SNR network (poplar LinearModel);  / D_L(z)
                    P_det(rho) = ncx2(df=2, nc=rho^2).sf(20^2)   -> lookup table (exactly what poplar's detection_probability_from_optimal_snr evaluates)
                    alpha(Lambda) = mean P_det

  python bench_selection.py --device cpu --threads 8
  python bench_selection.py --device cuda
Prints sources/s per stage, a check of the lookup table against poplar/scipy, the fiducial alpha with its Monte Carlo error, and the projected
cost of an emulator training set (N_lambda x N_sources).
"""
import argparse, os, sys, time, json
import numpy as np
import torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"
sys.path.insert(0, f"{ROOT}/flow"); sys.path.insert(0, "/home/2673888s/poplar"); sys.path.insert(0, "/home/2673888s/LISA_selection_function/lisa_snr_package")
os.environ.setdefault("POPLAR_PATH", "/home/2673888s/poplar")

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--n", type=int, nargs="+", default=[10_000, 100_000], help="batch sizes to time")
ap.add_argument("--threshold", type=float, default=20.0); ap.add_argument("--df", type=int, default=2)
ap.add_argument("--n_lambda", type=int, default=100_000); ap.add_argument("--n_src", type=int, default=200_000)
a = ap.parse_args()
torch.set_num_threads(a.threads); dev = torch.device(a.device)

from diagnostics import load_flow
from common import LamScaler
from snr_model import load_snr_model, build_features
from poplar.selection import detection_probability_from_optimal_snr
from astropy.cosmology import Planck18

def sync():
    if dev.type == "cuda": torch.cuda.synchronize()
def timed(fn, reps=3):
    fn(); sync(); t = time.time()
    for _ in range(reps): out = fn()
    sync(); return (time.time() - t) / reps, out

# ---------------- models
flow, xt, ck = load_flow(f"{DATA}/models/flow_emri_best/best.pt", dev)
meta = np.load(f"{DATA}/simulations/consolidated/meta.npz"); lam = torch.tensor(meta["lam"][:1], dtype=torch.float32)   # first prior draw as the 'fiducial' Lambda
lam_s = LamScaler().forward(lam).to(dev)
e = torch.load(f"{DATA}/models/detectability/p0e0_emulator/model.pt", map_location=dev, weights_only=False)
layers, i = [], 5
for _ in range(e["layers"]): layers += [nn.Linear(i, e["hidden"]), nn.SiLU()]; i = e["hidden"]
emu = nn.Sequential(*layers, nn.Linear(i, 2)).to(dev); emu.load_state_dict(e["state"]); emu.eval()
xm, xs, ym, ys = [torch.tensor(e[k], dtype=torch.float32, device=dev) for k in ("xm", "xs", "ym", "ys")]
snr = load_snr_model(f"{DATA}/models/detectability/emri_snr/model.pth", device=a.device)
zg = np.linspace(0.0, 8.5, 4001); dlg = torch.tensor(Planck18.luminosity_distance(np.maximum(zg, 1e-4)).to("Mpc").value, dtype=torch.float32, device=dev)
zgt = torch.tensor(zg, dtype=torch.float32, device=dev)
def dl_of_z(z):                                    # linear interpolation on a dense grid (astropy per call would dominate)
    j = torch.bucketize(z, zgt).clamp(1, len(zgt) - 1); f = (z - zgt[j - 1]) / (zgt[j] - zgt[j - 1]); return dlg[j - 1] * (1 - f) + dlg[j] * f
# P_det(rho) lookup: exact survival function of a non-central chi^2 (df) at threshold^2, evaluated once with poplar/scipy
rg = np.linspace(0, a.threshold * 10, 4001)
pg = torch.tensor(detection_probability_from_optimal_snr(rg.copy(), a.threshold, a.df), dtype=torch.float32, device=dev); rgt = torch.tensor(rg, dtype=torch.float32, device=dev)
def pdet(rho):
    r = rho.clamp(0, rgt[-1].item()); j = torch.bucketize(r, rgt).clamp(1, len(rgt) - 1); f = (r - rgt[j - 1]) / (rgt[j] - rgt[j - 1]); return pg[j - 1] * (1 - f) + pg[j] * f

# ---------------- stages
@torch.no_grad()
def s_flow(n): return xt.inverse(flow.sample(lam_s.expand(n, -1)))
@torch.no_grad()
def s_nuis(n, x):
    g = torch.Generator(device=dev).manual_seed(0)
    return 2 * torch.rand(n, device=dev, generator=g) - 1, 0.9 + 5.3 * torch.rand(n, device=dev, generator=g)
@torch.no_grad()
def s_emu(x, Y0, T): return emu((torch.stack([x[:, 0], x[:, 1], Y0, x[:, 3], T], 1) - xm) / xs) * ys + ym
@torch.no_grad()
def s_snr(x, Y0, T, ep):
    e0, p0 = ep[:, 0].clamp(0, 0.95), 10 ** ep[:, 1]
    feat = build_features(np.stack([10 ** x[:, 0].cpu().numpy(), np.full(len(x), 10.0), x[:, 1].cpu().numpy(), e0.cpu().numpy(), p0.cpu().numpy(), Y0.cpu().numpy(), T.cpu().numpy()], 1))
    out = snr.run_on_dataset(feat.to(dev), n_batches=max(1, len(x) // 50_000), luminosity_distances=dl_of_z(x[:, 2].contiguous()).to(dev)).to(dev)
    return out.ravel()

res = {}
for n in a.n:
    r = {}
    t, x = timed(lambda: s_flow(n)); r["flow_sample"] = n / t
    Y0, T = s_nuis(n, x)
    t, ep = timed(lambda: s_emu(x, Y0, T)); r["p0e0_emulator"] = n / t
    t, rho = timed(lambda: s_snr(x, Y0, T, ep), reps=2); r["snr_network(+features)"] = n / t
    t, p = timed(lambda: pdet(rho)); r["pdet_lookup"] = n / t
    tot = 1.0 / sum(1.0 / v for v in r.values()); r["WHOLE_PIPELINE"] = tot
    res[n] = r
    print(f"\nN = {n} on {a.device} ({a.threads} threads):")
    for k, v in r.items(): print(f"  {k:26s} {v:12.3e} sources/s")

# ---------------- accuracy of the lookup + fiducial alpha with MC error
n = max(a.n); x = s_flow(n); Y0, T = s_nuis(n, x); ep = s_emu(x, Y0, T); rho = s_snr(x, Y0, T, ep); p = pdet(rho)
ref = detection_probability_from_optimal_snr(rho[:20000].cpu().numpy().astype(float).copy(), a.threshold, a.df)
print(f"\nlookup vs poplar/scipy on 20000 sources: max |dP| = {np.abs(p[:20000].cpu().numpy() - ref).max():.2e}")
alpha = p.mean().item(); err = p.std().item() / np.sqrt(n)
print(f"fiducial Lambda {lam.numpy().round(3).tolist()[0]}: alpha = {alpha:.4e} +- {err:.1e} (N={n}, rel. MC error {err/max(alpha,1e-30):.1%});  rho_opt median {rho.median().item():.2f}, "
      f"frac(rho>20) = {(rho > 20).float().mean().item():.4f}")
rate = res[max(a.n)]["WHOLE_PIPELINE"]; tot = a.n_lambda * a.n_src
print(f"\nprojection: {a.n_lambda} Lambda x {a.n_src} sources = {tot:.2e} evaluations -> {tot/rate/3600:.1f} h on ONE {a.device} job; "
      f"{tot/rate/3600/50:.2f} h over 50 such jobs")
json.dump({str(k): v for k, v in res.items()}, open(f"{DATA}/logs/bench_selection_{a.device}.json", "w"), indent=1)
