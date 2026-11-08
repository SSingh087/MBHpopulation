#!/usr/bin/env python
"""
Deep check of the FIXED-POINTS selection-function scheme for EMRIs (not yet adopted).

  alpha(Lambda) = int p(x|Lambda) P_det(x) dx ~ (1/N_x) sum_j p(x_j|Lambda) P_j / q(x_j)
  x_j : fixed scrambled-Sobol points, x = (lgM, a, z, e_plunge); z log-uniform on [z_lo, z_hi], others uniform on the flow box; q(x) known
  P_j = mean over K nuisance draws (Y0~U(0.1,0.7), T~U(1,6) yr, mu=10) of  ncx2(df=2, nc=rho^2).sf(20^2)  [rho from p0/e0 emulator + SNR network]
  p(x|Lambda) = flow log_prob + logit Jacobian

Tests (each prints PASS/FAIL-style numbers):
  T0  normalisation: P=1  ->  (1/N) sum p/q must be ~ 1   (checks flow density, logit Jacobian, proposal)
  T1  indicator integrals, e.g. P = 1[z<0.5]: fixed-point sum vs plain flow-sampling fraction (no network involved; exact known-answer test)
  T2  convergence in N_x (independent scrambles -> real error bars) for the full P_j
  T3  convergence in K (nuisance draws per point)
  T4  comparison with brute-force Monte Carlo (flow samples + nuisance + network) at the same Lambda
  T5  timings (density throughput, table build, per-Lambda cost)  -> GPU vs CPU decision
  T6  where does P_det live in z (choice of z_hi)
  T7  sensitivity d ln alpha / d ln rho  (what the SNR network's own error does to alpha)

  python check_fixed_points.py --device cuda            # full
  python check_fixed_points.py --device cpu --quick     # smoke test (seconds)
"""
import argparse, functools, json, os, sys, time
import numpy as np
import torch, torch.nn as nn

print = functools.partial(print, flush=True)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"
sys.path.insert(0, f"{ROOT}/flow"); sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, "/home/2673888s/poplar"); sys.path.insert(0, "/home/2673888s/LISA_selection_function/lisa_snr_package")
os.environ.setdefault("POPLAR_PATH", "/home/2673888s/poplar")

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--quick", action="store_true"); ap.add_argument("--n_lambda", type=int, default=6); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--z_lo", type=float, default=1e-3); ap.add_argument("--z_hi", type=float, default=8.0)
ap.add_argument("--proposal", choices=["sobol", "mixture"], default="sobol", help="sobol: box x log-uniform z; mixture: 0.9 x (average of L flow densities over the prior) + 0.1 x box")
ap.add_argument("--repl", type=int, default=6); ap.add_argument("--k_list", type=int, nargs="+", default=[8, 32, 128, 512]); ap.add_argument("--m_list", type=int, nargs="+", default=[14, 16, 18, 20])
ap.add_argument("--L_mix", type=int, default=64)
ap.add_argument("--threshold", type=float, default=20.0); ap.add_argument("--df", type=int, default=2)
a = ap.parse_args()
torch.set_num_threads(a.threads); dev = torch.device(a.device)
Q = a.quick
M_LIST = [10, 12] if Q else a.m_list       # N_x = 2^m
M_MAIN = M_LIST[-1]; REPL = 2 if Q else a.repl; K_MAIN = 8 if Q else 64; K_LIST = [4, 16] if Q else a.k_list
N_MC = 200_000 if Q else 5_000_000; N_IND = 200_000 if Q else 4_000_000

from diagnostics import load_flow
from common import LamScaler
from priors import FIDUCIAL, LAMBDA_NAMES
from snr_model import load_snr_model
from poplar.selection import detection_probability_from_optimal_snr
from astropy.cosmology import Planck18
from scipy.stats import qmc

def sync():
    if dev.type == "cuda": torch.cuda.synchronize()

# ---------------- models
flow, xt, ck = load_flow(f"{DATA}/models/flow_emri_best/best.pt", dev)
LO, HI = xt.lo.to(dev), xt.hi.to(dev); STD = xt.std.to(dev); EPS = xt.EPS
e = torch.load(f"{DATA}/models/detectability/p0e0_emulator/model.pt", map_location=dev, weights_only=False)
layers, i = [], 5
for _ in range(e["layers"]): layers += [nn.Linear(i, e["hidden"]), nn.SiLU()]; i = e["hidden"]
emu = nn.Sequential(*layers, nn.Linear(i, 2)).to(dev); emu.load_state_dict(e["state"]); emu.eval()
xm, xs, ym, ys = [torch.tensor(e[k], dtype=torch.float32, device=dev) for k in ("xm", "xs", "ym", "ys")]
snr = load_snr_model(f"{DATA}/models/detectability/emri_snr/model.pth", device=a.device)
zg = np.linspace(0.0, 8.5, 4001); dlg = torch.tensor(Planck18.luminosity_distance(np.maximum(zg, 1e-4)).to("Mpc").value, dtype=torch.float32, device=dev); zgt = torch.tensor(zg, dtype=torch.float32, device=dev)
def dl_of_z(z):
    z = z.contiguous(); j = torch.bucketize(z, zgt).clamp(1, len(zgt) - 1); f = (z - zgt[j - 1]) / (zgt[j] - zgt[j - 1]); return dlg[j - 1] * (1 - f) + dlg[j] * f
rg = np.linspace(0, a.threshold * 10, 4001); pg = torch.tensor(detection_probability_from_optimal_snr(rg.copy(), a.threshold, a.df), dtype=torch.float32, device=dev); rgt = torch.tensor(rg, dtype=torch.float32, device=dev)
def pdet(rho):
    r = rho.clamp(0, rgt[-1].item()).contiguous(); j = torch.bucketize(r, rgt).clamp(1, len(rgt) - 1); f = (r - rgt[j - 1]) / (rgt[j] - rgt[j - 1]); return pg[j - 1] * (1 - f) + pg[j] * f

# ---------------- building blocks
@torch.no_grad()
def rho_of(x, Y0, T):
    """x (n,4) = lgM, a, z, e_plunge -> optimal SNR (SNR network + p0/e0 emulator)."""
    ep = emu((torch.stack([x[:, 0], x[:, 1], Y0, x[:, 3], T], 1) - xm) / xs) * ys + ym
    e0, p0 = ep[:, 0].clamp(0, 0.95), 10 ** ep[:, 1]
    feat = torch.stack([x[:, 0], torch.ones_like(Y0), x[:, 1], e0, p0, Y0, T], 1)           # M and mu enter as log10 (mu = 10 -> 1)
    return snr.run_on_dataset(feat, n_batches=1, luminosity_distances=dl_of_z(x[:, 2])).reshape(-1)

@torch.no_grad()
def pdet_points(x, K, gen, rho_scale=1.0, block=400_000):
    """P_j: average over K nuisance draws of the chi^2(df) detection probability."""
    out = []
    pts_per_block = max(1, block // K)
    for s in range(0, len(x), pts_per_block):
        xb = x[s:s + pts_per_block].repeat_interleave(K, 0); n = len(xb)
        Y0 = 0.1 + 0.6 * torch.rand(n, device=dev, generator=gen); T = 1.0 + 5.0 * torch.rand(n, device=dev, generator=gen)
        out.append(pdet(rho_of(xb, Y0, T) * rho_scale).reshape(-1, K).mean(1))
    return torch.cat(out)

_box_q = lambda x: torch.where((x[:, 2] >= a.z_lo) & (x[:, 2] <= a.z_hi), 1.0 / (3.0 * x[:, 2].double() * np.log(a.z_hi / a.z_lo)), torch.zeros_like(x[:, 2]).double())
LAM_MIX = None
def make_points(m, seed):
    if a.proposal == "mixture": return make_points_mix(m, seed)
    return make_points_sobol(m, seed)

@torch.no_grad()
def make_points_mix(m, seed):
    """x_j ~ q = 0.9 * (1/L) sum_l p(x|Lambda_l) + 0.1 * box ; L Lambdas drawn once from the prior sample in meta.npz. q evaluated exactly (deterministic mixture)."""
    global LAM_MIX
    if LAM_MIX is None:
        idx = np.random.default_rng(999).choice(len(meta["lam"]), a.L_mix, replace=False); LAM_MIX = [LamScaler().forward(torch.tensor(meta["lam"][i][None].astype(np.float32))).to(dev) for i in idx]
    n = 2 ** m; g = torch.Generator().manual_seed(seed); nb = int(0.1 * n); comp = torch.randint(0, a.L_mix, (n - nb,), generator=g); parts = []
    for l in range(a.L_mix):
        c = int((comp == l).sum())
        if c: parts.append(flow_samples(LAM_MIX[l], c))
    ub = torch.tensor(qmc.Sobol(4, scramble=True, seed=seed).random(nb), dtype=torch.float64); lg = np.log(a.z_hi / a.z_lo)
    xb = torch.stack([4.0 + 3.0 * ub[:, 0], ub[:, 1], a.z_lo * torch.exp(ub[:, 2] * lg), ub[:, 3]], 1).float().to(dev)
    x = torch.cat(parts + [xb])
    lq = torch.stack([logp_x(x, ls) for ls in LAM_MIX], 0); q = 0.9 * torch.exp(torch.logsumexp(lq, 0) - np.log(a.L_mix)) + 0.1 * _box_q(x)
    return x, q

def make_points_sobol(m, seed):
    u = torch.tensor(qmc.Sobol(4, scramble=True, seed=seed).random_base2(m), dtype=torch.float64)
    lg = np.log(a.z_hi / a.z_lo)
    x = torch.stack([4.0 + 3.0 * u[:, 0], u[:, 1], a.z_lo * torch.exp(u[:, 2] * lg), u[:, 3]], 1)
    q = 1.0 / (3.0 * 1.0 * x[:, 2] * lg * 1.0)                                                # density in x: 1/(3) * 1/(1) * 1/(z ln(zhi/zlo)) * 1/(1)
    return x.float().to(dev), q.to(dev)

@torch.no_grad()
def logp_x(x, lam_s, chunk=1_000_000):
    """log p(x|Lambda) in x-space: flow log_prob(u) + log|du/dx| of the logit-standardise transform."""
    out = []
    for s in range(0, len(x), chunk):
        xb = x[s:s + chunk]; u = xt.forward(xb)
        t = ((xb - LO) / (HI - LO)).clamp(EPS, 1 - EPS)
        out.append(flow.log_prob(u, lam_s.expand(len(xb), -1)) + (-torch.log(STD) - torch.log(t * (1 - t)) - torch.log(HI - LO)).sum(1))
    return torch.cat(out).double()

def alpha_hat(lam_s, x, q, P):
    w = torch.exp(logp_x(x, lam_s)) / q; t = w * P.double()
    est = t.mean().item(); ess = (t.sum() ** 2 / (t ** 2).sum()).item() if t.sum() > 0 else 0.0
    return est, t.std().item() / np.sqrt(len(t)), ess

@torch.no_grad()
def flow_samples(lam_s, n, chunk=500_000):
    return torch.cat([xt.inverse(flow.sample(lam_s.expand(min(chunk, n - s), -1))) for s in range(0, n, chunk)])

_PC, _PP = {}, {}
def pts(m, seed):
    if (m, seed) not in _PC: t0 = time.time(); _PC[(m, seed)] = make_points(m, seed); print(f"    [points 2^{m} seed {seed}: {time.time()-t0:.1f}s]")
    return _PC[(m, seed)]
def P_of(m, seed, K):
    if (m, seed, K) not in _PP: _PP[(m, seed, K)] = pdet_points(pts(m, seed)[0], K, torch.Generator(device=dev).manual_seed(1000 * seed + K))
    return _PP[(m, seed, K)]

# ---------------- Lambda set
meta = np.load(f"{DATA}/simulations/consolidated/meta.npz"); rng = np.random.default_rng(a.seed)
lams = [np.array([FIDUCIAL[k] for k in LAMBDA_NAMES], dtype=np.float32)] + [meta["lam"][j].astype(np.float32) for j in rng.choice(len(meta["lam"]), a.n_lambda - 1, replace=False)]
LAM = [LamScaler().forward(torch.tensor(l[None])).to(dev) for l in lams]
print(f"device {dev}, quick={Q}; Lambda set (A,B,scatter,logM0,delta,theta):")
for k, l in enumerate(lams): print(f"  L{k}: {np.round(l, 3).tolist()}")
R = {}
gen = torch.Generator(device=dev).manual_seed(123)

# ======== T0 normalisation (P = 1)
print("\n[T0] normalisation  (1/N) sum p/q  with P=1  (should be ~1; deficit = mass outside z in [%.0e, %g])" % (a.z_lo, a.z_hi))
x, q = pts(M_MAIN, 100); one = torch.ones(len(x), device=dev); R["T0"] = []
for k, ls in enumerate(LAM):
    est, err, ess = alpha_hat(ls, x, q, one); R["T0"].append(est); print(f"  L{k}: {est:.5f} +- {err:.5f}  (ESS {ess:.0f} of {len(x)})")

# ======== T1 indicator integrals vs flow sampling
print("\n[T1] indicator integrals: fixed-point sum (%d scrambles) vs plain flow-sampling fraction (N=%d)" % (REPL, N_IND))
R["T1"] = []
INDS = {"z<0.5": lambda x: (x[:, 2] < 0.5), "lgM>6.5 & z<1 & a>0.5": lambda x: (x[:, 0] > 6.5) & (x[:, 2] < 1.0) & (x[:, 1] > 0.5), "z<0.05": lambda x: (x[:, 2] < 0.05)}
for k, ls in enumerate(LAM):
    xs_ = flow_samples(ls, N_IND)
    for nm, f in INDS.items():
        mc = f(xs_).float().mean().item(); mc_err = np.sqrt(max(mc * (1 - mc), 1e-12) / N_IND)
        ests = []
        for r in range(REPL):
            xr, qr = pts(M_MAIN, 100 + r); ests.append(alpha_hat(ls, xr, qr, f(xr).float())[0])
        fp, fpe = np.mean(ests), np.std(ests, ddof=1) / np.sqrt(REPL)
        R["T1"].append(dict(lam=k, ind=nm, mc=mc, mc_err=mc_err, fp=fp, fp_err=fpe))
        print(f"  L{k} {nm:24s} MC {mc:.5f} +- {mc_err:.5f}   fixed-points {fp:.5f} +- {fpe:.5f}   rel diff {(fp - mc) / max(mc, 1e-30):+.2%}")

# ======== T2 convergence in N_x (full P_j), T5 timings
print("\n[T2] alpha_hat vs N_x, K=%d, %d independent scrambled-Sobol replicates (error bar = spread across replicates)" % (K_MAIN, REPL))
R["T2"] = {}; tbl_time = {}
for m in M_LIST:
    ests = [[] for _ in LAM]; sync(); t0 = time.time(); tp = 0.0; tl = 0.0
    for r in range(REPL):
        x, q = pts(m, 100 + r); sync(); t1 = time.time(); P = P_of(m, 100 + r, K_MAIN); sync(); tp += time.time() - t1
        for k, ls in enumerate(LAM):
            sync(); t2 = time.time(); ests[k].append(alpha_hat(ls, x, q, P)[0]); sync(); tl += time.time() - t2
    tbl_time[m] = (tp / REPL, tl / REPL / len(LAM))
    R["T2"][m] = [(float(np.mean(v)), float(np.std(v, ddof=1) / np.mean(v)) if np.mean(v) > 0 else float("nan")) for v in ests]
    print(f"  N_x=2^{m}={2**m:8d}: " + "  ".join(f"L{k}: {v[0]:.4e} ({v[1]:.2%})" for k, v in enumerate(R['T2'][m])) + f"   [table {tbl_time[m][0]:.1f}s, density+sum/Lambda {tbl_time[m][1]*1e3:.0f} ms]")
print("  (the parenthesised number is the relative scatter of ONE alpha_hat; the mean of REPL replicates is better by sqrt(REPL))")

# ======== T3 K convergence
print("\n[T3] alpha_hat vs K at N_x=2^%d  (4 replicates)" % M_MAIN)
R["T3"] = {}
for K in K_LIST:
    ests = [[] for _ in LAM]
    for r in range(min(4, REPL) if not Q else 2):
        x, q = pts(M_MAIN, 100 + r); P = P_of(M_MAIN, 100 + r, K)
        for k, ls in enumerate(LAM): ests[k].append(alpha_hat(ls, x, q, P)[0])
    R["T3"][K] = [(float(np.mean(v)), float(np.std(v, ddof=1) / np.mean(v)) if np.mean(v) > 0 else float("nan")) for v in ests]
    print(f"  K={K:4d}: " + "  ".join(f"L{k}: {v[0]:.4e} ({v[1]:.2%})" for k, v in enumerate(R['T3'][K])))

# ======== T4 brute force
print("\n[T4] brute-force MC (flow samples + nuisance + network, N=%d per Lambda) vs fixed points (N_x=2^%d, K=%d, mean of %d scrambles)" % (N_MC, M_MAIN, K_MAIN, REPL))
R["T4"] = []
for k, ls in enumerate(LAM):
    sync(); t0 = time.time(); xs_ = flow_samples(ls, N_MC); p = pdet_points(xs_, 1, gen); sync(); tmc = time.time() - t0
    mc, mce = p.mean().item(), p.std().item() / np.sqrt(N_MC); fp = R["T2"][M_MAIN][k][0]; fpe = R["T2"][M_MAIN][k][1] * fp / np.sqrt(REPL)
    R["T4"].append(dict(lam=k, mc=mc, mc_err=mce, fp=fp, fp_err=fpe, t_mc=tmc)); sig = (fp - mc) / np.hypot(mce, fpe)
    print(f"  L{k}: MC {mc:.4e} +- {mce:.1e} ({mce/max(mc,1e-30):.1%}, {tmc:.0f}s)   fixed {fp:.4e} +- {fpe:.1e}   diff {(fp-mc)/max(mc,1e-30):+.2%} = {sig:+.1f} sigma")

# ======== T6 z-profile of P_j
print("\n[T6] max and mean P_det per z bin (largest table)  -> choose z_hi where max P < ~1e-8")
x, q = pts(M_MAIN, 100); P = P_of(M_MAIN, 100, K_MAIN); zz = x[:, 2].cpu().numpy(); Pn = P.cpu().numpy(); edges = np.geomspace(a.z_lo, a.z_hi, 13); R["T6"] = []
for lo_, hi_ in zip(edges[:-1], edges[1:]):
    s = (zz >= lo_) & (zz < hi_)
    if s.any(): R["T6"].append((lo_, hi_, float(Pn[s].max()), float(Pn[s].mean()))); print(f"  z {lo_:7.3f}-{hi_:7.3f}: max P {Pn[s].max():.3e}  mean P {Pn[s].mean():.3e}")

# ======== T7 sensitivity to rho scale
print("\n[T7] sensitivity to a rescaling of the SNR network output rho (N_x=2^%d, K=%d)  -> d ln alpha / d ln rho" % (M_MAIN, K_MAIN))
R["T7"] = []; sc = {0.79: None, 1.0: None, 1.26: None}
for s_ in sc:
    gen2 = torch.Generator(device=dev).manual_seed(7); sc[s_] = pdet_points(x, K_MAIN, gen2, rho_scale=s_)
for k, ls in enumerate(LAM):
    al = {s_: alpha_hat(ls, x, q, sc[s_])[0] for s_ in sc}; slope = (np.log(al[1.26]) - np.log(al[0.79])) / (np.log(1.26) - np.log(0.79))
    R["T7"].append(slope); print(f"  L{k}: alpha(rho x0.79,1,1.26) = {al[0.79]:.3e}, {al[1.0]:.3e}, {al[1.26]:.3e}   d ln a/d ln rho = {slope:.2f}   (network 0.083 dex = x1.21 -> {abs(slope)*np.log(1.21):.0%} in alpha)")

# ======== T5 summary of timings
print("\n[T5] timings on %s:" % dev)
for m in M_LIST:
    tt, tl = tbl_time[m]; print(f"  N_x=2^{m}: one-off table (K={K_MAIN}) {tt:.1f}s ; per-Lambda density+sum {tl*1e3:.0f} ms  ->  1e5 Lambda: {tl*1e5/3600:.2f} h ; 1e6 Lambda: {tl*1e6/3600:.1f} h on ONE {dev.type} job")
print("  brute force per Lambda (N=%d): %.0f s" % (N_MC, np.mean([r['t_mc'] for r in R['T4']])))
json.dump(R, open(f"{DATA}/logs/check_fixed_points_{dev.type}{'_quick' if Q else ''}.json", "w"), indent=1, default=float)
print("done")
