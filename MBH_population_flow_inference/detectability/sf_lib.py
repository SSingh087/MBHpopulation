"""sf_lib.py -- shared pieces of the selection-function (SF) validation pipeline.

EMRI: x = (lgM, a, z, e_plunge) ~ flow p(x|Lambda);  nuisance Y0~U(0.1,0.7), T~U(1,6) yr, mu=10 Msun;
      (lgM,a,Y0,e_plunge,T) -> (e0,p0) emulator -> SNR network rho_net;  P_det = ncx2(df=2, nc=rho^2).sf(threshold^2)  (poplar/selection.py)
      "calibrated" variant: rho_true = rho_net * 10^(-r), r drawn from the network's OUT-OF-SAMPLE residual distribution in the cell (lgM, e0, rho_net)
Fixed-points estimator: alpha(Lambda) = (1/N) sum_j p(x_j|Lambda) P_j / q(x_j), x_j ~ q = 0.9*(1/L) sum_l p(x|Lambda_l) + 0.1*box.
"""
import os, sys, time, json
import numpy as np
import torch, torch.nn as nn
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"
OOS = "/data/wiay/postgrads/shashwat/LISA_SF_DATA/injection_param_with_snr_another_100k_clean.h5"
sys.path.insert(0, f"{ROOT}/flow"); sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, "/home/2673888s/poplar"); sys.path.insert(0, "/home/2673888s/LISA_selection_function/lisa_snr_package")
os.environ.setdefault("POPLAR_PATH", "/home/2673888s/poplar")
torch.set_grad_enabled(False)

from diagnostics import load_flow
from common import LamScaler, load_consolidated
from priors import FIDUCIAL, LAMBDA_NAMES
from snr_model import load_snr_model, build_features, luminosity_distance
from poplar.selection import detection_probability_from_optimal_snr
from astropy.cosmology import Planck18

LGM_E = np.array([0, 4.5, 5.5, 6.5, 99.0]); E0_E = np.array([0, 0.3, 0.6, 0.9, 1.01]); Z_E = np.array([0, 0.1, 0.3, 0.7, 1.5, 3.0, 99.0])   # residual-model cells: (lgM, e0, z).  NOT conditioned on rho_net: the injection rho-distribution is much flatter than a population's, which would bias r
POOL = 512


def lambda_set(meta, n, seed=0):
    rng = np.random.default_rng(seed)
    L = [np.array([FIDUCIAL[k] for k in LAMBDA_NAMES], dtype=np.float32)] + [meta["lam"][j].astype(np.float32) for j in rng.choice(len(meta["lam"]), n - 1, replace=False)]
    return L


def lam_scaled(l, dev): return LamScaler().forward(torch.tensor(np.asarray(l, dtype=np.float32)[None])).to(dev)


class EMRISF:
    def __init__(self, device="cuda", threshold=20.0, df=2, resid_file=None):
        self.dev = dev = torch.device(device); self.thr, self.df = threshold, df
        self.flow, self.xt, _ = load_flow(f"{DATA}/models/flow_emri_best/best.pt", dev)
        self.LO, self.HI, self.STD, self.EPS = self.xt.lo.to(dev), self.xt.hi.to(dev), self.xt.std.to(dev), self.xt.EPS
        e = torch.load(f"{DATA}/models/detectability/p0e0_emulator/model.pt", map_location=dev, weights_only=False)
        layers, i = [], 5
        for _ in range(e["layers"]): layers += [nn.Linear(i, e["hidden"]), nn.SiLU()]; i = e["hidden"]
        self.emu = nn.Sequential(*layers, nn.Linear(i, 2)).to(dev); self.emu.load_state_dict(e["state"]); self.emu.eval()
        self.xm, self.xs, self.ym, self.ys = [torch.tensor(e[k], dtype=torch.float32, device=dev) for k in ("xm", "xs", "ym", "ys")]
        self.snr = load_snr_model(f"{DATA}/models/detectability/emri_snr/model.pth", device=device)
        zg = np.linspace(0.0, 8.5, 4001); self.zg = torch.tensor(zg, dtype=torch.float32, device=dev)
        self.dlg = torch.tensor(Planck18.luminosity_distance(np.maximum(zg, 1e-4)).to("Mpc").value, dtype=torch.float32, device=dev)
        rg = np.linspace(0, threshold * 10, 4001); self.rg = torch.tensor(rg, dtype=torch.float32, device=dev)
        self.pg = torch.tensor(detection_probability_from_optimal_snr(rg.copy(), threshold, df), dtype=torch.float32, device=dev)
        self.resid = None
        if resid_file and os.path.exists(resid_file):
            r = np.load(resid_file); self.resid = dict(pool=torch.tensor(r["pool"], dtype=torch.float32, device=dev), fallback=torch.tensor(r["fallback"], dtype=torch.float32, device=dev), ok=torch.tensor(r["ok"], device=dev))

    # ---- detection probability from rho
    def pdet(self, rho):
        r = rho.clamp(0, self.rg[-1].item()).contiguous(); j = torch.bucketize(r, self.rg).clamp(1, len(self.rg) - 1)
        f = (r - self.rg[j - 1]) / (self.rg[j] - self.rg[j - 1]); return self.pg[j - 1] * (1 - f) + self.pg[j] * f

    def dl_of_z(self, z):
        z = z.contiguous(); j = torch.bucketize(z, self.zg).clamp(1, len(self.zg) - 1); f = (z - self.zg[j - 1]) / (self.zg[j] - self.zg[j - 1]); return self.dlg[j - 1] * (1 - f) + self.dlg[j] * f

    def rho_of(self, x, Y0, T):
        """x (n,4)=(lgM,a,z,e_plunge) -> (rho_net, e0)"""
        ep = self.emu((torch.stack([x[:, 0], x[:, 1], Y0, x[:, 3], T], 1) - self.xm) / self.xs) * self.ys + self.ym
        e0, p0 = ep[:, 0].clamp(0, 0.95), 10 ** ep[:, 1]
        feat = torch.stack([x[:, 0], torch.ones_like(Y0), x[:, 1], e0, p0, Y0, T], 1)
        return self.snr.run_on_dataset(feat, n_batches=1, luminosity_distances=self.dl_of_z(x[:, 2])).reshape(-1), e0

    # ---- residual model
    def cell(self, lgM, e0, z):
        b = lambda v, E: torch.bucketize(v, torch.tensor(E[1:-1], dtype=torch.float32, device=self.dev))
        return (b(lgM, LGM_E) * 4 + b(e0, E0_E)) * 6 + b(z, Z_E), b(z, Z_E)

    def draw_r(self, lgM, e0, z, gen):
        c, rb = self.cell(lgM, e0, z); k = torch.randint(0, POOL, (len(z),), device=self.dev, generator=gen)
        use = self.resid["ok"][c]
        return torch.where(use, self.resid["pool"][c, k], self.resid["fallback"][rb, k])

    # ---- P_j: mean over K nuisance draws
    def pdet_points(self, x, K, gen, calibrated=False, block=400_000):
        out = []
        pb = max(1, block // K)
        for s in range(0, len(x), pb):
            xb = x[s:s + pb].repeat_interleave(K, 0); n = len(xb)
            Y0 = 0.1 + 0.6 * torch.rand(n, device=self.dev, generator=gen); T = 1.0 + 5.0 * torch.rand(n, device=self.dev, generator=gen)
            rho, e0 = self.rho_of(xb, Y0, T)
            if calibrated: rho = rho * 10 ** (-self.draw_r(xb[:, 0], e0, xb[:, 2], gen))
            out.append(self.pdet(rho).reshape(-1, K).mean(1))
        return torch.cat(out)

    # ---- flow
    def logp_x(self, x, lam_s, chunk=1_000_000):
        out = []
        for s in range(0, len(x), chunk):
            xb = x[s:s + chunk]; u = self.xt.forward(xb); t = ((xb - self.LO) / (self.HI - self.LO)).clamp(self.EPS, 1 - self.EPS)
            out.append(self.flow.log_prob(u, lam_s.expand(len(xb), -1)) + (-torch.log(self.STD) - torch.log(t * (1 - t)) - torch.log(self.HI - self.LO)).sum(1))
        return torch.cat(out).double()

    def flow_samples(self, lam_s, n, chunk=500_000):
        return torch.cat([self.xt.inverse(self.flow.sample(lam_s.expand(min(chunk, n - s), -1))) for s in range(0, n, chunk)])

    def make_points_mix(self, m, seed, lam_pool, z_lo=1e-3, z_hi=8.0, box_frac=0.1):
        """x_j ~ q = (1-box)*(1/L)sum_l p(x|Lambda_l) + box*(log-z box).  Returns x (float32, dev), q (float64, dev)."""
        n = 2 ** m; g = torch.Generator().manual_seed(seed); nb = int(box_frac * n); L = len(lam_pool)
        comp = torch.randint(0, L, (n - nb,), generator=g); parts = []
        for l in range(L):
            c = int((comp == l).sum())
            if c: parts.append(self.flow_samples(lam_pool[l], c))
        ub = torch.tensor(qmc.Sobol(4, scramble=True, seed=seed).random(nb), dtype=torch.float64); lg = np.log(z_hi / z_lo)
        xb = torch.stack([4.0 + 3.0 * ub[:, 0], ub[:, 1], z_lo * torch.exp(ub[:, 2] * lg), ub[:, 3]], 1).float().to(self.dev)
        x = torch.cat(parts + [xb])
        lq = torch.stack([self.logp_x(x, ls) for ls in lam_pool], 0)
        box = torch.where((x[:, 2] >= z_lo) & (x[:, 2] <= z_hi), 1.0 / (3.0 * x[:, 2].double() * lg), torch.zeros_like(x[:, 2]).double())
        return x, (1 - box_frac) * torch.exp(torch.logsumexp(lq, 0) - np.log(L)) + box_frac * box

    def alphas(self, lam_s, tab, nblock=8):
        """naive and calibrated alpha from ONE density evaluation: returns (a_n, e_n, a_c, e_c)"""
        w = torch.exp(self.logp_x(tab["x"], lam_s)) / tab["q"]; o = []
        for key in ("P_naive", "P_cal"):
            t = w * tab[key].double(); o += [t.mean().item(), (t.std() / np.sqrt(len(t))).item()]   # points are independent draws from q (the 10% box part is a scrambled Sobol set, no worse than iid)
        return tuple(o)

    def alpha(self, lam_s, tab, key="P_naive", nblock=8):
        """fixed-points alpha and its standard error std(t)/sqrt(N) (the points are independent draws from q).  The earlier block error took blocks of the UNSHUFFLED table
        (blocks = different flow components of the mixture), which overstated the error."""
        t = torch.exp(self.logp_x(tab["x"], lam_s)) / tab["q"] * tab[key].double()
        return t.mean().item(), (t.std() / np.sqrt(len(t))).item(), t


def load_table(path, dev):
    d = np.load(path); return {k: torch.tensor(d[k], device=dev) for k in d.files}


def build_resid_model(net_rho_fn, out_file, max_rows=None):
    """Fit the residual pools r = log10(rho_net/rho_true) per (lgM, e0, rho_net) cell on the OUT-OF-SAMPLE injections (not used to train the network)."""
    import h5py
    f = h5py.File(OOS); d = {k: f[k][:] for k in ["M", "mu", "a", "e0", "p0", "Y0", "t_back", "z", "SNR"]}
    X = np.stack([d[k] for k in ["M", "mu", "a", "e0", "p0", "Y0", "t_back"]], 1)
    dl = torch.tensor(luminosity_distance(d["z"]), dtype=torch.float32)
    rho = net_rho_fn(X, dl)
    t = d["SNR"]; ok = np.isfinite(t) & (t > 0) & np.isfinite(rho) & (rho > 0)
    r = np.log10(rho[ok] / t[ok]); lgM = np.log10(d["M"][ok]); e0 = d["e0"][ok]; zz = d["z"][ok]
    ci = (np.digitize(lgM, LGM_E[1:-1]) * 4 + np.digitize(e0, E0_E[1:-1])) * 6 + np.digitize(zz, Z_E[1:-1]); rb = np.digitize(zz, Z_E[1:-1])
    rng = np.random.default_rng(0); ncell = 4 * 4 * 6
    pool = np.zeros((ncell, POOL), np.float32); okc = np.zeros(ncell, bool); cnt = np.zeros(ncell, int); fb = np.zeros((6, POOL), np.float32)
    for c in range(ncell):
        s = r[ci == c]; cnt[c] = len(s)
        if len(s) >= 30: pool[c] = rng.choice(s, POOL); okc[c] = True
    for b in range(6): fb[b] = rng.choice(r[rb == b], POOL)
    np.savez(out_file, pool=pool, fallback=fb, ok=okc, counts=cnt, n_rows=int(ok.sum()))
    return dict(n_rows=int(ok.sum()), cells_ok=int(okc.sum()), cells_total=ncell, counts=cnt.tolist())
