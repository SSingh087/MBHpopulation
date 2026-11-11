"""tde_pop_lib.py -- pieces of the TDE population-inference likelihood (stage 5).

Per-event likelihood (rate marginalised, i.e. shape only):
    l_i(Lambda) = [ int p(lgM,z|Lambda) P(lgM,z) N((lgM,z)_hat_i ; (lgM,z), C_i) dlgM dz ] / alpha(Lambda),   alpha = int p P
    ln L(Lambda) = sum_i ln l_i
p = flow density, P = classifier detection probability marginalised over the nuisance parameters (soft: P = mean p_clf), both on the fixed
(lgM, ln z) grid of the selection-function validation (simulations/sf/tde_grid_<survey>.npz, bilinearly interpolated onto a grid twice as fine).
The Gaussian kernels depend on the data only, so they are computed ONCE; one Lambda evaluation = one flow pass over the grid + one mat-vec.
C_i = the (lgM, z) block of the event's Fisher posterior covariance, from the error flow conditioned on the noisy estimate.
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (f"{ROOT}/flow", f"{ROOT}/src", f"{ROOT}/detectability/fisher"): sys.path.insert(0, p)
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"
from priors import LAMBDA_NAMES, PRIOR_BOX, FIDUCIAL
from common import LamScaler
from diagnostics import load_flow
from flows import ConditionalFlow

ZMAX = {"ZTF": 0.5, "LSST": 4.0}
NP = 7; TRI = np.tril_indices(NP, -1)


def lam_array(lam):
    return np.array([lam[k] for k in LAMBDA_NAMES], dtype=np.float32)


class TDEPopulation:
    """flow density p(lgM,z|Lambda) and detection map P(lgM,z) on the fixed integration grid."""
    def __init__(self, survey, device="cpu", soft=True, refine=2):
        from scipy.interpolate import RegularGridInterpolator
        self.survey, self.dev = survey, torch.device(device); zmax = ZMAX[survey]
        self.flow, xt, _ = load_flow(f"{DATA}/models/flow_tde_best/best.pt", self.dev); self.xt = xt
        self.LO, self.HI, self.STD, self.EPS = xt.lo.to(self.dev), xt.hi.to(self.dev), xt.std.to(self.dev), xt.EPS
        g = np.load(f"{DATA}/simulations/sf/tde_grid_{survey}.npz"); lgc, zc = g["lgc"], g["zc"]; nl, nz = refine * len(lgc), refine * len(zc)
        P = g["Psoft" if soft else "Pcut"]
        f = RegularGridInterpolator((lgc, np.log(zc)), P, bounds_error=False, fill_value=None)
        lgf = np.linspace(4, 7, nl + 1); zlf = np.linspace(np.log(1e-3), np.log(zmax), nz + 1)
        Gl, Gz = np.meshgrid(0.5 * (lgf[1:] + lgf[:-1]), np.exp(0.5 * (zlf[1:] + zlf[:-1])), indexing="ij")
        self.W = (np.diff(lgf)[:, None] * (Gz * np.diff(zlf)[None, :])).reshape(-1)
        self.lgM, self.z = Gl.reshape(-1), Gz.reshape(-1)
        self.P = np.clip(f(np.stack([Gl.reshape(-1), np.log(Gz.reshape(-1))], 1)), 0, 1)
        self.pts = torch.tensor(np.stack([self.lgM, self.z], 1), dtype=torch.float32, device=self.dev)
        self.WP = torch.tensor(self.W * self.P, dtype=torch.float64, device=self.dev)
        self.scaler = LamScaler()

    @torch.no_grad()
    def logp(self, lam):
        """log p(lgM, z | Lambda) at the grid points (lam: array of the 6 hyper-parameters)"""
        ls = self.scaler.forward(torch.tensor(np.asarray(lam, np.float32)[None])).to(self.dev)
        x = self.pts; u = self.xt.forward(x); t = ((x - self.LO) / (self.HI - self.LO)).clamp(self.EPS, 1 - self.EPS)
        return (self.flow.log_prob(u, ls.expand(len(x), -1)) + (-torch.log(self.STD) - torch.log(t * (1 - t)) - torch.log(self.HI - self.LO)).sum(1)).double()

    @torch.no_grad()
    def logp_batch(self, lams):
        """log p on the grid for B hyper-parameter vectors at once -> (B, n_grid)"""
        lams = np.asarray(lams, np.float32).reshape(-1, 6); B, n = len(lams), len(self.pts)
        ls = self.scaler.forward(torch.tensor(lams)).to(self.dev).repeat_interleave(n, 0); x = self.pts.repeat(B, 1)
        u = self.xt.forward(x); t = ((x - self.LO) / (self.HI - self.LO)).clamp(self.EPS, 1 - self.EPS)
        return (self.flow.log_prob(u, ls) + (-torch.log(self.STD) - torch.log(t * (1 - t)) - torch.log(self.HI - self.LO)).sum(1)).double().reshape(B, n)

    @torch.no_grad()
    def weights(self, lam):
        """w_g = p(g|Lambda) P(g) W_g on the grid; alpha = sum_g w_g"""
        return torch.exp(self.logp(lam)) * self.WP


class ErrorFlow:
    """Fisher-error flow of one survey: event parameters -> sampled 7x7 posterior covariance (order z, lgM6, eta, tvisc, bb, rph0, lphoto)."""
    def __init__(self, survey, device="cpu", name="tde_fisher_flow_big"):
        self.dev = torch.device(device)
        ck = torch.load(f"{DATA}/models/detectability/{name}_{survey}/best.pt", map_location=self.dev, weights_only=False); c = ck["cfg"]
        self.flow = ConditionalFlow(c["d"], c["ctx_dim"], n_layers=c["n_layers"], hidden=c["hidden"], K=c["K"], tail=c["tail"], ctx_hidden=c["ctx_hidden"]).to(self.dev)
        self.flow.load_state_dict(ck["model"]); self.flow.eval()
        self.cm, self.cs, self.ym, self.ys = ck["ctx_mean"], ck["ctx_std"], ck["y_mean"], ck["y_std"]; self.t0min, self.trange = ck["t0min"], ck["trange"]

    @staticmethod
    def decode(y):
        sig = np.exp(y[:, :NP]); L = np.zeros((len(y), NP, NP)); L[:, TRI[0], TRI[1]] = y[:, NP:]
        for i in range(NP):
            off = np.linalg.norm(L[:, i, :i], axis=1); s = np.where(off > 0.999, 0.999 / np.maximum(off, 1e-12), 1.0); L[:, i, :i] *= s[:, None]
            L[:, i, i] = np.sqrt(1 - np.linalg.norm(L[:, i, :i], axis=1) ** 2)
        R = L @ L.transpose(0, 2, 1); return R * sig[:, :, None] * sig[:, None, :]

    def context(self, d):
        return np.stack([d["lgM"], d["z"], np.log10(d["eta"]), np.log10(d["tvisc"]), d["bb"], np.log10(d["rph0"]), d["lphoto"], np.log10(d["alpha"]), d["beta"],
                         np.sin(d["ra"]), np.cos(d["ra"]), d["dec"], (d["t0"] - self.t0min) / self.trange], 1)

    @torch.no_grad()
    def sample_cov(self, d, gen=None, chunk=20000):
        c = self.context(d); out = []
        for s in range(0, len(c), chunk):
            x = torch.tensor((c[s:s + chunk] - self.cm) / self.cs, dtype=torch.float32, device=self.dev)
            y = self.flow.sample(x).cpu().numpy().astype(np.float64) * self.ys + self.ym; out.append(self.decode(y))
        return np.concatenate(out)


class TDELikelihood:
    """ln L(Lambda) for a catalogue obs (n,2)=(lgM_hat, z_hat), cov (n,2,2) in the same order.  Kernels precomputed once (float32 on the device)."""
    def __init__(self, pop, obs, cov, min_sigma_cells=0.5):
        self.pop = pop; dev = pop.dev; n = len(obs)
        dl = np.diff(np.unique(pop.lgM))[0]; zr = np.exp(np.diff(np.log(np.unique(pop.z)))[0]) - 1.0     # grid spacing in lgM and relative in z
        cov = cov.copy()
        # kernels thinner than min_sigma_cells grid cells (in ANY direction, including a near-degenerate correlated ridge) would alias on the grid: floor the eigenvalues of the
        # covariance in units of the local cell size (dl in lgM, zr*z in z)
        sc = np.stack([np.full(n, dl), zr * np.maximum(obs[:, 1], 1e-3)], 1); Cs = cov / (sc[:, :, None] * sc[:, None, :]); w, V = np.linalg.eigh(Cs)
        self.n_floored = int((w[:, 0] < min_sigma_cells ** 2).sum()); w = np.maximum(w, min_sigma_cells ** 2)
        cov = np.einsum("nij,nj,nkj->nik", V, w, V) * (sc[:, :, None] * sc[:, None, :])
        # kernels built on the device in chunks of events, stored once as float32 (n_events x n_grid; ~5.5 GB for 18,000 events on the 76,800-point grid)
        inv = np.linalg.inv(cov); det = np.linalg.det(cov); ng = len(pop.lgM)
        gl = torch.tensor(pop.lgM, dtype=torch.float32, device=dev); gz = torch.tensor(pop.z, dtype=torch.float32, device=dev); W = torch.tensor(pop.W, dtype=torch.float32, device=dev)
        t32 = lambda v: torch.tensor(v, dtype=torch.float32, device=dev)[:, None]
        self.K = torch.empty((n, ng), dtype=torch.float32, device=dev); self.mass_in_grid = np.empty(n)
        for s in range(0, n, 500):
            e = slice(s, min(s + 500, n)); dx = gl[None] - t32(obs[e, 0]); dz = gz[None] - t32(obs[e, 1])
            q = t32(inv[e, 0, 0]) * dx * dx + 2 * t32(inv[e, 0, 1]) * dx * dz + t32(inv[e, 1, 1]) * dz * dz
            self.K[e] = torch.exp(-0.5 * q) / (2 * np.pi * t32(np.sqrt(det[e])))
            self.mass_in_grid[e] = (self.K[e] @ W).cpu().numpy()          # fraction of each kernel inside the grid (diagnostic; <1 near the edges)
        self.n = n

    @torch.no_grad()
    def loglike_batch(self, lams, block=2):
        out = []
        for s in range(0, len(lams), block):
            w = torch.exp(self.pop.logp_batch(lams[s:s + block])) * self.pop.WP[None]; alpha = w.sum(1)
            num = (self.K @ w.float().T).double()                                   # (n_events, B)
            out.append((torch.log(num.clamp_min(1e-300)) - torch.log(alpha)[None]).sum(0))
        return torch.cat(out).cpu().numpy()

    @torch.no_grad()
    def loglike(self, lam):
        w = self.pop.weights(lam); alpha = w.sum()
        num = (self.K @ w.float()).double()
        return (torch.log(num.clamp_min(1e-300)) - torch.log(alpha)).sum().item(), alpha.item()
