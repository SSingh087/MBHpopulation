"""common.py -- paths, variable definitions, and the data/context transforms used by the flow scripts."""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from priors import LAMBDA_NAMES, FIDUCIAL, PRIOR_BOX            # noqa: E402  (light module)

DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"
FULL_RUN = f"{DATA}/simulations/full_run"
CONSOL = f"{DATA}/simulations/consolidated"

# x per source population (columns of the event arrays) and the physical bounds of each variable
POPS = {
    "emri": dict(names=["lgMBH", "spin", "z", "e_plunge"],
                 bounds=[(4.0, 7.0), (0.0, 1.0), (0.0, 8.0), (0.0, 1.0)],
                 labels=[r"$\log_{10}M_{\rm BH}$", "spin $a$", "$z$", r"$e_{\rm plunge}$"]),
    "tde": dict(names=["lgMBH", "z"], bounds=[(4.0, 7.0), (0.0, 8.0)],
                labels=[r"$\log_{10}M_{\rm BH}$", "$z$"]),
}
LAM_LABELS = ["$A$", "$B$", "scatter", r"$\log M_0$", r"$\delta$", r"$\theta$"]


class XTransform:
    """bounded x -> logit -> standardised u (unbounded, ~unit scale).  Fitted on training events only."""
    EPS = 1e-4

    def __init__(self, bounds, mean=None, std=None):
        self.lo = torch.tensor([b[0] for b in bounds], dtype=torch.float32)
        self.hi = torch.tensor([b[1] for b in bounds], dtype=torch.float32)
        self.mean, self.std = mean, std

    def _logit(self, x):
        u = ((x - self.lo.to(x.device)) / (self.hi - self.lo).to(x.device)).clamp(self.EPS, 1 - self.EPS)
        return torch.log(u / (1 - u))

    def fit(self, x):
        l = self._logit(x); self.mean, self.std = l.mean(0), l.std(0); return self

    def forward(self, x):
        return (self._logit(x) - self.mean.to(x.device)) / self.std.to(x.device)

    def inverse(self, u):
        l = u * self.std.to(u.device) + self.mean.to(u.device)
        lo, hi = self.lo.to(u.device), self.hi.to(u.device)
        return lo + (hi - lo) * torch.sigmoid(l)

    def state(self):
        return dict(mean=self.mean.cpu(), std=self.std.cpu())


class LamScaler:
    """Lambda (n,6) natural units -> [-1,1]^6 over the prior box (delta on a log axis)."""
    def __init__(self):
        self.lo = torch.tensor([PRIOR_BOX[k][0] for k in LAMBDA_NAMES]); self.hi = torch.tensor([PRIOR_BOX[k][1] for k in LAMBDA_NAMES])
        self.is_log = torch.tensor([k == "delta" for k in LAMBDA_NAMES])

    def forward(self, lam):
        lam = lam.float(); lo, hi, lg = self.lo.to(lam.device), self.hi.to(lam.device), self.is_log.to(lam.device)
        v = torch.where(lg, torch.log(lam), lam); l = torch.where(lg, torch.log(lo), lo); h = torch.where(lg, torch.log(hi), hi)
        return 2 * (v - l) / (h - l) - 1


def lam_dict_to_array(d):
    return np.array([d[k] for k in LAMBDA_NAMES], dtype=np.float32)


def load_consolidated(pop):
    meta = np.load(f"{CONSOL}/meta.npz"); ev = np.load(f"{CONSOL}/{pop}.npz")
    return {k: meta[k] for k in meta.files}, ev["x"], ev["uid"]
