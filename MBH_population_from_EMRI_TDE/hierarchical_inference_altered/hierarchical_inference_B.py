#!/usr/bin/env python
"""Hierarchical Bayesian inference of the MBH population (population B: power-law mass function, double-Gaussian spin/eccentricity) from EMRI and/or TDE catalogues."""


import os
os.environ["OMP_NUM_THREADS"]     = "1"
os.environ["MKL_NUM_THREADS"]     = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
from utils import *
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger
import h5py
import corner
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cpu")
print(f"Using device: {device}")

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--GALAXIES',        type=int, required=True)
parser.add_argument('--SOURCE',          type=str, required=True,
                    choices=['EMRI', 'TDE', 'EMRI_TDE'])
parser.add_argument('--SURVEY',          type=str, default='None',
parser.add_argument('--TDE_MODEL', type=str, default='tde_fallback', help='TDE light-curve model of the selection function')
                    choices=['ZTF', 'LSST', 'ALL', 'None'])
parser.add_argument("--mbh_mass_method", type=str, required=True, default="use_sigma", choices=["use_sigma", "use_lgMgal"])
parser.add_argument("--nucleation_method", type=str, required=True, default="hannah", choices=["hannah", "neumayer"])
parser.add_argument("--occupation_method", type=str, required=True, default="uniform", choices=["uniform", "varying"])

# MBH mass (FixedLimitsPowerLaw)
parser.add_argument('--lambda_M',        type=float, nargs=2)

# MBH spin (DoubleGaussianDistribution)
parser.add_argument('--mu_1_MBHSpin',    type=float, nargs=2)
parser.add_argument('--sigma_1_MBHSpin', type=float, nargs=2)
parser.add_argument('--mu_2_MBHSpin',    type=float, nargs=2)
parser.add_argument('--sigma_2_MBHSpin', type=float, nargs=2)
parser.add_argument('--weight_MBHSpin',  type=float, nargs=2)

# EMRI eccentricity (DoubleGaussianDistribution)
parser.add_argument('--mu_1_e_pl',       type=float, nargs=2)
parser.add_argument('--sigma_1_e_pl',       type=float, nargs=2)
parser.add_argument('--mu_2_e_pl',        type=float, nargs=2)
parser.add_argument('--sigma_2_e_pl',        type=float, nargs=2)
parser.add_argument('--weight_e_pl',       type=float, nargs=2)

# z_gal (FixedLimitsPowerLaw)
parser.add_argument('--lam_z_gal',      type=float, nargs=2)

args = parser.parse_args()

# ── Unpack bounds ─────────────────────────────────────────────────────────
lambda_M_min,      lambda_M_max      = args.lambda_M

mu_1_spin_min,     mu_1_spin_max     = args.mu_1_MBHSpin
sigma_1_spin_min,  sigma_1_spin_max  = args.sigma_1_MBHSpin
mu_2_spin_min,     mu_2_spin_max     = args.mu_2_MBHSpin
sigma_2_spin_min,  sigma_2_spin_max  = args.sigma_2_MBHSpin
weight_spin_min,   weight_spin_max   = args.weight_MBHSpin

mu_1_e_min,        mu_1_e_max        = args.mu_1_e_pl
sigma_1_e_min,     sigma_1_e_max     = args.sigma_1_e_pl
mu_2_e_min,        mu_2_e_max        = args.mu_2_e_pl
sigma_2_e_min,     sigma_2_e_max     = args.sigma_2_e_pl
weight_e_min,      weight_e_max      = args.weight_e_pl

lam_z_gal_min,     lam_z_gal_max     = args.lam_z_gal

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'selection_function'))
from selection_function import SelectionFunction

# ── Population setup ──────────────────────────────────────────────────────
popdist = make_dist("B")

# ── Load data ─────────────────────────────────────────────────────────────
data_EMRI, data_TDE = {}, {}
base = f'./data/inference_data/{args.GALAXIES}'

def load_torch(path):
    """Load a torch tensor file."""
    return torch.load(path, map_location='cpu')

emri_torch = load_torch(f'{base}/fisher_posteriors_EMRI_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}.pt')

if args.SURVEY in ['ZTF', 'LSST']:
    tde_torch = load_torch(f'{base}/fisher_posteriors_TDE_{args.SURVEY}_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}.pt')
    data_TDE  = tde_torch
elif args.SURVEY == 'ALL':
    ztf  = load_torch(f'{base}/fisher_posteriors_TDE_ZTF_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}.pt')
    lsst = load_torch(f'{base}/fisher_posteriors_TDE_LSST_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}.pt')
    data_TDE = {k: torch.cat([ztf[k], lsst[k]], dim=0) for k in ztf}

data_EMRI = emri_torch

# Events per survey (before padding): each TDE is divided by the selection function of its own survey
if args.SURVEY == 'ALL':
    tde_counts = {'ZTF': len(ztf['z_gal']), 'LSST': len(lsst['z_gal'])}
elif args.SURVEY in ['ZTF', 'LSST']:
    tde_counts = {args.SURVEY: len(tde_torch['z_gal'])}
else:
    tde_counts = {}

def ensure_min_events(data, min_events=2, label=""):
    """If a source has fewer than min_events rows, pad by repeating existing
    rows until there are at least min_events. Also restores the event
    dimension if it got squeezed to 1D (the case your manual check caught).
    NOTE: repeated rows are NOT independent new information -- each
    duplicate's log-likelihood term gets summed as if it were a separate
    real event, which artificially sharpens the posterior. This avoids a
    crash, it does not manufacture real statistical power."""
    if data is None or len(data) == 0:
        return data

    for k in list(data.keys()):
        if data[k].dim() == 1:
            data[k] = data[k][None, :]

    any_key = next(iter(data))
    n_events = data[any_key].shape[0]

    if n_events == 0:
        print(f"[WARN] {label}: 0 events -- cannot pad, leaving empty.")
        return data

    if n_events < min_events:
        reps = -(-min_events // n_events)  # ceil division
        print(f"[WARN] {label}: only {n_events} event(s) -- duplicating to reach "
              f"{min_events} (repeat {reps}x, trim to exactly {min_events}). "
              f"Treat this run's posterior as diagnostic, not a real result -- "
              f"see ensure_min_events() docstring.")
        for k in list(data.keys()):
            reps_shape = [1] * data[k].dim()
            reps_shape[0] = reps
            data[k] = data[k].repeat(*reps_shape)[:min_events]

    return data


def get_data(source):
    """Load the detected-event posteriors for the source."""
    if source == 'TDE':
        return {'TDE': data_TDE}, 0, 1
    elif source == 'EMRI_TDE':
        return {'EMRI': data_EMRI, 'TDE': data_TDE}, 1, 1
    else:  # EMRI
        return {'EMRI': data_EMRI}, 1, 0

popdist.data, has_EMRI, has_TDE = get_data(args.SOURCE)

if has_EMRI:
    popdist.data['EMRI'] = ensure_min_events(
        popdist.data['EMRI'], min_events=2,
        label=f"EMRI ({args.mbh_mass_method}/{args.nucleation_method}/{args.occupation_method})"
    )
if has_TDE:
    popdist.data['TDE'] = ensure_min_events(
        popdist.data['TDE'], min_events=2,
        label=f"TDE ({args.SURVEY}, {args.mbh_mass_method}/{args.nucleation_method}/{args.occupation_method})"
    )
    
output = f"{base}/{args.SOURCE}_pop_B_{args.SURVEY}_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}"
logger = setup_logger(output=output)

# ── Model ─────────────────────────────────────────────────────────────────
class PopulationModel(Model):

    """Population likelihood and prior for the hierarchical inference."""
    def __init__(self, popdist):
        super().__init__()
        self.popdist = popdist

        # Distributions (shared — TDE uses lgMBH only, no spin)
        self.dist_lgMBH = popdist.distributions['lgMBH_mass']
        self.dist_spin  = popdist.distributions['MBHspin']
        self.dist_e_pl  = popdist.distributions['e_pl']
        self.dist_z_gal = popdist.distributions['z_gal']

        if has_EMRI:
            self.emri_data = popdist.data['EMRI']
        if has_TDE:
            self.tde_data  = popdist.data['TDE']

        self.sel_emri = SelectionFunction.load("B", "EMRI") if has_EMRI else None
        self.sel_tde  = {s: SelectionFunction.load("B", f"TDE_{s}", args.TDE_MODEL) for s in tde_counts} if has_TDE else {}

        self.names  = self._param_names()
        self.bounds = self._bounds()

    def _param_names(self):
        # lgMBH params shared by both EMRI and TDE
        shared = ["lambda_M", "lam_z_gal"]
        # spin only used by EMRI
        emri   = ["mu_1_MBHSpin", "sigma_1_MBHSpin", "mu_2_MBHSpin", "sigma_2_MBHSpin", "weight_MBHSpin",
                  "mu_1_e_pl", "sigma_1_e_pl", "mu_2_e_pl", "sigma_2_e_pl", "weight_e_pl"]

        return shared + (emri if has_EMRI else [])

    def _bounds(self):
        b = {
            'lambda_M'       : [lambda_M_min,     lambda_M_max    ],
            'lam_z_gal'      : [lam_z_gal_min,      lam_z_gal_max   ],
        }
        if has_EMRI:
            b.update({
                'mu_1_MBHSpin'   : [mu_1_spin_min,    mu_1_spin_max   ],
                'sigma_1_MBHSpin': [sigma_1_spin_min, sigma_1_spin_max],
                'mu_2_MBHSpin'   : [mu_2_spin_min,    mu_2_spin_max   ],
                'sigma_2_MBHSpin': [sigma_2_spin_min, sigma_2_spin_max],
                'weight_MBHSpin' : [weight_spin_min,  weight_spin_max ],
                'mu_1_e_pl'      : [mu_1_e_min,       mu_1_e_max      ],
                'sigma_1_e_pl'   : [sigma_1_e_min,    sigma_1_e_max     ],
                'mu_2_e_pl'      : [mu_2_e_min,       mu_2_e_max      ],
                'sigma_2_e_pl'   : [sigma_2_e_min,    sigma_2_e_max     ],
                'weight_e_pl'     : [weight_e_min,     weight_e_max    ],
            })
        return b

    def log_prior(self, x):
        """Log prior of the hyperparameters."""
        log_p = np.log(self.in_bounds(x), dtype="float")
        for n in self.names:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p

    def _spin_params(self, x):
        """DoubleGaussianDistribution kwargs for MBH spin."""
        return {
            "mu_1"   : x["mu_1_MBHSpin"],
            "sigma_1": x["sigma_1_MBHSpin"],
            "mu_2"   : x["mu_2_MBHSpin"],
            "sigma_2": x["sigma_2_MBHSpin"],
            "weight" : x["weight_MBHSpin"],
        }

    def EMRI_likelihood(self, x):
        """EMRI log-likelihood including the selection effect."""
        if not has_EMRI:
            return 0.0
        pdf_mass = self.dist_lgMBH.pdf(
            self.emri_data['lgMBH_mass'],
            lam=x["lambda_M"]
        )
        pdf_spin = self.dist_spin.pdf(
            self.emri_data['MBHspin'],
            mu_1=x["mu_1_MBHSpin"], sigma_1=x["sigma_1_MBHSpin"],
            mu_2=x["mu_2_MBHSpin"], sigma_2=x["sigma_2_MBHSpin"],
            weight=x["weight_MBHSpin"]
        )
        pdf_e_pl = self.dist_e_pl.pdf(
            self.emri_data['e_pl'],
            mu_1=x["mu_1_e_pl"], sigma_1=x["sigma_1_e_pl"],
            mu_2=x["mu_2_e_pl"], sigma_2=x["sigma_2_e_pl"],
            weight=x["weight_e_pl"]
        )
        pdf_z_gal = self.dist_z_gal.pdf(
            self.emri_data['z_gal'],
            lam=x["lam_z_gal"]
        )
        per_event = torch.log(torch.nanmean(pdf_mass * pdf_spin * pdf_e_pl * pdf_z_gal, dim=1) + 1e-10)
        logL = torch.sum(per_event) - per_event.numel() * self.sel_emri.log_alpha(x)
        return logL.item()

    def TDE_likelihood(self, x):
        """TDE log-likelihood including the selection effect."""
        if not has_TDE:
            return 0.0
        pdf_mass = self.dist_lgMBH.pdf(
            self.tde_data['lgMBH_mass'],
            lam=x["lambda_M"]
        )
        pdf_z_gal = self.dist_z_gal.pdf(
            self.tde_data['z_gal'],
            lam=x["lam_z_gal"]
        )
        per_event = torch.log(torch.nanmean(pdf_mass * pdf_z_gal, dim=1) + 1e-10)
        # selection term: events split over the surveys in proportion to their original counts
        n_tot = sum(tde_counts.values())
        log_alpha = sum(n / n_tot * self.sel_tde[s].log_alpha(x) for s, n in tde_counts.items())
        logL = torch.sum(per_event) - per_event.numel() * log_alpha
        return logL.item()

    @torch.inference_mode()
    def log_likelihood(self, x):
        """Total log-likelihood of the enabled sources."""
        if isinstance(x, (np.void, np.ndarray)):
            x = {n: np.ascontiguousarray(x[n], dtype=np.float32) for n in x.dtype.names}
        x = {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
             for k, v in x.items()}
        return self.EMRI_likelihood(x) + self.TDE_likelihood(x)


# ── Run sampler ───────────────────────────────────────────────────────────
fs = FlowSampler(
    PopulationModel(popdist=popdist),
    output=output, resume=False, seed=123123, nlive=1000, n_pool=4
)
fs.run()

# ── Corner plot ───────────────────────────────────────────────────────────
with h5py.File(f'{output}/result.hdf5', 'r') as hf:
    post = load_posterior_samples_dynamic(hf)

# FIX: this used to hardcode "A" here -- a pop-B run was plotting against
labels_dict = get_latex_labels("B")

keys = [k for k in labels_dict if k in post]   # only params actually sampled this run

posterior_samples = np.vstack([post[k] for k in keys]).T
labels = [labels_dict[k] for k in keys]

print("Labels:", labels)

figure = corner.corner(
    posterior_samples, labels=labels,
    title_kwargs={"fontsize": 18}, show_titles=True,
    title_fmt=".3f", quantiles=[0.16, 0.5, 0.84],
    label_kwargs={"fontsize": 18},
    color='blue',
)

plt.savefig(f'{output}/hyperposterior.png', dpi=200)