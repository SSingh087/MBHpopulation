#!/usr/bin/env python


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

# CO mass (DoubleGaussianDistribution)
parser.add_argument('--mu_sBH',          type=float, nargs=2)
parser.add_argument('--sigma_sBH',       type=float, nargs=2)
parser.add_argument('--mu_star',         type=float, nargs=2)
parser.add_argument('--sigma_star',      type=float, nargs=2)
parser.add_argument('--weight_COmass',   type=float, nargs=2)

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

mu_sBH_min,        mu_sBH_max        = args.mu_sBH
sigma_sBH_min,     sigma_sBH_max     = args.sigma_sBH
mu_star_min,       mu_star_max       = args.mu_star
sigma_star_min,    sigma_star_max    = args.sigma_star
weight_CO_min,     weight_CO_max     = args.weight_COmass

lam_z_gal_min,     lam_z_gal_max     = args.lam_z_gal

# ── Population setup ──────────────────────────────────────────────────────
true_x  = get_true_x("B", args.mbh_mass_method, args.nucleation_method)
popdist = make_dist("B")

# ── Load data ─────────────────────────────────────────────────────────────
data_EMRI, data_TDE = {}, {}
base = f'./data/pseudo_inference_data/{args.GALAXIES}'

def load_torch(path):
    return torch.load(path, map_location='cpu')

emri_torch = load_torch(f'{base}/true_data_EMRI_pseudo_data_B_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}.pt')

if args.SURVEY in ['ZTF', 'LSST']:
    tde_torch = load_torch(f'{base}/true_data_TDE_{args.SURVEY}_pseudo_data_B_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}.pt')
    data_TDE  = tde_torch
elif args.SURVEY == 'ALL':
    ztf  = load_torch(f'{base}/true_data_TDE_ZTF_pseudo_data_B_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}.pt')
    lsst = load_torch(f'{base}/true_data_TDE_LSST_pseudo_data_B_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}.pt')
    data_TDE = {k: torch.cat([ztf[k], lsst[k]], dim=0) for k in ztf}

data_EMRI = emri_torch

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

    def __init__(self, popdist):
        super().__init__()
        self.popdist = popdist

        # Distributions (shared — TDE uses lgMBH + CO only, no spin)
        self.dist_lgMBH = popdist.distributions['lgMBH_mass']
        self.dist_spin  = popdist.distributions['MBHspin']
        self.dist_CO    = popdist.distributions['CO_mass']
        self.dist_e_pl  = popdist.distributions['e_pl']
        self.dist_z_gal = popdist.distributions['z_gal']

        if has_EMRI:
            self.emri_data = popdist.data['EMRI']
        if has_TDE:
            self.tde_data  = popdist.data['TDE']

        self.names  = self._param_names()
        self.bounds = self._bounds()

    def _param_names(self):
        # lgMBH + CO params shared by both EMRI and TDE
        shared = ["lambda_M",
                  "mu_sBH", "sigma_sBH", "mu_star", "sigma_star", "weight_COmass",
                  "lam_z_gal"]
        # spin only used by EMRI
        emri   = ["mu_1_MBHSpin", "sigma_1_MBHSpin", "mu_2_MBHSpin", "sigma_2_MBHSpin", "weight_MBHSpin",
                  "mu_1_e_pl", "sigma_1_e_pl", "mu_2_e_pl", "sigma_2_e_pl", "weight_e_pl"]

        return shared + (emri if has_EMRI else [])

    def _bounds(self):
        b = {
            'lambda_M'       : [lambda_M_min,     lambda_M_max    ],
            'mu_sBH'         : [mu_sBH_min,        mu_sBH_max      ],
            'sigma_sBH'      : [sigma_sBH_min,     sigma_sBH_max   ],
            'mu_star'        : [mu_star_min,        mu_star_max     ],
            'sigma_star'     : [sigma_star_min,     sigma_star_max  ],
            'weight_COmass'  : [weight_CO_min,      weight_CO_max   ],
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
        log_p = np.log(self.in_bounds(x), dtype="float")
        for n in self.names:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p

    def _co_params(self, x):
        """DoubleGaussianDistribution kwargs for CO mass."""
        return {
            "mu_1"   : x["mu_star"],       # stellar component (~1 Msun)
            "sigma_1": x["sigma_star"],
            "mu_2"   : x["mu_sBH"],        # sBH component (~10 Msun)
            "sigma_2": x["sigma_sBH"],
            "weight" : x["weight_COmass"],
        }

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
        pdf_CO = self.dist_CO.pdf(
            self.emri_data['CO_mass'],
            mu_1=x["mu_star"], sigma_1=x["sigma_star"],
            mu_2=x["mu_sBH"],  sigma_2=x["sigma_sBH"],
            weight=x["weight_COmass"]
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
        logL = torch.sum(torch.log(torch.nanmean(pdf_mass * pdf_spin * pdf_CO * pdf_e_pl * pdf_z_gal, dim=1) + 1e-10))
        return logL.item()

    def TDE_likelihood(self, x):
        if not has_TDE:
            return 0.0
        pdf_mass = self.dist_lgMBH.pdf(
            self.tde_data['lgMBH_mass'],
            lam=x["lambda_M"]
        )
        pdf_CO = self.dist_CO.pdf(
            self.tde_data['CO_mass'],
            mu_1=x["mu_star"], sigma_1=x["sigma_star"],
            mu_2=x["mu_sBH"],  sigma_2=x["sigma_sBH"],
            weight=x["weight_COmass"]
        )
        pdf_z_gal = self.dist_z_gal.pdf(
            self.tde_data['z_gal'],
            lam=x["lam_z_gal"]
        )
        logL = torch.sum(torch.log(torch.nanmean(pdf_mass * pdf_CO * pdf_z_gal, dim=1) + 1e-10))
        return logL.item()

    @torch.inference_mode()
    def log_likelihood(self, x):
        if isinstance(x, (np.void, np.ndarray)):
            x = {n: np.ascontiguousarray(x[n], dtype=np.float32) for n in x.dtype.names}
        x = {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
             for k, v in x.items()}
        return self.EMRI_likelihood(x) + self.TDE_likelihood(x)


# ── Run sampler ───────────────────────────────────────────────────────────
fs = FlowSampler(
    PopulationModel(popdist=popdist),
    output=output, resume=False, seed=123123, nlive=1000, n_pool=12
)
fs.run()

# ── Corner plot ───────────────────────────────────────────────────────────
with h5py.File(f'{output}/result.hdf5', 'r') as hf:
    post = load_posterior_samples_dynamic(hf)

# FIX: this used to hardcode "A" here -- a pop-B run was plotting against
# pop-A's parameter names and truth values.
labels_dict = get_latex_labels("B")
truths_dict = get_true_x_flat("B", args.mbh_mass_method, args.nucleation_method)

keys = [k for k in labels_dict if k in post]   # only params actually sampled this run

posterior_samples = np.vstack([post[k] for k in keys]).T
truths = [truths_dict[k] for k in keys]
labels = [labels_dict[k] for k in keys]

print("Truths:", truths)
print("Labels:", labels)

figure = corner.corner(
    posterior_samples, labels=labels,
    title_kwargs={"fontsize": 18}, show_titles=True,
    title_fmt=".3f", quantiles=[0.16, 0.5, 0.84],
    label_kwargs={"fontsize": 18},
    color='blue',
)

plt.savefig(f'{output}/hyperposterior.png', dpi=200)