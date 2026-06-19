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
torch.set_num_threads(1)          # each worker uses 1 thread — parallelism comes from the pool, not from within each call
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
parser.add_argument('--GALAXIES',       type=int, required=True)
parser.add_argument('--SOURCE',         type=str, required=True,
                    choices=['EMRI', 'TDE', 'EMRI_TDE'])
parser.add_argument('--SURVEY',         type=str, default='None',
                    choices=['ZTF', 'LSST', 'ALL', 'None'])

# MBH mass (FixedLimitSchechterFunction)
parser.add_argument('--xc',             type=float, nargs=2)
parser.add_argument('--lam_schechter',  type=float, nargs=2)

# MBH spin (DoubleBetaDistribution)
parser.add_argument('--alpha_1_MBHSpin', type=float, nargs=2)
parser.add_argument('--alpha_2_MBHSpin', type=float, nargs=2)
parser.add_argument('--beta_1_MBHSpin',  type=float, nargs=2)
parser.add_argument('--beta_2_MBHSpin',  type=float, nargs=2)
parser.add_argument('--weight_MBHSpin',  type=float, nargs=2)

# EMRI eccentricity (DoubleGaussianDistribution)
parser.add_argument('--alpha_1_e_pl',       type=float, nargs=2)
parser.add_argument('--alpha_2_e_pl',       type=float, nargs=2)
parser.add_argument('--beta_1_e_pl',        type=float, nargs=2)
parser.add_argument('--beta_2_e_pl',        type=float, nargs=2)
parser.add_argument('--weight_e_pl',       type=float, nargs=2)

# CO mass (DoubleGaussianDistribution)
parser.add_argument('--mu_sBH',         type=float, nargs=2)
parser.add_argument('--sigma_sBH',      type=float, nargs=2)
parser.add_argument('--mu_star',        type=float, nargs=2)
parser.add_argument('--sigma_star',     type=float, nargs=2)
parser.add_argument('--weight_COmass',  type=float, nargs=2)

# z_gal (BetaDistribution)
parser.add_argument('--lam_z_gal',      type=float, nargs=2)

args = parser.parse_args()

# ── Unpack bounds ─────────────────────────────────────────────────────────
xc_min,             xc_max             = args.xc
lam_schechter_min,  lam_schechter_max  = args.lam_schechter

alpha_1_spin_min,   alpha_1_spin_max   = args.alpha_1_MBHSpin
alpha_2_spin_min,   alpha_2_spin_max   = args.alpha_2_MBHSpin
beta_1_spin_min,    beta_1_spin_max    = args.beta_1_MBHSpin
beta_2_spin_min,    beta_2_spin_max    = args.beta_2_MBHSpin
weight_spin_min,    weight_spin_max    = args.weight_MBHSpin

alpha_1_e_pl_min,    alpha_1_e_pl_max    = args.alpha_1_e_pl
alpha_2_e_pl_min,    alpha_2_e_pl_max    = args.alpha_2_e_pl
beta_1_e_pl_min,     beta_1_e_pl_max     = args.beta_1_e_pl
beta_2_e_pl_min,     beta_2_e_pl_max     = args.beta_2_e_pl
weight_e_pl_min,    weight_e_pl_max    = args.weight_e_pl

mu_sBH_min,         mu_sBH_max         = args.mu_sBH
sigma_sBH_min,      sigma_sBH_max      = args.sigma_sBH
mu_star_min,        mu_star_max        = args.mu_star
sigma_star_min,     sigma_star_max     = args.sigma_star
weight_CO_min,      weight_CO_max      = args.weight_COmass

lam_z_gal_min,      lam_z_gal_max      = args.lam_z_gal

# ── Population setup ──────────────────────────────────────────────────────
true_x  = get_true_x("A")
popdist = make_dist("A")

# ── Load data ─────────────────────────────────────────────────────────────
data_EMRI, data_TDE = {}, {}
# base = f'./data/inference_data/{args.GALAXIES}'
base = f'./data/pseudo_inference_data/{args.GALAXIES}'

def load_torch(path):
    return torch.load(path, map_location='cpu')

# This will be needed when we  use the real simulation. 
# For now, we can directly load the pseudo data.
# emri_torch = load_torch(f'{base}/true_data_EMRI_no_clustering_torch.pt')

# if args.SURVEY in ['ZTF', 'LSST']:
#     tde_torch = load_torch(f'{base}/true_data_TDE_{args.SURVEY}_no_clustering_torch.pt')
#     data_TDE  = tde_torch
# elif args.SURVEY == 'ALL':
#     ztf  = load_torch(f'{base}/true_data_TDE_ZTF_no_clustering_torch.pt')
#     lsst = load_torch(f'{base}/true_data_TDE_LSST_no_clustering_torch.pt')
#     data_TDE = {k: torch.cat([ztf[k], lsst[k]], dim=0) for k in ztf}


emri_torch = load_torch(f'{base}/true_data_EMRI_no_clustering_pseudo_data_A.pt')

if args.SURVEY in ['ZTF', 'LSST']:
    tde_torch = load_torch(f'{base}/true_data_TDE_{args.SURVEY}_no_clustering_pseudo_data_A.pt')
    data_TDE  = tde_torch
elif args.SURVEY == 'ALL':
    ztf  = load_torch(f'{base}/true_data_TDE_ZTF_no_clustering_pseudo_data_A.pt')
    lsst = load_torch(f'{base}/true_data_TDE_LSST_no_clustering_pseudo_data_A.pt')
    data_TDE = {k: torch.cat([ztf[k], lsst[k]], dim=0) for k in ztf}

data_EMRI = emri_torch

def get_data(source):
    if source == 'TDE':
        return {'TDE': data_TDE}, 0, 1
    elif source == 'EMRI_TDE':
        return {'EMRI': data_EMRI, 'TDE': data_TDE}, 1, 1
    else:  # EMRI
        return {'EMRI': data_EMRI}, 1, 0

popdist.data, has_EMRI, has_TDE = get_data(args.SOURCE)

output = f"{base}/{args.SOURCE}_pop_A"
logger = setup_logger(output=output)

# print(f"[DEBUG] {args.SOURCE} keys: {popdist.data[args.SOURCE].keys()}")

# ── Model ─────────────────────────────────────────────────────────────────
class PopulationModel(Model):

    def __init__(self, popdist):
        super().__init__()
        self.popdist = popdist

        # Distributions (shared between EMRI and TDE)
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
        # shared by both EMRI and TDE
        shared = ["xc", "lam_schechter",
                  "mu_sBH", "sigma_sBH", "mu_star", "sigma_star", "weight_COmass",
                  "lam_z_gal"]
        
        # EMRI-only (spin & eccentricity)
        emri   = ["alpha_1_MBHSpin", "alpha_2_MBHSpin", "beta_1_MBHSpin",  "beta_2_MBHSpin", "weight_MBHSpin",
                  "alpha_1_e_pl", "alpha_2_e_pl", "beta_1_e_pl",  "beta_2_e_pl", "weight_e_pl"]
        
        return shared + (emri if has_EMRI else [])

    def _bounds(self):
        b = {
            'xc'             : [xc_min,            xc_max           ],
            'lam_schechter'  : [lam_schechter_min,  lam_schechter_max],
            'mu_sBH'         : [mu_sBH_min,         mu_sBH_max       ],
            'sigma_sBH'      : [sigma_sBH_min,      sigma_sBH_max    ],
            'mu_star'        : [mu_star_min,         mu_star_max      ],
            'sigma_star'     : [sigma_star_min,      sigma_star_max   ],
            'weight_COmass'  : [weight_CO_min,       weight_CO_max    ],
            'lam_z_gal'      : [lam_z_gal_min,      lam_z_gal_max     ],
        }
        if has_EMRI:
            b.update({
                'alpha_1_MBHSpin': [alpha_1_spin_min, alpha_1_spin_max],
                'alpha_2_MBHSpin': [alpha_2_spin_min, alpha_2_spin_max],
                'beta_1_MBHSpin' : [beta_1_spin_min,  beta_1_spin_max ],
                'beta_2_MBHSpin' : [beta_2_spin_min,  beta_2_spin_max ],
                'weight_MBHSpin' : [weight_spin_min,  weight_spin_max ],
                'alpha_1_e_pl': [alpha_1_e_pl_min, alpha_1_e_pl_max],
                'alpha_2_e_pl': [alpha_2_e_pl_min, alpha_2_e_pl_max],
                'beta_1_e_pl' : [beta_1_e_pl_min,  beta_1_e_pl_max ],
                'beta_2_e_pl' : [beta_2_e_pl_min,  beta_2_e_pl_max ],
                'weight_e_pl' : [weight_e_pl_min,  weight_e_pl_max ],
            })
        return b

    def log_prior(self, x):
        log_p = np.log(self.in_bounds(x), dtype="float")
        for n in self.names:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p

    def EMRI_likelihood(self, x):
        if not has_EMRI:
            return 0.0
        pdf_mass = self.dist_lgMBH.pdf(
            self.emri_data['lgMBH_mass'],
            xc=x["xc"], lam_schechter=x["lam_schechter"]
        )
        pdf_spin = self.dist_spin.pdf(
            self.emri_data['MBHspin'],
            alpha_1=x["alpha_1_MBHSpin"], alpha_2=x["alpha_2_MBHSpin"],
            beta_1=x["beta_1_MBHSpin"],   beta_2=x["beta_2_MBHSpin"], weight=x["weight_MBHSpin"]
        )
        pdf_CO = self.dist_CO.pdf(
            self.emri_data['CO_mass'],
            mu_1=x["mu_star"], sigma_1=x["sigma_star"],
            mu_2=x["mu_sBH"],  sigma_2=x["sigma_sBH"], weight=x["weight_COmass"]
        )
        pdf_e_pl = self.dist_e_pl.pdf(
            self.emri_data['e_pl'],
            alpha_1=x["alpha_1_e_pl"], alpha_2=x["alpha_2_e_pl"],
            beta_1=x["beta_1_e_pl"],   beta_2=x["beta_2_e_pl"], weight=x["weight_e_pl"]
        )
        pdf_z_gal = self.dist_z_gal.pdf(
            self.emri_data['z_gal'],
            lam=x["lam_z_gal"]
        )
        logL = torch.sum(torch.log(
            torch.nanmean(pdf_mass * pdf_spin * pdf_CO * pdf_e_pl * pdf_z_gal, dim=1) + 1e-10
        ))
        return logL.item()

    def TDE_likelihood(self, x):
        if not has_TDE:
            return 0.0
        pdf_mass = self.dist_lgMBH.pdf(
            self.tde_data['lgMBH_mass'],
            xc=x["xc"], lam_schechter=x["lam_schechter"]
        )
        pdf_CO = self.dist_CO.pdf(
            self.tde_data['CO_mass'],
            mu_1=x["mu_star"], sigma_1=x["sigma_star"],
            mu_2=x["mu_sBH"],  sigma_2=x["sigma_sBH"], weight=x["weight_COmass"]
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
    output=output, resume=False, seed=123123, nlive=1000, n_pool=4
)
fs.run()

# ── Corner plot ───────────────────────────────────────────────────────────
with h5py.File(f'{output}/result.hdf5', 'r') as hf:
    post = get_posterior_samples("A", hf)

posterior_samples = np.vstack(list(post.values())).T
truths = [v for params in get_true_x("A").values() for v in params.values()]
labels = list(get_latex_labels("A").values())

print("Truths:", truths)
print("Labels:", labels)

figure = corner.corner(
    posterior_samples, labels=labels,
    title_kwargs={"fontsize": 18}, show_titles=True,
    title_fmt=".3f", quantiles=[0.16, 0.5, 0.84],
    label_kwargs={"fontsize": 18},
    color='blue', 
    # truth_color='red', truths=truths,
)
plt.savefig(f'{output}/hyperposterior.png', dpi=200)