#!/usr/bin/env python
"""Hierarchical Bayesian inference of the MBH population (population A: Schechter mass function, double-beta spin/eccentricity) from EMRI and/or TDE catalogues."""


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
parser.add_argument('--GALAXIES',       type=int, required=True)
parser.add_argument('--SOURCE',         type=str, required=True,
                    choices=['EMRI', 'TDE', 'EMRI_TDE'])
parser.add_argument('--SURVEY',         type=str, default='None',
                    choices=['ZTF', 'LSST', 'ALL', 'None'])

parser.add_argument("--mbh_mass_method", type=str, required=True, default="use_sigma", choices=["use_sigma", "use_lgMgal"])
parser.add_argument("--nucleation_method", type=str, required=True, default="hannah", choices=["hannah", "neumayer"])
parser.add_argument("--occupation_method", type=str, required=True, default="uniform", choices=["uniform", "varying"])

# MBH mass (FixedLimitSchechterFunction)
parser.add_argument('--xc',             type=float, nargs=2)
parser.add_argument('--lam_schechter',  type=float, nargs=2)

# MBH spin (DoubleBetaDistribution)
parser.add_argument('--alpha_1_MBHSpin', type=float, nargs=2)
parser.add_argument('--alpha_2_MBHSpin', type=float, nargs=2)
parser.add_argument('--beta_1_MBHSpin',  type=float, nargs=2)
parser.add_argument('--beta_2_MBHSpin',  type=float, nargs=2)
parser.add_argument('--weight_MBHSpin',  type=float, nargs=2)

# EMRI eccentricity (DoubleBetaDistribution)
parser.add_argument('--alpha_1_e_pl',       type=float, nargs=2)
parser.add_argument('--alpha_2_e_pl',       type=float, nargs=2)
parser.add_argument('--beta_1_e_pl',        type=float, nargs=2)
parser.add_argument('--beta_2_e_pl',        type=float, nargs=2)
parser.add_argument('--weight_e_pl',       type=float, nargs=2)

# z_gal (FixedLimitsPowerLaw)
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

lam_z_gal_min,      lam_z_gal_max      = args.lam_z_gal

# ── Population setup ──────────────────────────────────────────────────────
true_x  = get_true_x("A", args.mbh_mass_method, args.nucleation_method, args.occupation_method)
popdist = make_dist("A")

# ── Load data ─────────────────────────────────────────────────────────────
data_EMRI, data_TDE = {}, {}

base = f'./data/altered_inference_data/pseudo_inference_data/{args.GALAXIES}'

def load_torch(path):
    """Load a torch tensor file."""
    return torch.load(path, map_location='cpu')

emri_torch = load_torch(f'{base}/true_data_EMRI_pseudo_data_A_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}.pt')

if args.SURVEY in ['ZTF', 'LSST']:
    tde_torch = load_torch(f'{base}/true_data_TDE_{args.SURVEY}_pseudo_data_A_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}.pt')
    data_TDE  = tde_torch
elif args.SURVEY == 'ALL':
    ztf  = load_torch(f'{base}/true_data_TDE_ZTF_pseudo_data_A_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}.pt')
    lsst = load_torch(f'{base}/true_data_TDE_LSST_pseudo_data_A_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}.pt')
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

output = f"{base}/{args.SOURCE}_pop_A_{args.SURVEY}_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}"
logger = setup_logger(output=output)

# ── Model ─────────────────────────────────────────────────────────────────
class PopulationModel(Model):

    """Population likelihood and prior for the hierarchical inference."""
    def __init__(self, popdist):
        super().__init__()
        self.popdist = popdist

        # Distributions (shared between EMRI and TDE)
        self.dist_lgMBH = popdist.distributions['lgMBH_mass']
        self.dist_spin  = popdist.distributions['MBHspin']
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
        shared = ["xc", "lam_schechter", "lam_z_gal"]

        # EMRI-only (spin & eccentricity)
        emri   = ["alpha_1_MBHSpin", "alpha_2_MBHSpin", "beta_1_MBHSpin",  "beta_2_MBHSpin", "weight_MBHSpin",
                  "alpha_1_e_pl", "alpha_2_e_pl", "beta_1_e_pl",  "beta_2_e_pl", "weight_e_pl"]

        return shared + (emri if has_EMRI else [])

    def _bounds(self):
        b = {
            'xc'             : [xc_min,            xc_max           ],
            'lam_schechter'  : [lam_schechter_min,  lam_schechter_max],
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
        """Log prior of the hyperparameters."""
        log_p = np.log(self.in_bounds(x), dtype="float")
        for n in self.names:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p

    def EMRI_likelihood(self, x):
        """EMRI log-likelihood including the selection effect."""
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
        pdf_e_pl = self.dist_e_pl.pdf(
            self.emri_data['e_pl'],
            alpha_1=x["alpha_1_e_pl"], alpha_2=x["alpha_2_e_pl"],
            beta_1=x["beta_1_e_pl"],   beta_2=x["beta_2_e_pl"], weight=x["weight_e_pl"]
        )
        pdf_z_gal = self.dist_z_gal.pdf(
            self.emri_data['z_gal'],
            lam=x["lam_z_gal"]
        )

        alpha_EMRI = torch.tensor(1.0, dtype=torch.float32)
        logL = torch.sum(torch.log(
            torch.nanmean(pdf_mass * pdf_spin *  pdf_e_pl * pdf_z_gal, dim=1) - torch.log(alpha_EMRI) + 1e-10
        ))
        return logL.item()

    def TDE_likelihood(self, x):
        """TDE log-likelihood including the selection effect."""
        if not has_TDE:
            return 0.0
        pdf_mass = self.dist_lgMBH.pdf(
            self.tde_data['lgMBH_mass'],
            xc=x["xc"], lam_schechter=x["lam_schechter"]
        )
        pdf_z_gal = self.dist_z_gal.pdf(
            self.tde_data['z_gal'],
            lam=x["lam_z_gal"]
        )

        alpha_TDE = torch.tensor(1.0, dtype=torch.float32)
        logL = torch.sum(torch.log(
            torch.nanmean(pdf_mass * pdf_z_gal, dim=1) - torch.log(alpha_TDE) + 1e-10
        ))
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

labels_dict = get_latex_labels("A")
truths_dict = get_true_x_flat("A", args.mbh_mass_method, args.nucleation_method, args.occupation_method)

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