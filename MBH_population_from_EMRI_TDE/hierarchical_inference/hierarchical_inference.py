#!/usr/bin/env python

import argparse
from utils import *
import numpy as np
import torch

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger
import h5py
import corner
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# Use CPU for inference to avoid CUDA multiprocessing issues
# GPU can be used for data loading separately if needed
device = torch.device("cpu")
print(f"Using device: {device}")

base_parser = argparse.ArgumentParser(add_help=False)
base_parser.add_argument('--POPULATION', type=str, required=True, choices=['A', 'B', 'MIX'])
base_parser.add_argument('--is_clustering', type=int, default=0)

args_partial, remaining_argv = base_parser.parse_known_args()

parser = argparse.ArgumentParser(parents=[base_parser])

parser.add_argument('--GALAXIES', type=int, required=True)
parser.add_argument('--SOURCE', type=str, required=True, choices=['EMRI', 'TDE', 'EMRI_TDE'])
parser.add_argument('--SURVEY', type=str, required=False, choices=['ZTF', 'LSST', 'ALL', 'None'], help="Which survey's TDE data to use. If ALL is chosen, we use both ZTF and LSST data")

add_args(args_partial.POPULATION, parser)

args = parser.parse_args()

true_x = get_true_x(args.POPULATION)
popdist = make_dist(args.POPULATION)

if args.POPULATION == "A":
    x_c_min, x_c_max, lam_schechter_min, lam_schechter_max, beta_min, beta_max, lambda_alpha_min, lambda_alpha_max, mu_sBH_min, mu_sBH_max, mu_star_min, mu_star_max = get_min_max(args)
elif args.POPULATION == "B":
    lambda_M_min, lambda_M_max, mu_MBHSpin_min, mu_MBHSpin_max, sigma_MBHSpin_min, sigma_MBHSpin_max, mu_sBH_min, mu_sBH_max, mu_star_min, mu_star_max = get_min_max(args)
elif args.POPULATION == "MIX":
    None

# trained_model_path = './data/EMRI/trained_models'

data_EMRI = {}
data_TDE = {}

# load data
hf_EMRI = h5py.File(f'./data/inference_data/{args.GALAXIES}/true_data_EMRI_{ "clustering" if args.is_clustering else "no_clustering" }.h5', 'r')

if args.SURVEY in ['ZTF', 'LSST']:
    
    data_EMRI_torch = torch.load(f'./data/inference_data/{args.GALAXIES}/true_data_EMRI_{ "clustering" if args.is_clustering else "no_clustering" }_torch.pt', map_location='cpu')
    data_TDE_torch = torch.load(f'./data/inference_data/{args.GALAXIES}/true_data_TDE_{args.SURVEY}_{ "clustering" if args.is_clustering else "no_clustering" }_torch.pt', map_location='cpu')
    for key in true_x:
        data_EMRI[key] = data_EMRI_torch[key]
        data_TDE[key] = data_TDE_torch[key]
elif args.SURVEY == 'ALL':
    
    data_EMRI_torch = torch.load(f'./data/inference_data/{args.GALAXIES}/true_data_EMRI_{ "clustering" if args.is_clustering else "no_clustering" }_torch.pt', map_location='cpu')
    data_TDE_ZTF_torch = torch.load(f'./data/inference_data/{args.GALAXIES}/true_data_TDE_ZTF_{ "clustering" if args.is_clustering else "no_clustering" }_torch.pt', map_location='cpu')
    data_TDE_LSST_torch = torch.load(f'./data/inference_data/{args.GALAXIES}/true_data_TDE_LSST_{ "clustering" if args.is_clustering else "no_clustering" }_torch.pt', map_location='cpu')
    for key in true_x:
        data_EMRI[key] = data_EMRI_torch[key]
        data_TDE[key] = torch.cat([data_TDE_ZTF_torch[key], data_TDE_LSST_torch[key]], dim=0)
else:
    
    data_EMRI_torch = torch.load(f'./data/inference_data/{args.GALAXIES}/true_data_EMRI_{ "clustering" if args.is_clustering else "no_clustering" }_torch.pt', map_location='cpu')
    for key in true_x:
        data_EMRI[key] = data_EMRI_torch[key]

hf_EMRI.close()

def get_data(source):
    if source == 'TDE':
        return {'TDE' : data_TDE}, 0, 1
    elif source == 'EMRI_TDE':
        return {'EMRI' : data_EMRI, 'TDE' : data_TDE}, 1, 1
    elif source == 'EMRI':
        return {'EMRI' : data_EMRI}, 1, 0

popdist.data, has_EMRI, has_TDE = get_data(args.SOURCE)

THETA_PARAMETER_NAMES = ['lgMBH_mass', 'MBHspin', 'CO_mass']

output = f"./data/inference_data/{args.GALAXIES}/{args.SOURCE}_pop_{args.POPULATION}"
logger = setup_logger(output=output)

class PopulationModel(Model):
    def __init__(self, popdist, population):
        super().__init__()
        self.data = popdist
        self.population = population
        
        shared_pop_params, emri_specific_pop_params, tde_specific_pop_params = self.get_parameter_names()

        self.bounds = self.get_bounds()
        
        self.names = shared_pop_params + emri_specific_pop_params + tde_specific_pop_params

        if has_EMRI:
            self.emri_data = self.data.data.get('EMRI', None)

            self.emri_lgMBH = self.emri_data["lgMBH_mass"]
            self.emri_spin = self.emri_data["MBHspin"]
            self.emri_CO = self.emri_data["CO_mass"]

            self.dist_lgMBH_EMRI = self.data.distributions['lgMBH_mass']
            self.dist_MBHspin_EMRI = self.data.distributions['MBHspin']
            self.dist_CO_EMRI = self.data.distributions['CO_mass']

        if has_TDE:
            self.tde_data = self.data.data.get('TDE', None)

            self.tde_lgMBH = self.tde_data["lgMBH_mass"]
            self.tde_CO = self.tde_data["CO_mass"]

            self.dist_lgMBH_TDE = self.tde_data.distributions['lgMBH_mass']
            self.dist_CO_TDE = self.tde_data.distributions['CO_mass']

    def get_parameter_names(self):
        if self.population == "A":
            shared_pop_params = ["xc", "lam_schechter"]
            emri_specific_pop_params = ["beta", "lambda_alpha", "mu_sBH"]
            tde_specific_pop_params = ["mu_star"]
            return shared_pop_params, emri_specific_pop_params, tde_specific_pop_params
        elif self.population == "B":
            shared_pop_params = ["lambda_M", "mu_MBHSpin", "sigma_MBHSpin"]
            emri_specific_pop_params = ["mu_sBH"]
            tde_specific_pop_params = ["mu_star"]
            return shared_pop_params, emri_specific_pop_params, tde_specific_pop_params
        elif self.population == "MIX":
            None
    
    def get_bounds(self):
        if self.population == "A":
            return {
                    'xc': [x_c_min, x_c_max],
                    'lam_schechter': [lam_schechter_min, lam_schechter_max],
                    'beta': [beta_min, beta_max],
                    'lambda_alpha': [lambda_alpha_min, lambda_alpha_max],
                    'mu_sBH': [mu_sBH_min, mu_sBH_max],
                    'mu_star': [mu_star_min, mu_star_max],
                    }
        elif self.population == "B":
            return {
                    'lambda_M': [lambda_M_min, lambda_M_max],
                    'mu_MBHSpin': [mu_MBHSpin_min, mu_MBHSpin_max],
                    'sigma_MBHSpin': [sigma_MBHSpin_min, sigma_MBHSpin_max],
                    'mu_sBH': [mu_sBH_min, mu_sBH_max],
                    'mu_star': [mu_star_min, mu_star_max]
                    }
        elif self.population == "MIX":
            return None
 
    def log_prior(self, x):
        log_p = np.log(self.in_bounds(x), dtype="float")
        for n in self.names:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p
    

    def EMRI_likelihood(self, shared_params_trial_x, emri_trial_x):
        if has_EMRI:
            pdf_lgMBH_mass_EMRI = self.dist_lgMBH_EMRI.pdf(self.emri_data['lgMBH_mass'], **shared_params_trial_x['lgMBH_mass'])
            pdf_MBHspin_EMRI = self.dist_MBHspin_EMRI.pdf(self.emri_data['MBHspin'], lgMBH=self.emri_data['lgMBH_mass'], **emri_trial_x['MBHspin'])
            pdf_CO_mass_EMRI = self.dist_CO_EMRI.pdf(self.emri_data['CO_mass'], **shared_params_trial_x['CO_mass'])

            combined_pdf = pdf_lgMBH_mass_EMRI * pdf_MBHspin_EMRI * pdf_CO_mass_EMRI
            
            # Average over posterior samples (dim=1), then sum over objects
            log_likelihood = torch.sum(torch.log(torch.nanmean(combined_pdf, dim=1) + 1e-10))

            return log_likelihood.item()
        
        else :
            return 0.0
        
    def TDE_likelihood(self, shared_params_trial_x, tde_trial_x):
        if has_TDE:
            # ---- TDE likelihood contribution ----
            pdf_lgMBH_mass_TDE  = self.dist_lgMBH_TDE.pdf(self.tde_data['lgMBH_mass'], **shared_params_trial_x['lgMBH_mass'])
            pdf_CO_mass_TDE = self.dist_CO_TDE.pdf(self.tde_data['CO_mass'], **shared_params_trial_x['CO_mass'])
            combined_pdf = pdf_lgMBH_mass_TDE * pdf_CO_mass_TDE
            
            log_likelihood = torch.sum(torch.log(torch.nanmean(combined_pdf, dim=1) + 1e-10))
            
            return log_likelihood.item()
        
        else :
            return 0.0

    @torch.inference_mode()
    def log_likelihood(self, x):

        if isinstance(x, np.void):
            x = {name: np.array(x[name], dtype=np.float32) for name in x.dtype.names}
        elif isinstance(x, np.ndarray):
            x = {name: np.ascontiguousarray(x[name], dtype=np.float32) for name in x.dtype.names}
        
        x = {
            k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
            for k, v in x.items()
        }
        
        logL_EMRI = self.EMRI_likelihood(
            {
                "lgMBH_mass": {"xc": x["xc"], "lam_schechter": x["lam_schechter"]},
                "CO_mass": {"mu_sBH": x["mu_sBH"], "mu_star": x["mu_star"]},
            },
            {
                "MBHspin": {"beta": x["beta"], "lambda_alpha": x["lambda_alpha"]},
            }
        )

        logL_TDE = self.TDE_likelihood(
            {
                "lgMBH_mass": {"xc": x["xc"], "lam_schechter": x["lam_schechter"]},
                "CO_mass": {"mu_sBH": x["mu_sBH"], "mu_star": x["mu_star"]},
            },
            {}
        )

        return logL_EMRI + logL_TDE
    
fs = FlowSampler(PopulationModel(popdist=popdist, population=args.POPULATION), output=output, resume=False, seed=123123, nlive=1000, n_pool=8)
fs.run()

## plot the corner plot
hf = h5py.File(f'{output}/result.hdf5', 'r')

posterior_samples = np.vstack(list(get_posterior_samples(args.POPULATION, hf).values())).T
truths, labels = [], []
for params in list(get_true_x(args.POPULATION).values()):
    for keys in params.keys():
        print(keys, params[keys])
        truths.append(params[keys])

for key in get_latex_labels(args.POPULATION).keys():
    labels.append(get_latex_labels(args.POPULATION)[key])

print(truths, labels)

figure = corner.corner(posterior_samples, truths=truths, labels=labels, title_kwargs={"fontsize": 18}, show_titles=True, title_fmt=".3f", quantiles=[0.16, 0.5, 0.84], label_kwargs={"fontsize": 18}, color='blue', truth_color='red', truth_kwargs={"markersize": 10, "marker": "X", "color": "red"})
plt.savefig(f'{output}/hyperposterior.png', dpi=200)