#!/usr/bin/env python

import os, argparse
from population import *
import numpy as np
from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger
import h5py, torch, pickle
import corner
import matplotlib.pyplot as plt
from nessai.utils.multiprocessing import initialise_pool_variables
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Generate data for training.")
parser.add_argument('--lambda_M', type=float, nargs=2, help="Lower and upper boundars for lambda_M")
parser.add_argument('--lambda_mu', type=float, nargs=2, help="Lower and upper bounds for lambda_mu")
parser.add_argument('--mu_a', type=float, nargs=2, help="Lower and upper bounds for mu_a")
parser.add_argument('--sigma_a', type=float, nargs=2, help="Lower and upper bounds for sigma_a")
parser.add_argument('--source', type=str, help="Source")
parser.add_argument('--work_dir', type=str, required=True, help="Working directory path")

args = parser.parse_args()

output_dir = args.work_dir 
# trained_model_path = './data/EMRI/trained_models'

lambda_M_min, lambda_M_max = args.lambda_M
lambda_mu_min, lambda_mu_max = args.lambda_mu
mu_a_min, mu_a_max = args.mu_a
sigma_a_min, sigma_a_max = args.sigma_a

### load data
hf_EMRI = h5py.File(f'{args.work_dir}/true_pop_data_EMRI.h5', 'r')
hf_TDE = h5py.File(f'{args.work_dir}/true_pop_data_TDE.h5', 'r')

data_EMRI = {}
data_TDE = {}

for key in true_x_B:
    data_EMRI[key] = np.array(hf_EMRI.get(key))
    data_TDE[key] = np.array(hf_TDE.get(key))

def get_data(source):
    if source == 'TDE':
        return {'TDE' : data_TDE}, 0, 1
    elif source == 'EMRI_TDE':
        return {'EMRI' : data_EMRI, 'TDE' : data_TDE}, 1, 1
    elif source == 'EMRI':
        return {'EMRI' : data_EMRI}, 1, 0


popdist_B.data, has_EMRI, has_TDE = get_data(args.source)


output = f"{args.work_dir}/inference/{args.source}"
logger = setup_logger(output=output)

class PopulationModel(Model):
    def __init__(self, popdist):
        super().__init__()
        self.data = popdist

        shared_params = ["lambda_M", "lambda_mu", "mu_a", "sigma_a"]
        # emri_params = ["e0_mu", "e0_sigma"]
        # tde_params = ["photosphere_mu", "photosphere_sigma"]

        self.bounds = {
                    'lambda_M': [lambda_M_min, lambda_M_max],
                    'lambda_mu': [lambda_mu_min, lambda_mu_max],
                    'mu_a': [mu_a_min, mu_a_max],
                    'sigma_a': [sigma_a_min, sigma_a_max],
                    # # EMRI-ONLY
                    # "e0_mu": [e0_mu_min, e0_mu_max],
                    # "e0_sigma": [e0_sigma_min, e0_sigma_max],
                    # # TDE-ONLY
                    # "photosphere_mu": [ph_mu_min, ph_mu_max],
                    # "photosphere_sigma": [ph_sigma_min, ph_sigma_max],
                    }
        
        self.names = shared_params #+ emri_params + tde_params

        
    def log_prior(self, x):
        log_p = np.log(self.in_bounds(x), dtype="float")
        for n in self.names:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p
    

    def EMRI_likelihood(self, shared_params_trial_x):
        if has_EMRI:
            # ---- EMRI likelihood contribution ----

            pdf_M_EMRI = self.data.distributions['log10_M'].pdf(self.data.data['EMRI']['log10_M'], **shared_params_trial_x['log10_M'])
            pdf_mu_EMRI = self.data.distributions['log10_mu'].pdf(self.data.data['EMRI']['log10_mu'], **shared_params_trial_x['log10_mu'])
            pdf_a_EMRI = self.data.distributions['a'].pdf(self.data.data['EMRI']['a'], **shared_params_trial_x['a'])

            return torch.sum(torch.log(torch.nanmean(pdf_M_EMRI * pdf_mu_EMRI * pdf_a_EMRI, dim=1)))
        
        else :
            return 0
        
    def TDE_likelihood(self, shared_params_trial_x):
        if has_TDE:
            # ---- EMRI likelihood contribution ----

            pdf_M_TDE  = self.data.distributions['log10_M'].pdf(self.data.data['TDE']['log10_M'], **shared_params_trial_x['log10_M'])
            pdf_mu_TDE = self.data.distributions['log10_mu'].pdf(self.data.data['TDE']['log10_mu'], **shared_params_trial_x['log10_mu'])
            pdf_a_TDE  = self.data.distributions['a'].pdf(self.data.data['TDE']['a'], **shared_params_trial_x['a'])
            # pdf_ph_TdE = self.data.distributions['photosphere'].pdf(self.data.data['TDE']['photosphere'], **shared_params_trial_x['photosphere'])

            return torch.sum(torch.log(torch.nanmean(pdf_M_TDE * pdf_mu_TDE * pdf_a_TDE, dim=1)))
        
        else :
            return 0


    def log_likelihood(self, x):
        
        shared_params_trial_x = {
                "log10_M":  {"lam": torch.from_numpy(np.array(x['lambda_M']))},
                "log10_mu": {"lam": torch.from_numpy(np.array(x['lambda_mu']))},
                "a":  {"mu": torch.from_numpy(np.array(x['mu_a'])),
                       "sigma": torch.from_numpy(np.array(x['sigma_a']))},
                       }


        # emri_trial_x = {
        #         "e0": {"mu": torch.tensor(x["e0_mu"]), "sigma": torch.tensor(x["e0_sigma"])}
        #             }


        # tde_trial_x = {
        #         "photosphere": {"mu": torch.tensor(x["photosphere_mu"]),
        #                         "sigma": torch.tensor(x["photosphere_sigma"])}
        #             }   

        

        logL_EMRI = self.EMRI_likelihood(shared_params_trial_x)
        logL_TDE = self.TDE_likelihood(shared_params_trial_x)

        # logL_EMRI -= N_E * torch.log(alpha_E(shared, emri_only))
        # logL_TDE  -= N_T * torch.log(alpha_T(shared, tde_o

        return logL_EMRI + logL_TDE
    
fs = FlowSampler(PopulationModel(popdist_B), output=output, resume=False, seed=123123, nlive=1000)
fs.run()

## plot the corner plot
hf = h5py.File(f'{output}/result.hdf5', 'r')
lambda_M = np.array(hf.get('posterior_samples')['lambda_M'])
lambda_mu = np.array(hf.get('posterior_samples')['lambda_mu'])
mu_a = np.array(hf.get('posterior_samples')['mu_a'])
sigma_a = np.array(hf.get('posterior_samples')['sigma_a'])

samples = np.vstack([lambda_M, lambda_mu, mu_a, sigma_a]).T

truths = [
            true_x_B['log10_M']['lam'], true_x_B['log10_mu']['lam'],
            true_x_B['a']['mu'], true_x_B['a']['sigma'],
        ]

labels = ["$\\Lambda_M$", "$\\Lambda_{\\mu}$", "$\\mu_a$", "$\\sigma_a$"]

figure = corner.corner(samples, truths=truths, labels = labels, title_kwargs={"fontsize": 18}, show_titles=True)

plt.savefig(f'{output}/hyperposterior.png', dpi=200)