#!/usr/bin/env python

import argparse
from utils import *
import numpy as np

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger
import h5py
import corner
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

base_parser = argparse.ArgumentParser(add_help=False)
base_parser.add_argument('--POPULATION', type=str, required=True, choices=['A', 'B', 'MIX'])

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
    x_c_min, x_c_max, lam_schechter_min, lam_schechter_max, beta_min, beta_max, lambda_alpha_min, lambda_alpha_max = get_min_max(args)
elif args.POPULATION == "B":
    lambda_M_min, lambda_M_max, mu_a_min, mu_a_max, sigma_a_min, sigma_a_max = get_min_max(args)
elif args.POPULATION == "MIX":
    None

# trained_model_path = '/data/wiay/postgrads/shashwat/EMRI_data/trained_models'

data_EMRI = {}
data_TDE = {}

# load data
hf_EMRI = h5py.File(f'/data/wiay/postgrads/shashwat/EMRI_TDE_data/inference_data/{args.GALAXIES}/true_data_EMRI.h5', 'r')

if args.SURVEY in ['ZTF', 'LSST']:
    hf_TDE = h5py.File(f'/data/wiay/postgrads/shashwat/EMRI_TDE_data/inference_data/{args.GALAXIES}/true_data_TDE_{args.SURVEY}.h5', 'r')
    for key in true_x:
        data_EMRI[key] = np.array(hf_EMRI.get(key))
        data_TDE[key] = np.array(hf_TDE.get(key))
elif args.SURVEY == 'ALL':
    hf_TDE_ZTF = h5py.File(f'/data/wiay/postgrads/shashwat/EMRI_TDE_data/inference_data/{args.GALAXIES}/true_data_TDE_ZTF.h5', 'r')
    hf_TDE_LSST = h5py.File(f'/data/wiay/postgrads/shashwat/EMRI_TDE_data/inference_data/{args.GALAXIES}/true_data_TDE_LSST.h5', 'r')
    for key in true_x:
        data_EMRI[key] = np.array(hf_EMRI.get(key))
        data_TDE[key] = np.concatenate([np.array(hf_TDE_ZTF.get(key)), np.array(hf_TDE_LSST.get(key))], axis=0)
else:
    for key in true_x:
        data_EMRI[key] = np.array(hf_EMRI.get(key))

hf_EMRI.close()

if args.SURVEY in ['ZTF', 'LSST']:
    hf_TDE.close()
elif args.SURVEY == 'ALL':
    hf_TDE_ZTF.close()
    hf_TDE_LSST.close()

def get_data(source):
    if source == 'TDE':
        return {'TDE' : data_TDE}, 0, 1
    elif source == 'EMRI_TDE':
        return {'EMRI' : data_EMRI, 'TDE' : data_TDE}, 1, 1
    elif source == 'EMRI':
        return {'EMRI' : data_EMRI}, 1, 0

popdist.data, has_EMRI, has_TDE = get_data(args.SOURCE)

output = f"/data/wiay/postgrads/shashwat/EMRI_TDE_data/inference_data/{args.GALAXIES}/{args.SOURCE}_pop_{args.POPULATION}"
logger = setup_logger(output=output)

class PopulationModel(Model):
    def __init__(self, popdist, population):
        super().__init__()
        self.data = popdist
        self.population = population

        
        shared_pop_params, emri_specific_pop_params, tde_specific_pop_params = self.get_parameter_names()

        # emri_params = ["e0_mu", "e0_sigma"]
        # tde_params = ["photosphere_mu", "photosphere_sigma"]

        self.bounds = self.get_bounds()
        
        self.names = shared_pop_params + emri_specific_pop_params + tde_specific_pop_params

    def get_parameter_names(self):
        if self.population == "A":
            shared_pop_params = ["xc", "lam_schechter"]
            emri_specific_pop_params = ["beta", "lambda_alpha"]
            tde_specific_pop_params = []
            return shared_pop_params, emri_specific_pop_params, tde_specific_pop_params
        elif self.population == "B":
            shared_pop_params = ["lambda_M", "mu_a", "sigma_a"]
            emri_specific_pop_params = []
            tde_specific_pop_params = []
            return shared_pop_params, emri_specific_pop_params, tde_specific_pop_params
        elif self.population == "MIX":
            None
    
    def get_bounds(self):
        if self.population == "A":
            return {
                    'xc': [x_c_min, x_c_max],
                    'lam_schechter': [lam_schechter_min, lam_schechter_max],
                    'beta': [beta_min, beta_max],
                    'lambda_alpha': [lambda_alpha_min, lambda_alpha_max]
                    }
        elif self.population == "B":
            return {
                    'lambda_M': [lambda_M_min, lambda_M_max],
                    'mu_a': [mu_a_min, mu_a_max],
                    'sigma_a': [sigma_a_min, sigma_a_max]
                    }
        elif self.population == "MIX":
            return None
        
    def get_trial_x(self, x):

        if self.population == "A":
            shared_params_trial_x = {
                "lgMBH_mass":  {
                                "xc": torch.from_numpy(np.array(x['xc'])),
                                "lam_schechter": torch.from_numpy(np.array(x['lam_schechter']))
                                },
                "MBHspin":  {
                            "beta": torch.from_numpy(np.array(x['beta'])),
                            "lambda_alpha": torch.from_numpy(np.array(x['lambda_alpha']))
                            },
                    }    

            emri_trial_x = {
                    # "e0": {"mu": torch.tensor(x["e0_mu"]), "sigma": torch.tensor(x["e0_sigma"])}
                        }


            tde_trial_x = {
                            # "photosphere": {"mu": torch.tensor(x["photosphere_mu"]),
                            #         "sigma": torch.tensor(x["photosphere_sigma"])}
                        }   

            return shared_params_trial_x, emri_trial_x, tde_trial_x
        
        if self.population == "B":
            shared_params_trial_x = {
                "lgMBH_mass":  {"lam": torch.from_numpy(np.array(x['lambda_M']))},
                "MBHspin":  {"mu": torch.from_numpy(np.array(x['mu_a'])),
                       "sigma": torch.from_numpy(np.array(x['sigma_a']))},
                       }

            emri_trial_x = {
                    # "e0": {"mu": torch.tensor(x["e0_mu"]), "sigma": torch.tensor(x["e0_sigma"])}
                        }


            tde_trial_x = {
                            # "photosphere": {"mu": torch.tensor(x["photosphere_mu"]),
                            #         "sigma": torch.tensor(x["photosphere_sigma"])}
                        }   

            return shared_params_trial_x, emri_trial_x, tde_trial_x
        
        if self.population == "MIX":
            None
    
    def log_prior(self, x):
        log_p = np.log(self.in_bounds(x), dtype="float")
        for n in self.names:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p
    

    def EMRI_likelihood(self, shared_params_trial_x):
        if has_EMRI:
            # ---- EMRI likelihood contribution ----

            pdf_M_EMRI = self.data.distributions['lgMBH_mass'].pdf(self.data.data['EMRI']['lgMBH_mass'], **shared_params_trial_x['lgMBH_mass'])
            pdf_a_EMRI = self.data.distributions['MBHspin'].pdf(self.data.data['EMRI']['MBHspin'], lgMBH=self.data.data['EMRI']['lgMBH_mass'], **shared_params_trial_x['MBHspin'])

            # breakpoint()
            return torch.sum(torch.log(torch.nanmean(pdf_M_EMRI * pdf_a_EMRI, dim=1)))
        
        else :
            return 0
        
    def TDE_likelihood(self, shared_params_trial_x):
        if has_TDE:
            # ---- EMRI likelihood contribution ----

            pdf_M_TDE  = self.data.distributions['lgMBH_mass'].pdf(self.data.data['TDE']['lgMBH_mass'], **shared_params_trial_x['lgMBH_mass'])
            # pdf_a_TDE  = self.data.distributions['MBHspin'].pdf(self.data.data['TDE']['MBHspin'], **shared_params_trial_x['MBHspin'])
            # pdf_ph_TdE = self.data.distributions['photosphere'].pdf(self.data.data['TDE']['photosphere'], **shared_params_trial_x['photosphere'])

            return torch.sum(torch.log(torch.nanmean(pdf_M_TDE,  dim=1)))
        
        else :
            return 0


    def log_likelihood(self, x):
        
        shared_params_trial_x, emri_trial_x, tde_trial_x = self.get_trial_x(x)
        

        logL_EMRI = self.EMRI_likelihood(shared_params_trial_x)
        logL_TDE = self.TDE_likelihood(shared_params_trial_x)

        # logL_EMRI -= N_E * torch.log(alpha_E(shared, emri_only))
        # logL_TDE  -= N_T * torch.log(alpha_T(shared, tde_o

        return logL_EMRI + logL_TDE
    
fs = FlowSampler(PopulationModel(popdist=popdist, population=args.POPULATION), output=output, resume=False, seed=123123, nlive=1000)
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