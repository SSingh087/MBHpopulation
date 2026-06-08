#!/usr/bin/env python

import os, sys
sys.path.insert(0, os.path.abspath('../poplar'))
from poplar.distributions import *
import re
import torch
import numpy as np

device = "cpu"

def get_min_max(args):

    if args.POPULATION == "A":
        x_c_min, x_c_max = args.xc
        lam_schechter_min, lam_schechter_max = args.lam_schechter
        beta_min, beta_max = args.beta
        lambda_alpha_min, lambda_alpha_max = args.lambda_alpha
        mu_sBH_min, mu_sBH_max = args.mu_sBH
        mu_star_min, mu_star_max = args.mu_star
        return x_c_min, x_c_max, lam_schechter_min, lam_schechter_max, beta_min, beta_max, lambda_alpha_min, lambda_alpha_max, mu_sBH_min, mu_sBH_max, mu_star_min, mu_star_max

    elif args.POPULATION == "B":
        lambda_M_min, lambda_M_max = args.lambda_M
        mu_MBHSpin_min, mu_MBHSpin_max = args.mu_MBHSpin
        sigma_MBHSpin_min, sigma_MBHSpin_max = args.sigma_MBHSpin
        mu_sBH_min, mu_sBH_max = args.mu_sBH
        mu_star_min, mu_star_max = args.mu_star
        return lambda_M_min, lambda_M_max, mu_MBHSpin_min, mu_MBHSpin_max, sigma_MBHSpin_min, sigma_MBHSpin_max, mu_sBH_min, mu_sBH_max, mu_star_min, mu_star_max

    elif args.POPULATION == "MIX":
        None

def add_args(pop, parser):
    if pop == "A":
       parser.add_argument('--xc', type=float, nargs=2, help="Lower and upper bounds for xc in population A")
       parser.add_argument('--lam_schechter', type=float, nargs=2, help="Lower and upper bounds for lam_schechter in population A")
       parser.add_argument('--beta', type=float, nargs=2, help="Lower and upper bounds for beta in population A")
       parser.add_argument('--lambda_alpha', type=float, nargs=2, help="Lower and upper bounds for lambda_alpha in population A")
       parser.add_argument('--mu_sBH', type=float, nargs=2, help="Lower and upper bounds for mu_sBH in population A")
       parser.add_argument('--mu_star', type=float, nargs=2, help="Lower and upper bounds for mu_star in population A")
    
    elif pop == "B":
        parser.add_argument('--lambda_M', type=float, nargs=2, help="Lower and upper boundars for lambda_M")
        parser.add_argument('--mu_MBHSpin', type=float, nargs=2, help="Lower and upper bounds for mu_MBHSpin")
        parser.add_argument('--sigma_MBHSpin', type=float, nargs=2, help="Lower and upper bounds for sigma_MBHSpin")
        parser.add_argument('--mu_sBH', type=float, nargs=2, help="Lower and upper bounds for mu_sBH in population B")
        parser.add_argument('--mu_star', type=float, nargs=2, help="Lower and upper bounds for mu_star in population B")

    elif pop == "MIX":
        None
    
def make_dist(pop):

    if pop == "A":
        DIST = {
                "lgMBH_mass": FixedLimitSchechterFunction([4, 10], device=device),
                "MBHspin" : FixedLimitMassDependentTruncatedBetaDistribution([0.01, 0.99], device=device),
                "CO_mass" : DoubleGaussianDistribution([0.1, 15], device=device)
            }   
        return PopulationDistribution(distributions=DIST, data=None)

    elif pop =="B":
        DIST = {
                "lgMBH_mass": FixedLimitsPowerLaw([4, 10], device=device),
                "MBHspin" : FixedLimitsTruncatedGaussian([0.01, 0.99], device=device),
                "CO_mass" : DoubleGaussianDistribution([0.1, 15], device=device)
            }
        return PopulationDistribution(distributions=DIST, data=None)

    elif pop=="MIX":
        DIST = {

        }
        return PopulationDistribution(distributions=DIST, data=None)

def get_true_x(pop):

    true_x = {}

    if pop == "A":
        return {
                "lgMBH_mass": {"xc": 5.5, "lam_schechter": 10},
                "MBHspin": {'beta': 12.0, 'lambda_alpha': 0.5},
                "CO_mass": {"mu_sBH": 10.0, "mu_star": 1.0}
            }

    if pop == "B":
        return{
                "lgMBH_mass": {"lam": -2.5},
                "MBHspin": {'mu': 0.7, 'sigma': 0.03},
                "CO_mass": {"mu_sBH": 10.0, "mu_star": 1.0}
                }

    if pop == "MIX":
        return{

                }

    return true_x

def get_posterior_samples(pop, hf):
    
    if pop == "A":
        return {
                "xc": np.array(hf.get('posterior_samples')['xc']),
                "lam_schechter": np.array(hf.get('posterior_samples')['lam_schechter']),
                "beta": np.array(hf.get('posterior_samples')['beta']),
                "lambda_alpha": np.array(hf.get('posterior_samples')['lambda_alpha']),
                "mu_sBH": np.array(hf.get('posterior_samples')['mu_sBH']),
                "mu_star": np.array(hf.get('posterior_samples')['mu_star']),
            }

    if pop == "B":
        return{
                "lambda_M": np.array(hf.get('posterior_samples')['lambda_M']),
                "mu_MBHSpin": np.array(hf.get('posterior_samples')['mu_MBHSpin']),
                "sigma_MBHSpin": np.array(hf.get('posterior_samples')['sigma_MBHSpin']),
                "mu_sBH": np.array(hf.get('posterior_samples')['mu_sBH']),
                "mu_star": np.array(hf.get('posterior_samples')['mu_star']),
                }

    if pop == "MIX":
        return{

                }

def get_latex_labels(pop):

    if pop == "A":
        return {
                "xc": "$x_c$",
                "lam_schechter": "$\\lambda_{x_c}$",
                "beta": "$\\beta$",
                "lambda_alpha": "$\\lambda_{\\alpha}$",
                "mu_sBH": "$\\mu_{sBH}$",
                "mu_star": "$\\mu_{\\star}$"
            }

    if pop == "B":
        return{
                "lambda_M": "$\\lambda_M$",
                "mu_MBHSpin": "$\\mu_a$",
                "sigma_MBHSpin": "$\\sigma_a$",
                "mu_sBH": "$\\mu_{sBH}$",
                "mu_star": "$\\mu_{\\star}$"
                }

    if pop == "MIX":
        return{

                }   

class PopulationDistribution:
    def __init__(self, distributions, data) -> None:
        self.distributions = distributions
        self.data = data

    def draw_samples(self, x, weight=1.0, size=500):
        out = {}

        self.weight = weight
        
        for key in self.distributions.keys():

            hyperparams = list(x[key].items())
            cleaned_hyperparams_A = {re.sub(r'_A$', '', k): v for k, v in hyperparams if k.endswith('_A')} # k.replace
            cleaned_hyperparams_B = {re.sub(r'_B$', '', k): v for k, v in hyperparams if k.endswith('_B')}
            
            if not cleaned_hyperparams_A and not cleaned_hyperparams_B:  
                # If no '_A' or '_B' suffixes exist, use params directly
                out[key] = self.distributions[key].draw_samples(**x[key], size=int(size))
            
            else:

                choices = torch.bernoulli(torch.full((size,), self.weight)).bool()
                
                if cleaned_hyperparams_A == {'UNIFORM': {}}:
                    samples_A = self.distributions[key][0].draw_samples(size=size)
                    samples_B = self.distributions[key][1].draw_samples(**cleaned_hyperparams_B, size=size)

                elif cleaned_hyperparams_B == {'UNIFORM': {}}:
                    samples_A = self.distributions[key][0].draw_samples(**cleaned_hyperparams_A, size=size)
                    samples_B = self.distributions[key][1].draw_samples(size=size)

                else :
                    samples_A = self.distributions[key][0].draw_samples(**cleaned_hyperparams_A, size=size)
                    samples_B = self.distributions[key][1].draw_samples(**cleaned_hyperparams_B, size=size)

                # Select from A or B based on choices
                out[key] = torch.where(choices, samples_A, samples_B)

        return out