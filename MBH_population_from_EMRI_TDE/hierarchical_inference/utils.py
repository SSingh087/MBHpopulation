#!/usr/bin/env python

import os, sys
# sys.path.insert(0, os.path.abspath('../poplar'))
sys.path.insert(0, os.path.abspath('../../poplar'))
from poplar.distributions import *
import numpy as np

device = "cpu"
 
def make_dist(pop):

    if pop == "A":
        DIST = {
                "lgMBH_mass": FixedLimitSchechterFunction([4, 10], device=device),
                "MBHspin" : DoubleBetaDistribution([0.01, 0.99], device=device),
                "CO_mass" : DoubleGaussianDistribution([0.1, 15], device=device),
                "e_pl" : DoubleBetaDistribution([0.01, 0.99], device=device),
                "z_gal" : FixedLimitsPowerLaw([0.01, 10.0], device=device)
            }   
        return PopulationDistribution(distributions=DIST, data=None)

    elif pop =="B":
        DIST = {
                "lgMBH_mass": FixedLimitsPowerLaw([4, 10], device=device),
                "MBHspin" : DoubleGaussianDistribution([0.01, 0.99], device=device),
                "CO_mass" : DoubleGaussianDistribution([0.1, 15], device=device),
                "e_pl" : DoubleGaussianDistribution([0.01, 0.99], device=device),
                "z_gal" : FixedLimitsPowerLaw([0.01, 10.0], device=device)
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
                "lgMBH_mass": {"xc": 5.5, "lam_schechter": 10.0},
                "MBHspin": {'alpha_1': 2.5, 'alpha_2': 9.5, 'beta_1': 5.0, 'beta_2': 5.0, 'weight': 0.7},
                "CO_mass": {"mu_1": 1.0, "sigma_1": 0.5, "mu_2": 10.0, "sigma_2": 0.5, "weight": 0.5},
                "e_pl": {'alpha_1': 2.0, 'alpha_2': 9.0, 'beta_1': 5.0, 'beta_2': 5.0, 'weight': 0.65},
                "z_gal": {"lam": -3}
            }

    if pop == "B":
        return{
                "lgMBH_mass": {"lam": -2.5},
                "MBHspin": {'mu_1': 0.3, 'sigma_1': 0.1, 'mu_2': 0.7, 'sigma_2': 0.1, 'weight': 0.7},
                "CO_mass": {"mu_1": 1.0, "sigma_1": 0.5, "mu_2": 10.0, "sigma_2": 0.5, "weight": 0.5},
                "e_pl": {'mu_1': 0.2, 'sigma_1': 0.1, 'mu_2': 0.8, 'sigma_2': 0.1, 'weight': 0.65},
                "z_gal": {"lam": -3}
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

                "alpha_1_MBHSpin": np.array(hf.get('posterior_samples')['alpha_1_MBHSpin']),
                "alpha_2_MBHSpin": np.array(hf.get('posterior_samples')['alpha_2_MBHSpin']),
                "beta_1_MBHSpin": np.array(hf.get('posterior_samples')['beta_1_MBHSpin']),
                "beta_2_MBHSpin": np.array(hf.get('posterior_samples')['beta_2_MBHSpin']),
                "weight_MBHSpin": np.array(hf.get('posterior_samples')['weight_MBHSpin']),
                
                "alpha_1_e_pl": np.array(hf.get('posterior_samples')['alpha_1_e_pl']),
                "alpha_2_e_pl": np.array(hf.get('posterior_samples')['alpha_2_e_pl']),
                "beta_1_e_pl": np.array(hf.get('posterior_samples')['beta_1_e_pl']),
                "beta_2_e_pl": np.array(hf.get('posterior_samples')['beta_2_e_pl']),
                "weight_e_pl": np.array(hf.get('posterior_samples')['weight_e_pl']),                

                "mu_sBH": np.array(hf.get('posterior_samples')['mu_sBH']),
                "mu_star": np.array(hf.get('posterior_samples')['mu_star']),
                "sigma_sBH": np.array(hf.get('posterior_samples')['sigma_sBH']),
                "sigma_star": np.array(hf.get('posterior_samples')['sigma_star']),
                "weight_COmass": np.array(hf.get('posterior_samples')['weight_COmass']),
               
                "lam_z_gal": np.array(hf.get('posterior_samples')['lam_z_gal'])
            }

    if pop == "B":
        return{
                "lambda_M": np.array(hf.get('posterior_samples')['lambda_M']),
                
                "mu_1_MBHSpin": np.array(hf.get('posterior_samples')['mu_1_MBHSpin']),
                "sigma_1_MBHSpin": np.array(hf.get('posterior_samples')['sigma_1_MBHSpin']),
                "mu_2_MBHSpin": np.array(hf.get('posterior_samples')['mu_2_MBHSpin']),
                "sigma_2_MBHSpin": np.array(hf.get('posterior_samples')['sigma_2_MBHSpin']),
                "weight_MBHSpin": np.array(hf.get('posterior_samples')['weight_MBHSpin']),
                
                "mu_1_e_pl":    np.array(hf.get('posterior_samples')['mu_1_e_pl']),
                "sigma_1_e_pl": np.array(hf.get('posterior_samples')['sigma_1_e_pl']),
                "mu_2_e_pl":    np.array(hf.get('posterior_samples')['mu_2_e_pl']),
                "sigma_2_e_pl": np.array(hf.get('posterior_samples')['sigma_2_e_pl']),
                "weight_e_pl":  np.array(hf.get('posterior_samples')['weight_e_pl']),

                "mu_sBH": np.array(hf.get('posterior_samples')['mu_sBH']),
                "mu_star": np.array(hf.get('posterior_samples')['mu_star']),
                "sigma_sBH": np.array(hf.get('posterior_samples')['sigma_sBH']),
                "sigma_star": np.array(hf.get('posterior_samples')['sigma_star']),
                "weight_COmass": np.array(hf.get('posterior_samples')['weight_COmass']),
                
                "lam_z_gal": np.array(hf.get('posterior_samples')['lam_z_gal'])
                }

    if pop == "MIX":
        return{

                }

def get_latex_labels(pop):

    if pop == "A":
        return {
            "xc": r"$x_c$",
            "lam_schechter": r"$\lambda_{x_c}$",

            "alpha_1_MBHSpin": r"$\alpha_{1,a}$",
            "alpha_2_MBHSpin": r"$\alpha_{2,a}$",
            "beta_1_MBHSpin": r"$\beta_{1,a}$",
            "beta_2_MBHSpin": r"$\beta_{2,a}$",
            "weight_MBHSpin": r"$w_a$",

            "alpha_1_e_pl": r"$\alpha_{1,e}$",
            "alpha_2_e_pl": r"$\alpha_{2,e}$",
            "beta_1_e_pl": r"$\beta_{1,e}$",
            "beta_2_e_pl": r"$\beta_{2,e}$",
            "weight_e_pl": r"$w_e$",

            "mu_sBH": r"$\mu_{\rm sBH}$",
            "mu_star": r"$\mu_{\star}$",
            "sigma_sBH": r"$\sigma_{\rm sBH}$",
            "sigma_star": r"$\sigma_{\star}$",
            "weight_COmass": r"$w_{\rm CO}$",

            "lam_z_gal": r"$\lambda_z$"
        }

    if pop == "B":
        return {
            "lambda_M": r"$\lambda_M$",

            "mu_1_MBHSpin": r"$\mu_{1,a}$",
            "sigma_1_MBHSpin": r"$\sigma_{1,a}$",
            "mu_2_MBHSpin": r"$\mu_{2,a}$",
            "sigma_2_MBHSpin": r"$\sigma_{2,a}$",
            "weight_MBHSpin": r"$w_a$",

            "mu_1_e_pl": r"$\mu_{1,e}$",
            "sigma_1_e_pl": r"$\sigma_{1,e}$",
            "mu_2_e_pl": r"$\mu_{2,e}$",
            "sigma_2_e_pl": r"$\sigma_{2,e}$",
            "weight_e_pl": r"$w_e$",

            "mu_sBH": r"$\mu_{\rm sBH}$",
            "mu_star": r"$\mu_{\star}$",
            "sigma_sBH": r"$\sigma_{\rm sBH}$",
            "sigma_star": r"$\sigma_{\star}$",
            "weight_COmass": r"$w_{\rm CO}$",

            "lam_z_gal": r"$\lambda_z$"
        }

    if pop == "MIX":
        return {}
    
def get_true_x_flat(pop):

    tx = get_true_x(pop)

    if pop == "A":
        return {
            "xc": tx["lgMBH_mass"]["xc"],
            "lam_schechter": tx["lgMBH_mass"]["lam_schechter"],

            "alpha_1_MBHSpin": tx["MBHspin"]["alpha_1"],
            "alpha_2_MBHSpin": tx["MBHspin"]["alpha_2"],
            "beta_1_MBHSpin":  tx["MBHspin"]["beta_1"],
            "beta_2_MBHSpin":  tx["MBHspin"]["beta_2"],
            "weight_MBHSpin":  tx["MBHspin"]["weight"],

            "alpha_1_e_pl": tx["e_pl"]["alpha_1"],
            "alpha_2_e_pl": tx["e_pl"]["alpha_2"],
            "beta_1_e_pl":  tx["e_pl"]["beta_1"],
            "beta_2_e_pl":  tx["e_pl"]["beta_2"],
            "weight_e_pl":  tx["e_pl"]["weight"],

            "mu_sBH":        tx["CO_mass"]["mu_2"],
            "mu_star":       tx["CO_mass"]["mu_1"],
            "sigma_sBH":     tx["CO_mass"]["sigma_2"],
            "sigma_star":    tx["CO_mass"]["sigma_1"],
            "weight_COmass": tx["CO_mass"]["weight"],

            "lam_z_gal": tx["z_gal"]["lam"]
        }

    if pop == "B":
        return {
            "lambda_M": tx["lgMBH_mass"]["lam"],

            "mu_1_MBHSpin":    tx["MBHspin"]["mu_1"],
            "sigma_1_MBHSpin": tx["MBHspin"]["sigma_1"],
            "mu_2_MBHSpin":    tx["MBHspin"]["mu_2"],
            "sigma_2_MBHSpin": tx["MBHspin"]["sigma_2"],
            "weight_MBHSpin":  tx["MBHspin"]["weight"],

            "mu_1_e_pl":    tx["e_pl"]["mu_1"],
            "sigma_1_e_pl": tx["e_pl"]["sigma_1"],
            "mu_2_e_pl":    tx["e_pl"]["mu_2"],
            "sigma_2_e_pl": tx["e_pl"]["sigma_2"],
            "weight_e_pl":  tx["e_pl"]["weight"],

            "mu_sBH":        tx["CO_mass"]["mu_2"],
            "mu_star":       tx["CO_mass"]["mu_1"],
            "sigma_sBH":     tx["CO_mass"]["sigma_2"],
            "sigma_star":    tx["CO_mass"]["sigma_1"],
            "weight_COmass": tx["CO_mass"]["weight"],

            "lam_z_gal": tx["z_gal"]["lam"]
        }

    if pop == "MIX":
        return {}

class PopulationDistribution:
    # this will work until we make mass dependance explicit.
    def __init__(self, distributions, data) -> None:
        self.distributions = distributions
        self.data = data

    def draw_samples(self, x, size):
        out = {}
        lgMBH_mass_samples = self.distributions['lgMBH_mass'].draw_samples(**x['lgMBH_mass'], size=size)
        MBHspin_samples = self.distributions['MBHspin'].draw_samples(**x['MBHspin'], size=size)
        COMass_samples = self.distributions['CO_mass'].draw_samples(**x['CO_mass'], size=size)
        e_pl_samples = self.distributions['e_pl'].draw_samples(**x['e_pl'], size=size)
        z_gal_samples = self.distributions['z_gal'].draw_samples(**x['z_gal'], size=size)
        
        out['lgMBH_mass'] = lgMBH_mass_samples
        out['MBHspin'] = MBHspin_samples
        out['CO_mass'] = COMass_samples
        out['e_pl'] = e_pl_samples
        out['z_gal'] = z_gal_samples
        return out