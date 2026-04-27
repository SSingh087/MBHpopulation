import os, sys
sys.path.insert(0, os.path.abspath('../../poplar'))
from poplar.distributions import *
import numpy as np
import matplotlib.pyplot as plt

device = "cpu"

# this population is derived from https://doi.org/10.1093/mnras/stad1397
distributions_B = {
    "lgMBH_mass": FixedLimitSchechterFunction([4, 10], device=device),
    "MBHspin" : FixedLimitMassDependentTruncatedBetaDistribution([0.01, 0.99], device=device),
}

class PopulationDistribution:
    def __init__(self, distributions, data) -> None:
        self.distributions = distributions
        self.data = data

    def draw_samples(self, x, size):
        out = {}
        lgMBH_mass_samples = self.distributions['lgMBH_mass'].draw_samples(**x['lgMBH_mass'], size=size)
        MBHspin_samples = self.distributions['MBHspin'].draw_samples(**x['MBHspin'], size=size, lgMBH=lgMBH_mass_samples)
        
        out['lgMBH_mass'] = lgMBH_mass_samples
        out['MBHspin'] = MBHspin_samples
        return out

popdist_B = PopulationDistribution(distributions=distributions_B, data=None)

true_x_B = {
        "lgMBH_mass": {"xc": 5.5, "lam_schechter": 15},
        "MBHspin": {'beta': 2.0, 'lambda_alpha': 1.0}
        }

samples_B = popdist_B.draw_samples(true_x_B, size=10000)

for key in samples_B:
    plt.hist(samples_B[key], bins=50, density=True)
    plt.xlabel('')
    plt.ylabel('Probability Density')
    plt.savefig(f'{key}.png')
    plt.close()