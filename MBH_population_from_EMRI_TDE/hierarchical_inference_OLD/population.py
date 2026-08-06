#!/usr/bin/env python

import os, sys
sys.path.insert(0, os.path.abspath('../../poplar'))
from poplar.distributions import *
from numpy import pi, arccos

device = "cpu"

# this population is derived from https://doi.org/10.1093/mnras/stad1397
distributions_B = {
    "log10_M": FixedLimitsPowerLaw([6, 10], device=device),
    "log10_mu": FixedLimitsPowerLaw([1, 2], device=device),
    "a" : FixedLimitsTruncatedGaussian([0.1, 0.7], device=device),
}

class PopulationDistribution:
    def __init__(self, distributions, data) -> None:
        self.distributions = distributions
        self.data = data

    def draw_samples(self, x, size):
        out = {}
        for key in self.distributions.keys():
            out[key] = self.distributions[key].draw_samples(**x[key], size=size)
        return out

popdist_B = PopulationDistribution(distributions=distributions_B, data=None)

true_x_B = {
        "log10_M": {"lam": -1.5},
        "log10_mu": {"lam": -3.0},
        "a": {"mu": 0.6, "sigma": 0.01},
        }