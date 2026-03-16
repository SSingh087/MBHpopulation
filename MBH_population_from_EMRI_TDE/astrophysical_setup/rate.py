import numpy as np

from config import (MBH_A, MBH_B, MBH_sigma0, MBH_scatter)
from nsc import NSC
from scipy.stats import beta, norm

# from utils import Plotting, Distributions

class RateModel:
    def __init__(self, nsc: NSC, tau_grid: np.ndarray):
        self.nsc = nsc
        self.tau_grid = tau_grid

    def peak_EMRI_rate(self, A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter):
        """
        $\hat{\Gamma}_\mathrm{EMRI}
        """
        sigma = self.nsc.gal.sigma(unit='km/s')
        lgMBH = self.nsc.gal.lgMBH_mass(A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter)
        MBH = 10.0**lgMBH
        a, b, c = 6.2e-6, -0.25, 3.09 

        return a * (MBH)**b + (sigma/sigma_0)**c 

    def time_to_peak_EMRI_rate(self, A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter):
        """
        $t_\mathrm{EMRI}
        """
        sigma = self.nsc.gal.sigma(unit='km/s')
        lgMBH = self.nsc.gal.lgMBH_mass(A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter)
        MBH = 10.0**lgMBH
        a, b, c = 6.4e-9, 1.29, -2.97
        
        return a * (MBH)**b + (sigma/sigma_0)**c 

    def universal_EMRI_rate(self):
        # this should be replaced by the values that Luca will provide, but for now we can use similar looking distributions to test the code
        # From arXiv:2205.06277v1 Fig 6 
        a, b = 2.0, 5.0
        return beta.pdf(self.tau_grid, a, b)


    def universal_TDE_rate(self):
        # this should be replaced by the values that Luca will provide, but for now we can use similar looking distributions to test the code
        # From arXiv:2205.06277v1 Fig 6 
        mu, sigma = 1e-7, 0.05
        a, b = 2.0, 5.0
        return norm.pdf(self.tau_grid, mu, sigma) + beta.pdf(self.tau_grid, a, b)

# tau = np.linspace(0, 1, 1000)
# rate = RateModel(tau)

# # Plotting.plot_rate_evolution(tau, rate.universal_EMRI_rate(), rate.universal_TDE_rate())

# pdf = rate.universal_EMRI_rate()
# samples = Distributions(tau, pdf).get_samples(size=1000)

# import matplotlib.pyplot as plt
# plt.hist(samples, bins=50, density=True)
# plt.xlabel(r'$\tau$')
# plt.ylabel('samples')
# plt.show()time_to_peak_EMRI_rate