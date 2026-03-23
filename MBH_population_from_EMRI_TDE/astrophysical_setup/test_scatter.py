import numpy as np
import matplotlib.pyplot as plt
from galaxy import Galaxy

Mstar = np.logspace(6, 12, 1000)     # Msun, linear
lgMgal = np.log10(Mstar)
lgMBH = []
sigma = []

z_obs = 0.5
m = 2.0

for i in range(len(lgMgal)):
    gal = Galaxy.check_nucleation(lgMgal[i], z_obs)  # returns None if no nucleation
    if gal is None:
        lgMBH.append(np.nan)
        sigma.append(np.nan)
    else:
        lgMBH.append(gal.lgMBH_mass())
        sigma.append(gal.sigma(unit='km/s'))

Plotting.plot_lgMgal_vs_lgMBH(lgMgal, lgMBH)
Plotting.plot_lgMgal_vs_lgsigma(lgMgal, sigma)
Plotting.plot_lgsigma_vs_lgMBH(sigma, lgMBH)
