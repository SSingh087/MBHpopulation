import sys, os
sys.path.insert(0, os.path.abspath('../'))

import numpy as np
from galaxy import Galaxy
from utils import Plotting

Mstar = np.logspace(6, 12, 1000)     # Msun
lgMgal = np.log10(Mstar)
lgMBH = []
sigma = []

z_obs = 0.5

gal = Galaxy(z_gal=z_obs, lgMgal=lgMgal)

Plotting.plot_lgMgal_vs_lgMBH(lgMgal, gal.lgMBH_mass, color='lightcoral', lw=2, filename='./data/astrophysical_data/lgMgal_vs_lgMBH.png')

Plotting.plot_lgMgal_vs_lgsigma(lgMgal, gal.sigma(unit='km/s'), color='lightcoral', lw=2, filename='./data/astrophysical_data/lgMgal_vs_lgsigma.png')

Plotting.plot_lgsigma_vs_lgMBH(gal.sigma(unit='km/s'), gal.lgMBH_mass, color='lightcoral', lw=2, filename='./data/astrophysical_data/lgsigma_vs_lgMBH.png')