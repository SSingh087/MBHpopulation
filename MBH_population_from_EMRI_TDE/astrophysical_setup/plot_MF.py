import sys, os
sys.path.insert(0, os.path.abspath('../'))

from cosmology import GalaxyStellarMassFunction, MBHMassFunction
import numpy as np
from utils import Plotting


z_grid = np.linspace(0.01, 10.0, 5)

GSMF = GalaxyStellarMassFunction()
MBHMF = MBHMassFunction(gsmf=GSMF)

lgMgal, phi_gsmf = GSMF.get_gsmf(z_gal=z_grid, n_points_mass=1000)
logMBH, phi_mbhmf = MBHMF.get_mbhmf(z_gal=z_grid, n_points_mass=1000)

Plotting.plot_mass_functions(lgMgal_grid=lgMgal, phi_gsmf_grid=phi_gsmf, lgMBH_grid=logMBH, phi_mbhmf_grid=phi_mbhmf, z_grid=z_grid, filename='./data/astrophysical_data/MF_comparison.png')
