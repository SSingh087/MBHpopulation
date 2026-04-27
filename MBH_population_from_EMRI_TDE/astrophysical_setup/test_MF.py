import matplotlib
from cosmology import GalaxyStellarMassFunction, MBHMassFunction
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rc('font', family='serif', serif=['Computer Modern'], size=15)
matplotlib.rc('text', usetex=True)

import matplotlib.colors as mcolors

z_grid = np.linspace(0.01, 10.0, 5)


GSMF = GalaxyStellarMassFunction()
MBHMF = MBHMassFunction(gsmf=GSMF)

lgMgal, phi_batch = GSMF.get_gsmf(z_gal=z_grid, n_points_mass=1000)
lgMgal_samples = GSMF.sample_gsmf(z_gal=z_grid, size=1000)


logMBH_grid, dlogMBH_dlogMgal = MBHMF.get_mbhmf(z_gal=z_grid)


cmap = plt.get_cmap('plasma')
colors = cmap(np.linspace(0, 1, len(z_grid)))

for i in range(len(z_grid)):
    plt.plot(lgMgal, phi_batch[i], color=colors[i])
# plt.plot(logMBH_grid, dlogMBH_dlogMgal.T, linestyle='--')
plt.legend([f"$z = {z_value:2.2f}$" for z_value in z_grid], frameon=False)
plt.xlabel(r'$\log_{10}(M_\mathrm{gal}/M_{\odot})$')
plt.ylabel(r'$\log_{10}(\phi/\mathrm{cMpc}^{-3} \mathrm{dex}^{-1})$')
plt.xlim(6.5, 12)
plt.ylim(-5, 1)
plt.tight_layout()
plt.savefig('/data/wiay/postgrads/shashwat/EMRI_TDE_data/astrophysical_data/MF_comparison.png', dpi=300)
plt.show()
plt.close()