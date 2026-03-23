from cosmology import GalaxyStellarMassFunction, MBHMassFunction
import matplotlib.pyplot as plt
import numpy as np

z_grid = np.random.uniform(0.01, 10.0, 10)

GSMF = GalaxyStellarMassFunction()
MBHMF = MBHMassFunction(gsmf=GSMF)

lgMgal, phi_batch = GSMF.get_gsmf(z_gal=z_grid, n_points_mass=100)
lgMgal_samples = GSMF.sample_gsmf(z_gal=z_grid, size=1000)


logMBH_grid, dlogMBH_dlogMgal = MBHMF.get_mbhmf(z_gal=z_grid)

plt.plot(lgMgal, phi_batch.T)
plt.plot(logMBH_grid, dlogMBH_dlogMgal.T, linestyle='--')
plt.legend([f"$z = {z_value:2.2f}$" for z_value in z_grid])
plt.xlabel(r'$\log_{10}(M/M_{\odot})$')
plt.ylabel(r'$\log_{10}(\phi/\mathrm{cMpc}^{-3} \mathrm{dex}^{-1})$')
plt.tight_layout()
plt.savefig('MF_comparison.pdf', dpi=200)
plt.show()
plt.close()