from cosmology import GalaxyStellarMassFunction, MBHMassFunction
import matplotlib.pyplot as plt
# from galaxy import Galaxy

z = 3.148

GSMF = GalaxyStellarMassFunction()

lgMgal = GSMF.sample_gsmf(z_gal=z, size=1)

MBHMF = MBHMassFunction(gsmf=GSMF)

plt.plot(GSMF.get_gsmf(z_gal=z), label='GSMF')
plt.plot(MBHMF.get_mbhmf(z_gal=z), label='MBHMF')
plt.xlabel(r'$\log_{10}(M/M_{\odot})$')
plt.ylabel(r'$\log_{10}(\phi/\mathrm{cMpc}^{-3} \mathrm{dex}^{-1})$')
plt.legend()
plt.tight_layout()
plt.savefig('MF_comparison.pdf', dpi=200)
plt.show()
plt.close()

plt.hist(GSMF.sample_gsmf(z_gal=z, size=10000), bins=50, density=True, alpha=0.5, label='GSMF samples')
plt.hist(MBHMF.sample_mbhmf(z_gal=z, size=10000), bins=50, density=True, alpha=0.5, label='MBHMF samples')
plt.xlabel(r'$\log_{10}(M/M_{\odot})$')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('MF_samples.pdf', dpi=200)
plt.show()
plt.close()