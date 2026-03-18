from cosmology import GalaxyStellarMassFunction, MBHMassFunction
import matplotlib.pyplot as plt
# from galaxy import Galaxy

z = 3.148

GSMF = GalaxyStellarMassFunction()

lgMgal = GSMF.sample_gsmf(z_gal=z, size=1)

MBHMF = MBHMassFunction(gsmf=GSMF)

# plt.plot(GSMF.get_gsmf(z_gal=z), label='GSMF')
# plt.plot(MBHMF.get_mbhmf(z_gal=z), label='MBHMF')
# plt.hist(GSMF.sample_gsmf(z_gal=z, size=10000), bins=50, density=True, alpha=0.5, label='GSMF samples')
# plt.hist(MBHMF.sample_mbhmf(z_gal=z, size=10000), bins=50, density=True, alpha=0.5, label='MBHMF samples')

plt.legend()
plt.show()
