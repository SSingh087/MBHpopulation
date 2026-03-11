import numpy as np 
import matplotlib.pyplot as plt
import matplotlib

from astropy.constants import G, M_sun, c
from astropy import units as u

# ----- Constants (cgs) -----
G_pc3_per_Msun_yr2 = G.to(u.pc**3 / (u.M_sun * u.yr**2)).value # G to units of pc^3 / (M_sun * yr^2)
c_pc_per_year = c.to(u.pc / u.yr).value # pc/year from m/s
G_cgs    = 6.6743e-8                            # cm^3 g^-1 s^-2
c_cgs = c.to(u.cm/u.s).value  # numeric cm/s
Msun_to_grams = 1.98847e33                           # g
kpc_to_cm  = 3.0856775814913673e21                  # kpc to cm
pc_to_cm = 3.0856775814913673e18                   # pc to cm
Msun, Rsun = 1.0, 1.0                              # solar mass
sec_per_year = 365 * 24 * 60 * 60


class Plotting:
    
    matplotlib.rc('font', family='serif', serif=['Computer Modern'], size=15)
    matplotlib.rc('text', usetex=True)

    @staticmethod
    def plot_lgMgal_vs_lgsigma(lgMgal: np.ndarray, sigma_kms: np.ndarray,
                               color='lightcoral', lw=2, label=None):
        plt.figure(figsize=(6.2, 4.6))
        plt.scatter(lgMgal, np.log10(sigma_kms), color=color, lw=lw, label=label)
        plt.xlabel(r'$\log_{10}(M_\star/M_\odot)$')
        plt.ylabel(r'$\log_{10}(\sigma\,[\mathrm{km\,s^{-1}}])$')
        if label:
            plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig('lgMgal_vs_lgsigma.pdf')
        plt.show()
        plt.close()

    @staticmethod
    def plot_lgMgal_vs_lgMBH(lgMgal: np.ndarray, lgMBH: np.ndarray,
                             color='lightcoral', lw=2, label=None):
        plt.figure(figsize=(6.2, 4.6))
        plt.scatter(lgMgal, lgMBH, color=color, lw=lw, label=label)
        plt.xlabel(r'$\log_{10}(M_\star/M_\odot)$')
        plt.ylabel(r'$\log_{10}(M_{\rm BH}/M_\odot)$')
        if label:
            plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig('lgMgal_vs_lgMBH.pdf')
        plt.show()
        plt.close()

    @staticmethod
    def plot_lgsigma_vs_lgMBH(sigma_kms: np.ndarray, lgMBH: np.ndarray,
                              color='lightcoral', lw=2, label=None):
        plt.figure(figsize=(6.2, 4.6))
        plt.scatter(np.log10(sigma_kms), lgMBH, color=color, lw=lw, label=label)
        plt.xlabel(r'$\log_{10}(\sigma\,[\mathrm{km\,s^{-1}}])$')
        plt.ylabel(r'$\log_{10}(M_{\rm BH}/M_\odot)$')
        if label:
            plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig('lgsigma_vs_lgMBH.pdf')
        plt.show()
        plt.close()

    @staticmethod
    def plot_lgMstar_vs_f_NSC(lgMgal: np.ndarray, f_NSC_Hannah: np.ndarray,
                             f_NSC_Neumayer: np.array, color='lightcoral', lw=2, label=None):
        plt.scatter(lgMgal, f_NSC_Hannah, label='Hannah+24')
        plt.scatter(lgMgal, f_NSC_Neumayer, label='Neumayer+20')
        plt.ylabel(r'$f_{\mathrm{NSC}}$')
        plt.xlabel(r'$\log_{10}(M_\star/M_\odot)$')
        plt.legend()
        plt.savefig('lgMstar_vs_f_NSC.pdf')
        plt.show()
        plt.close()