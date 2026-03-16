import numpy as np 
import matplotlib.pyplot as plt
import matplotlib

from astropy.constants import G, M_sun, c
from astropy import units as u

# ----- Constants (cgs) -----

class Distributions:
    def __init__(self, x, pdf):
        self.x = x
        self.pdf_unnormalized = pdf 
        self.pdf = self.pdf_unnormalized / np.trapezoid(self.pdf_unnormalized, self.x)
        self.cdf = self.cdf_from_pdf()

    def cdf_from_pdf(self):
        
        cdf = np.empty_like(self.pdf)
        cdf[0] = 0.0
        cdf[1:] = np.cumsum(0.5 * (self.pdf[1:] + self.pdf[:-1]) * np.diff(self.x))
        # Numerical guard : enforce last value to be exactly 1
        cdf = np.clip(cdf, 0.0, 1.0)
        return cdf

    def draw_samples(self, size=1000):
        u = np.random.uniform(0.0, 1.0, size=size)
        u = np.clip(u, self.cdf.min(), self.cdf.max())
        return np.interp(u, self.cdf, self.x)

    def get_samples(self, size=1000):
        cdf = self.cdf_from_pdf()
        samples = self.draw_samples(size=size)
        return samples


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


    @staticmethod
    def plot_NSCprofile(NSC_obj, profile_obj, r: np.ndarray, component_masses: np.ndarray, kind='TDE', Ntot=1e5):
        
        r_inf = NSC_obj.r_influence(unit='pc')
        r_cap = NSC_obj.r_capture(unit='pc')
        r_tid = NSC_obj.r_tidal(unit='pc')

        n_star = profile_obj.dehnen_number_density(r, Ntot=Ntot, kind=kind, unit='1/pc^3')
        nr_star = profile_obj.radial_number_distribution(r, Ntot=Ntot, kind=kind, unit='1/pc')
        rho_star = profile_obj.mass_density(r=r, Ntot=Ntot, component_masses=component_masses, kind=kind, unit='Msun/pc^3')

        fig, ax = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
        ax[0].loglog(r, n_star, label='$n_i^\mathrm{EMRI}(r)$')
        ax[0].set_xlabel('r [pc]')
        ax[0].set_ylabel(r'$1/\mathrm{pc}^3$')
        ax[0].vlines(r_inf, np.min(n_star), np.max(n_star), label='$r_\mathrm{infl.}$', linestyle='--', color='black')
        ax[0].legend()

        ax[1].loglog(r, nr_star, label='$n_r^\mathrm{EMRI}(r)$')
        ax[1].set_xlabel('r [pc]')
        ax[1].set_ylabel(r'$1/\mathrm{pc}$')
        ax[1].vlines(r_inf, np.min(nr_star), np.max(nr_star), label='$r_\mathrm{infl.}$', linestyle='--', color='black')
        ax[1].legend()  

        ax[2].loglog(r, rho_star, label='$\\rho^\mathrm{EMRI}(r)$')
        ax[2].set_xlabel('r [pc]')
        ax[2].set_ylabel('$M_\odot/\mathrm{pc}^3$')
        ax[2].vlines(r_inf, np.min(rho_star), np.max(rho_star), label='$r_\mathrm{infl.}$', linestyle='--', color='black')
        ax[2].legend()

        plt.savefig(f"{NSC_obj.gal.lgMgal}_properties.pdf", dpi=200)
        plt.show()

    @staticmethod
    def plot_rate_evolution(tau: np.ndarray, rate_EMRI: np.ndarray, rate_TDE: np.ndarray):
        plt.plot(tau, rate_EMRI, label='EMRI')
        plt.plot(tau, rate_TDE, label='TDE')
        plt.legend()
        plt.xlabel(r'$\tau=t/t_{\mathrm{EMRI}}$')
        plt.ylabel(r'$\Gamma_k/\hat{\Gamma}_k$')
        plt.tight_layout()
        plt.savefig('rate_evolution.pdf', dpi=200)
        plt.show()