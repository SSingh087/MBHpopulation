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


    @staticmethod
    def plot_NSCprofile(NSC_obj, r: np.ndarray, component_masses: np.ndarray, kind='TDE', Ntot=1e5):
        
        r_inf = NSC_obj.influence_radius(unit='pc')
        r_cap = NSC_obj.capture_radius(unit='pc')
        r_tid = NSC_obj.tidal_radius_star(unit='pc')

        n_star = NSC_obj.dehnen_number_density(r, Ntot=Ntot, kind=kind, unit='1/pc^3')
        nr_star = NSC_obj.radial_number_distribution(r, Ntot=Ntot, kind=kind, unit='1/pc')
        rho_star = NSC_obj.mass_density(r=r, Ntot=Ntot, component_masses=component_masses, kind=kind, unit='Msun/pc^3')

        fig, ax = plt.subplots(1, 3, figsize=(12, 5), sharex=True)
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

        plt.savefig(f"{NSC_obj.lgMgal}_properties.pdf", dpi=200)
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

