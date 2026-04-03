import numpy as np 
import matplotlib.pyplot as plt
import matplotlib

from astropy.constants import G, M_sun, c
from astropy import units as u

from scipy.stats import gaussian_kde

import torch

def _to_numpy(x):
    """Convert torch tensor or numpy array to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

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
    def plot_NSCprofile(NSC_obj, profile_obj, r_grid: np.ndarray,
                        component_masses: np.ndarray, kind='TDE', Ntot=1e5):
        """
        Plots:
            - number density n(r)
            - shell number nr(r)
            - mass density rho(r)
        for all nucleated galaxies in the NSC population.

        r_grid must be shape (N, Nr).
        """
        from density import ensure_2d, squeeze_if_single, vectorize_r_grid

        # ---------------------------------------------------------------
        # Ensure all inputs have correct shapes
        # ---------------------------------------------------------------
        r_grid = ensure_2d(r_grid)          # (N, Nr)
        N, Nr = r_grid.shape

        # Broadcast Ntot correctly
        if np.isscalar(Ntot):
            Ntot = Ntot * np.ones(N)
        Ntot = np.asarray(Ntot).reshape(N)

        # component_masses: allow 1D or (N, species)
        comp_mass = np.asarray(component_masses)
        if comp_mass.ndim == 1:
            comp_mass = np.broadcast_to(comp_mass, (N, comp_mass.size))

        # ---------------------------------------------------------------
        # Compute fundamental NSC radii
        # ---------------------------------------------------------------
        r_inf = NSC_obj.r_influence(unit='pc')   # (N,)
        r_cap = NSC_obj.r_capture(unit='pc')     # (N,)
        r_tid = NSC_obj.r_tidal(unit='pc')       # (N,)

        # ---------------------------------------------------------------
        # Compute profiles
        # ---------------------------------------------------------------
        n_star   = profile_obj.dehnen_number_density(r_grid, Ntot=Ntot, kind=kind)
        nr_star  = profile_obj.radial_number_distribution(r_grid, Ntot=Ntot, kind=kind)
        rho_star = profile_obj.mass_density(r_grid=r_grid, Ntot=Ntot,
                                            component_masses=comp_mass,
                                            kind=kind, unit='Msun/pc^3')

        # ---------------------------------------------------------------
        # Plotting
        # ---------------------------------------------------------------
        fig, ax = plt.subplots(1, 3, figsize=(17, 5), sharex=False)

        # ------------------------------
        # Panel 1: n(r)
        # ------------------------------
        for i in range(N):
            ax[0].loglog(r_grid[i], n_star[i], label=f'Galaxy {i}')
        ax[0].set_xlabel('r [pc]')
        ax[0].set_ylabel(r'$n(r)$ [1/pc$^3$]')
        for i in range(N):
            ax[0].axvline(r_inf[i], linestyle='--', color='gray', alpha=0.5)
        ax[0].set_title("Number Density $n(r)$")

        # ------------------------------
        # Panel 2: n_r(r)
        # ------------------------------
        for i in range(N):
            ax[1].loglog(r_grid[i], nr_star[i], label=f'Galaxy {i}')
        ax[1].set_xlabel('r [pc]')
        ax[1].set_ylabel(r'$4\pi r^2 n(r)$ [1/pc]')
        for i in range(N):
            ax[1].axvline(r_inf[i], linestyle='--', color='gray', alpha=0.5)
        ax[1].set_title("Shell Number Distribution $n_r(r)$")

        # ------------------------------
        # Panel 3: Mass Density rho(r)
        # ------------------------------
        for i in range(N):
            ax[2].loglog(r_grid[i], rho_star[i], label=f'Galaxy {i}')
        ax[2].set_xlabel('r [pc]')
        ax[2].set_ylabel(r'$\rho(r)$ [M$_\odot$/pc$^3$]')
        for i in range(N):
            ax[2].axvline(r_inf[i], linestyle='--', color='gray', alpha=0.5)
        ax[2].set_title("Mass Density $\rho(r)$")

        # ---------------------------------------------------------------
        # Legend and saving
        # ---------------------------------------------------------------
        ax[2].legend(loc='upper right', fontsize=8)

        filename = f"NSC_profile_{NSC_obj.gal.lgMgal}.pdf"

        plt.tight_layout()
        plt.savefig(filename, dpi=200)
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

    @staticmethod
    def plot_allowed_region(z, theta, ax=None, bins=100):
        None

    @staticmethod
    def plot_marginal_theta(theta, pdf, bins=40, ax=None):
        """
        Plot weighted 1D marginal distribution over log10(M_BH).
        """
        theta = _to_numpy(theta)
        pdf   = _to_numpy(pdf)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6,4))

        ax.hist(theta, bins=bins, weights=pdf, density=True, alpha=0.75)
        ax.set_xlabel(r'$\log_{10}(M_{\rm BH}/M_\odot)$')
        ax.set_ylabel(r'$p(\log M_{\rm BH})$')
        ax.set_title("Marginal MBH Distribution")

        return ax

    @staticmethod
    def plot_marginal_z(z, pdf, bins=40, ax=None):
        """
        Plot weighted 1D marginal distribution over redshift.
        """
        z = _to_numpy(z)
        pdf = _to_numpy(pdf)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6,4))

        ax.hist(z, bins=bins, weights=pdf, density=True, alpha=0.75)
        ax.set_xlabel(r'$z$')
        ax.set_ylabel(r'$p(z)$')
        ax.set_title("Redshift Marginal Distribution")

        return ax

    @staticmethod
    def plot_joint_2D(z, theta, pdf, bins=40, ax=None):
        """
        Weighted 2D joint distribution p(z, logMBH)
        """
        z = _to_numpy(z)
        theta = _to_numpy(theta)
        pdf = _to_numpy(pdf)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6,5))

        h = ax.hist2d(z, theta, bins=bins, weights=pdf,
                      density=True, cmap='viridis')
        plt.colorbar(h[3], ax=ax, label=r'$p(z, \log M_{\rm BH})$')

        ax.set_xlabel(r'$z$')
        ax.set_ylabel(r'$\log_{10}(M_{\rm BH}/M_\odot)$')
        ax.set_title("Joint Distribution p(z, log M_BH)")

        return ax

    @staticmethod
    def plot_joint_2D_smooth(z, theta, pdf, bins=100, ax=None):
        """
        Kernel-smoothed 2D density estimator for nicer publication-quality plots.
        """
        from scipy.stats import gaussian_kde

        z = _to_numpy(z)
        theta = _to_numpy(theta)
        pdf = _to_numpy(pdf)
        # Weighted KDE
        kde = gaussian_kde(np.vstack([z, theta]), weights=pdf)

        # Create grid for contour plot
        z_lin = np.linspace(z.min(), z.max(), bins)
        t_lin = np.linspace(theta.min(), theta.max(), bins)
        Zg, Tg = np.meshgrid(z_lin, t_lin)

        pos = np.vstack([Zg.ravel(), Tg.ravel()])
        density = kde(pos).reshape(Zg.shape)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6,5))
        plot_allowed_region(z, theta, ax=ax)

        cf = ax.contourf(Zg, Tg, density, levels=40, cmap="inferno")
        plt.colorbar(cf, ax=ax, label="Density")

        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"$\log_{10}(M_{\rm MBH}/M_\odot)$")
        ax.set_title("Smoothed Joint PDF (KDE)")

        return ax

    @staticmethod
    def plot_joint_with_marginals(z, theta, pdf, theta_label, bins=40, smooth=False, cmap="magma"):
        """
        Produce a corner-style plot with:
        - Joint 2D histogram or KDE
        - 1D marginals (histograms or KDE)
        - External colorbar
        - Correct shared axes
        """

        # Convert tensors → numpy
        z = _to_numpy(z)
        theta = _to_numpy(theta)
        pdf = _to_numpy(pdf)
        pdf = pdf / pdf.sum()

        fig = plt.figure(figsize=(9, 9))
        gs  = fig.add_gridspec(4, 4, wspace=0.05, hspace=0.05)

        ax_joint = fig.add_subplot(gs[1:, 0:3])
        ax_top   = fig.add_subplot(gs[0, 0:3], sharex=ax_joint)
        ax_right = fig.add_subplot(gs[1:, 3],  sharey=ax_joint)

        # ---------------------------------------------------
        # Histogram mode
        # ---------------------------------------------------
        if not smooth:
            h = ax_joint.hist2d(theta, z, bins=bins, weights=pdf,
                                density=True, cmap=cmap)

            cax = fig.add_axes([0.92, 0.15, 0.02, 0.6])
            fig.colorbar(h[3], cax=cax, label=r"$p(z, " + theta_label + ")$")

            ax_top.hist(theta, bins=bins, weights=pdf, density=True,
                        color='C0', alpha=0.7)
            ax_right.hist(z, bins=bins, weights=pdf, density=True,
                        orientation='horizontal', color='C1', alpha=0.7)

        # ---------------------------------------------------
        # KDE mode
        # ---------------------------------------------------
        else:
            kde_2d = gaussian_kde(np.vstack([theta, z]), weights=pdf)

            th_lin = np.linspace(theta.min(), theta.max(), bins)
            z_lin  = np.linspace(z.min(), z.max(), bins)
            TH, ZH = np.meshgrid(th_lin, z_lin)

            dens_2d = kde_2d(np.vstack([TH.ravel(), ZH.ravel()])).reshape(TH.shape)

            cf = ax_joint.contourf(TH, ZH, dens_2d, 40, cmap=cmap)

            cax = fig.add_axes([0.92, 0.15, 0.02, 0.6])
            fig.colorbar(cf, cax=cax, label="KDE Density")

            # 1D KDE marginals
            kde_theta = gaussian_kde(theta, weights=pdf)
            kde_z     = gaussian_kde(z, weights=pdf)

            d_theta = kde_theta(th_lin)
            d_z     = kde_z(z_lin)

            d_theta /= np.trapezoid(d_theta, th_lin)
            d_z     /= np.trapezoid(d_z, z_lin)

            ax_top.plot(th_lin, d_theta, color="C0")
            ax_top.fill_between(th_lin, d_theta, color="C0", alpha=0.3)

            ax_right.plot(d_z, z_lin, color="C1")
            ax_right.fill_betweenx(z_lin, d_z, color="C1", alpha=0.3)

        # Labels
        ax_joint.set_xlabel(r"$" + theta_label + "$")
        ax_joint.set_ylabel(r"$z$")
        ax_top.set_ylabel(r"$p(" + theta_label + ")$")
        ax_right.set_xlabel(r"$p(z)$")

        ax_top.tick_params(axis="x", labelbottom=False)
        ax_right.tick_params(axis="y", labelleft=False)

        return fig, ax_joint, ax_top, ax_right
