import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import gaussian_kde
import seaborn as sns
import seaborn as sns
from matplotlib import gridspec

import torch

def _to_numpy(x):
    """Convert torch tensor or numpy array to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def ensure_2d(arr):
    """
    Ensures arr has shape (N, Nr).
    If arr is (Nr,), returns (1, Nr).
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def squeeze_if_single(arr):
    """
    If arr has shape (1, Nr), return (Nr,).
    Otherwise return arr unchanged.
    """
    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr[0]
    return arr


def vectorize_r_grid(r_grid, N):
    """
    Ensures r_grid has shape (N, Nr).
    Accepts (Nr,) or (1, Nr).
    """
    r_grid = np.asarray(r_grid)
    if r_grid.ndim == 1:
        return np.broadcast_to(r_grid, (N, len(r_grid)))
    if r_grid.shape[0] == 1 and N > 1:
        return np.broadcast_to(r_grid, (N, r_grid.shape[1]))
    return r_grid

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
        plt.savefig('lgMgal_vs_lgsigma.png')
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
        plt.savefig('/data/wiay/postgrads/shashwat/EMRI_TDE_data/lgMgal_vs_lgMBH.png')
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
        plt.savefig('/data/wiay/postgrads/shashwat/EMRI_TDE_data/lgsigma_vs_lgMBH.png')
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
        plt.savefig('/data/wiay/postgrads/shashwat/EMRI_TDE_data/lgMstar_vs_f_NSC.png')
        plt.show()
        plt.close()

    @staticmethod
    def plot_NSCprofile(NSC_obj, CO_obj, profile_obj, r_grid: np.ndarray, kind='TDE', Ntot=1e5):
        """
        Plots:
            - number density n(r)
            - shell number nr(r)
            - mass density rho(r)
        for all nucleated galaxies in the NSC population.

        r_grid must be shape (N, Nr).
        """

        # ---------------------------------------------------------------
        # Ensure all inputs have correct shapes
        # ---------------------------------------------------------------
        r_grid = ensure_2d(r_grid)          # (N, Nr)
        N, Nr = r_grid.shape

        # ---------------------------------------------------------------
        # Compute fundamental NSC radii
        # ---------------------------------------------------------------
        r_inf = NSC_obj.r_influence(unit='pc')   # (N,)
        r_cap = NSC_obj.r_capture(unit='pc')     # (N,)
        r_tid = NSC_obj.r_tidal(unit='pc')       # (N,)

        # ---------------------------------------------------------------
        # Compute profiles
        # ---------------------------------------------------------------
        n_star   = profile_obj.dehnen_number_density(r_grid, kind=kind)
        nr_star  = profile_obj.radial_number_distribution(r_grid, kind=kind)
        rho_star = profile_obj.mass_density(r_grid=r_grid, unit='Msun/pc^3')

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
        ax[0].set_title("$n(r)$")

        # ------------------------------
        # Panel 2: n_r(r)
        # ------------------------------
        for i in range(N):
            ax[1].loglog(r_grid[i], nr_star[i], label=f'Galaxy {i}')
        ax[1].set_xlabel('r [pc]')
        ax[1].set_ylabel(r'$4\pi r^2 n(r)$ [1/pc]')
        for i in range(N):
            ax[1].axvline(r_inf[i], linestyle='--', color='gray', alpha=0.5)
        ax[1].set_title("$n_r(r)$")

        # ------------------------------
        # Panel 3: Mass Density rho(r)
        # ------------------------------
        for i in range(N):
            ax[2].loglog(r_grid[i], rho_star[i], label=f'$10^{{{NSC_obj.gal.lgMgal[i]:.2f}}}~M_\odot$')
            ax[2].scatter(r_inf[i], profile_obj.mass_density_at_rinfl(kvir=1.0, unit='Msun/pc^3')[i], color='red', marker='x', s=100, label=f'$\\rho(r_{{inf}})$ Galaxy {i}' if i == 0 else None)
        ax[2].set_xlabel('r [pc]')
        ax[2].set_ylabel(r'$\rho(r)$ [M$_\odot$/pc$^3$]')
        for i in range(N):
            ax[2].axvline(r_inf[i], linestyle='--', color='gray', alpha=0.5)
        ax[2].set_title(r"$\rho(r)$")

        # ---------------------------------------------------------------
        # Legend and saving
        # ---------------------------------------------------------------
        ax[2].legend(loc='upper right', fontsize=8)

        filename = f"/data/wiay/postgrads/shashwat/EMRI_TDE_data/NSC_profile_{NSC_obj.gal.lgMgal}.png"

        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.show()

    @staticmethod
    def MBHmass_vs_spin(mbhmass, spin, z, loc, cmap="plasma"):   
        fig = plt.figure(figsize=(9, 9))
        gs = gridspec.GridSpec(
            4, 4,
            figure=fig,
            wspace=0.0,
            hspace=0.0
        )

        ax_main  = fig.add_subplot(gs[1:4, 0:3])
        ax_top   = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

        # --------------------------------------------------
        # MAIN PANEL: SCATTER
        # --------------------------------------------------
        sc = ax_main.scatter(
            mbhmass,
            spin,
            c=z,
            cmap="viridis",
            s=18,
            alpha=0.8,
            linewidths=0,
            zorder=2
        )

        # KDE LINE CONTOURS ONLY (NO FILL)
        sns.kdeplot(
            x=mbhmass,
            y=spin,
            ax=ax_main,
            levels=5,
            cmap=cmap,
            linewidths=1.1,
            fill=True,
            zorder=3
        )

        ax_main.set_xlabel(r'$\log_{10}(M_{\rm BH}/M_\odot)$')
        ax_main.set_ylabel(r'$a_{\rm BH}$')

        # --------------------------------------------------
        # TOP MARGINAL KDE (FILLED, BOXED)
        # --------------------------------------------------
        sns.kdeplot(
            x=mbhmass,
            ax=ax_top,
            fill=True,
            color="#8fb1ff",
            linewidth=1.0,
            alpha=0.85
        )

        # --------------------------------------------------
        # RIGHT MARGINAL KDE (FILLED, BOXED)
        # --------------------------------------------------
        sns.kdeplot(
            y=spin,
            ax=ax_right,
            fill=True,
            color="#ffb199",
            linewidth=1.0,
            alpha=0.85
        )

        # --------------------------------------------------
        # MARGINAL AXES: NO TICKS, KEEP BOXES
        # --------------------------------------------------
        for ax in [ax_top, ax_right]:
            ax.tick_params(
                left=False, bottom=False,
                labelleft=False, labelbottom=False
            )
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.9)


        cax = fig.add_axes([0.16, 0.935, 0.68, 0.022])
        cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cbar.set_label("")
        cbar.ax.tick_params(labelsize=9)

        # DETECTABILITY MARKERS
        z_ZTF  = 1.0
        z_LSST = 2.0
        z_LISA = 8.0

        for z_val, label, colour in [
            (z_ZTF,  "ZTF",  "blue"),
            (z_LSST, "LSST", "orange"),
            (z_LISA, "LISA", "red")
        ]:
            cbar.ax.axvline(
                z_val,
                color=colour,
                linestyle="--",
                linewidth=2
            )
            cbar.ax.text(
                z_val,
                1.4,
                label,
                ha="center",
                va="bottom",
                fontsize=10,
                color=colour
            )

        # --------------------------------------------------
        # FINAL ADJUSTMENTS
        # --------------------------------------------------
        sns.despine(ax=ax_main)
        plt.subplots_adjust(top=0.92)

        plt.savefig(loc, dpi=300)
        plt.close()

