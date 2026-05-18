import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import gaussian_kde
import seaborn as sns
from matplotlib import gridspec

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
        plt.savefig('./data/lgMgal_vs_lgMBH.png')
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
        plt.savefig('./data/lgsigma_vs_lgMBH.png')
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
        plt.savefig('./data/lgMstar_vs_f_NSC.png')
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

        filename = f"./data/NSC_profile_{NSC_obj.gal.lgMgal}.png"

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

    @staticmethod
    def plot_event_rates(z_gal, lgMBH, generated_EMRIs, generated_TDEs, T_obs, filename):
        import matplotlib.colors as mcolors

        fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

        # --- Color normalization across full z range ---
        norm = mcolors.Normalize(vmin=np.min(z_gal), vmax=np.max(z_gal))
        colors = z_gal

        # --- Top: EMRIs ---
        sc = axes[0].scatter(lgMBH, generated_EMRIs, c=colors, cmap='plasma', norm=norm,
                            marker='o', alpha=0.9, s=20)

        axes[0].set_title(f'EMRIs = {np.sum(generated_EMRIs)}')
        axes[0].set_yscale('log')
        axes[0].set_ylabel(f'Number of events in {T_obs} yrs')

        axes[1].scatter(lgMBH, generated_TDEs, c=colors, cmap='plasma', norm=norm,
                        marker='d', alpha=0.9, s=20)

        axes[1].set_title(f'TDEs = {np.sum(generated_TDEs)}')
        axes[1].set_yscale('log')
        axes[1].set_xlabel(r'$\log_{10}(M_{\mathrm{MBH}} / M_\odot)$')
        axes[1].set_ylabel(f'Number of events in {T_obs} yrs')

        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
        fig.colorbar(sc, cax=cbar_ax, label='Redshift')

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(filename, dpi=300)
        plt.close()

    @staticmethod
    def plot_event_rates_hist_2d(lgMBH, z_gal, generated_EMRIs, generated_TDEs, filename):
        fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

        mass_bins = np.linspace(np.min(lgMBH), 
                                np.max(lgMBH), 100)
        z_bins = np.linspace(np.min(z_gal), 
                            np.max(z_gal), 100)

        # --- Top: EMRIs ---
        H_emri, xedges, yedges = np.histogram2d(lgMBH, z_gal, bins=[mass_bins, z_bins], weights=generated_EMRIs)

        # Optionally smooth
        # H_emri = gaussian_filter(H_emri, sigma=1.5)

        X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
        pcm = axes[0].pcolormesh(X, Y, H_emri, cmap='plasma', shading='auto')

        axes[0].set_title(f'EMRIs = {np.sum(generated_EMRIs)}')
        axes[0].set_ylabel('Redshift (z)')

        # Colorbar for EMRIs
        cbar_emri = fig.colorbar(pcm, ax=axes[0], label='Number of EMRI events')

        # --- Bottom: TDEs ---
        H_tde, _, _ = np.histogram2d(lgMBH, z_gal, bins=[mass_bins, z_bins], weights=generated_TDEs)
        # H_tde = gaussian_filter(H_tde, sigma=1.5)

        pcm2 = axes[1].pcolormesh(X, Y, H_tde, cmap='plasma', shading='auto')

        axes[1].set_title(f'TDEs = {np.sum(generated_TDEs)}')
        axes[1].set_xlabel(r'$\log_{10}(M_{\mathrm{MBH}} / M_\odot)$')
        axes[1].set_ylabel('Redshift (z)')

        # Colorbar for TDEs
        cbar_tde = fig.colorbar(pcm2, ax=axes[1], label='Number of TDE events')

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    @staticmethod
    def plot_polar_ra_dec(ra_deg, dec_deg, z_gal, title, filename, cmap='plasma'):
        """Create a single figure with two stacked polar subplots:
        - Top: RA (deg) vs redshift
        - Bottom: Dec (shifted) vs redshift

        Parameters:
            ra_deg: array-like, right ascension in degrees
            dec_deg: array-like, declination in degrees (range [-90, 90])
            z: array-like, redshift values (used for radial coordinate and coloring)
            title: str, title for the top (RA) subplot
            filename: str, path to save the combined figure
            cmap: str or Colormap, matplotlib colormap for points
        """
        # Prepare figure with two polar subplots stacked vertically
        fig, axes = plt.subplots(2, 1, figsize=(8, 10), subplot_kw={'projection': 'polar'})
        fig.patch.set_facecolor('white')

        colors = z_gal

        # Top: RA vs z
        ax = axes[0]
        ax.set_facecolor('#f9f9f9')
        sc1 = ax.scatter(np.deg2rad(ra_deg), z_gal, c=colors, cmap=cmap, marker='o', s=20, alpha=0.6)
        ax.set_rlabel_position(240)
        ax.grid(True, color='gray', linestyle='--', alpha=0.3)
        ax.set_title(title, fontsize=14)

        # Bottom: Dec vs z
        ax2 = axes[1]
        ax2.set_facecolor('#f9f9f9')
        # Shift Dec from [-90,90] -> [0,180] for polar plotting
        dec_shifted = np.asarray(dec_deg) + 90.0
        sc2 = ax2.scatter(np.deg2rad(dec_shifted), z_gal, c=colors, cmap=cmap, marker='o', s=20, alpha=0.6)
        ax2.set_rlabel_position(240)
        ax2.grid(True, color='gray', linestyle='--', alpha=0.3)
        ax2.set_title('Galaxy Distribution: Dec vs Redshift', fontsize=14)

        # Shared colorbar on the right
        cbar = fig.colorbar(sc1, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
        cbar.set_label('Redshift', fontsize=12)

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close(fig)

    @staticmethod
    def plot_3D_galaxy_distribution(x, y, z, z_filtered, filename, cmap='plasma'):
        fig = plt.figure(figsize=(10,8), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(x, y, z, c=z_filtered, s=15, cmap=cmap)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
        cbar.set_label('Redshift', fontsize=12)
        ax.set_xlabel('X [$z$]')
        ax.set_ylabel('Y [$z$]')
        ax.set_zlabel('Z [$z$]')
        ax.set_title('3D Galaxy Distribution (Concentric Shells)', fontsize=14)
        ax.grid(False)
        ax.set_box_aspect([1,1,1])
        plt.savefig(filename, dpi=300)
        plt.close()

    @staticmethod
    def mollweide_sky_map(ra_deg, dec_deg, z_gal, zmin_slice, zmax_slice, filename):

        mask = (z_gal > zmin_slice) & (z_gal < zmax_slice)

        ra_rad  = np.deg2rad(ra_deg[mask])
        dec_rad = np.deg2rad(dec_deg[mask])
        ra_wrap = (ra_rad + np.pi) % (2*np.pi) - np.pi

        fig = plt.figure(figsize=(8,4.5))
        ax = fig.add_subplot(111, projection="mollweide")
        ax.scatter(-ra_wrap, dec_rad, s=5, alpha=0.8, rasterized=True)  # minus flips RA like sky maps
        ax.grid(True, alpha=0.3)
        ax.set_title(f"${zmin_slice} \leq z \leq {zmax_slice}$")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    @staticmethod
    def angular_density_slice(zmin_slice, zmax_slice, ra_deg, dec_deg, z_gal, filename):
        zmin_slice, zmax_slice = 0.5, 1.0
        mask = (z_gal > zmin_slice) & (z_gal < zmax_slice)

        from scipy.ndimage import gaussian_filter

        H, xedges, yedges = np.histogram2d(ra_deg[mask], dec_deg[mask], bins=(400,200))
        Hs = gaussian_filter(H, sigma=2.0)

        fig, ax = plt.subplots(figsize=(7,4))
        im = ax.imshow(Hs.T, origin='lower', aspect='auto',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    cmap='magma')
        ax.set_xlabel("RA [deg]"); ax.set_ylabel("Dec [deg]")
        ax.set_title("Angular density (smoothed)")
        plt.colorbar(im, ax=ax, label="Smoothed counts")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    @staticmethod
    def plot_event_rates_hist_2d_waterfall(lgMBH, z_gal, generated_EMRIs, generated_TDEs, T_obs, filename):
        from matplotlib.colors import LogNorm
        from scipy.ndimage import gaussian_filter

        plt.style.use("dark_background")

        # plt.rcParams.update({
        #     "font.size": 13,
        #     "axes.linewidth": 1.2,
        #     "xtick.direction": "in",
        #     "ytick.direction": "in",
        #     "xtick.top": True,
        #     "ytick.right": True,
        #     "axes.facecolor": "#0d0d0d",
        #     "figure.facecolor": "#0d0d0d",
        #     })

        mass_bins = np.linspace(np.min(lgMBH), np.max(lgMBH), 400)
        z_bins = np.linspace(np.min(z_gal), np.max(z_gal), 1000)


        H_emri, xedges, yedges = np.histogram2d(lgMBH, z_gal,
            bins=[mass_bins, z_bins],
            weights=generated_EMRIs,
        )

        H_tde, _, _ = np.histogram2d(lgMBH, z_gal,
            bins=[mass_bins, z_bins],
            weights=generated_TDEs,
        )

        # Smooth for visualization
        H_emri = gaussian_filter(H_emri, sigma=1.2)
        H_tde = gaussian_filter(H_tde, sigma=1.2)

        X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')

        logM = X
        Z = Y

        # EMRI MODEL
        snr_emri = (500 * np.exp(-((logM - 5.7)**2) / 0.45) / (1 + Z)**1.2)
        EMRI_THRESHOLD = 20

        # TDE MODEL
        tde_lum = np.exp(-((logM - 6.0)**2) / 1.2)
        hills_cutoff = 1 / (1 + np.exp((logM - 7.7) * 12))
        distance_factor = np.exp(-Z / 0.35)

        tde_signal = tde_lum * hills_cutoff * distance_factor

        ztf_snr = 100 * tde_signal
        lsst_snr = 500 * tde_signal

        TDE_THRESHOLD = 5

        ztf_detect = ztf_snr >= TDE_THRESHOLD
        lsst_detect = lsst_snr >= TDE_THRESHOLD

        fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

        # TOP: EMRI POPULATION

        ax = axes[0]

        pcm = ax.pcolormesh(X, Y, H_emri, cmap='magma', shading='auto',
            norm=LogNorm(
                vmin=max(np.nanmin(H_emri[H_emri > 0]), 1e-3),
                vmax=np.nanmax(H_emri)
            )
        )

        levels = [20, 30, 50, 100, 200]

        cs = ax.contour(X, Y, snr_emri, levels=levels, colors='white', linewidths=1.2, alpha=0.9)

        ax.clabel(cs, inline=True, fontsize=9, fmt='%d', colors='white')

        # EMRI threshold (muted cyan)
        ax.contour(X, Y, snr_emri, levels=[EMRI_THRESHOLD],
            colors=['#4cc9f0'], linewidths=2.2
        )

        ax.text(
            4.3,
            3.8,
            r'EMRI SNR threshold',
            color='#4cc9f0',
            fontsize=11
        )

        ax.set_title(
            rf'EMRI Population ($N={np.sum(generated_EMRIs):.0f}$)',
            fontsize=15
        )

        ax.set_ylabel('Redshift $z$')

        cbar1 = fig.colorbar(pcm, ax=ax, pad=0.02)
        cbar1.set_label('Generated EMRI Events')

        # BOTTOM: TDE POPULATION
        ax2 = axes[1]

        pcm2 = ax2.pcolormesh(
            X,
            Y,
            H_tde,
            cmap='magma',
            shading='auto',
            norm=LogNorm(
                vmin=max(np.nanmin(H_tde[H_tde > 0]), 1e-3),
                vmax=np.nanmax(H_tde)
            )
        )

        # ZTF (amber contour)
        ax2.contour(
            X,
            Y,
            ztf_detect,
            levels=[0.5],
            colors=['#ffb703'],
            linewidths=2
        )

        # LSST (red contour)
        ax2.contour(
            X,
            Y,
            lsst_detect,
            levels=[0.5],
            colors=['#d62828'],
            linewidths=2.5
        )

        ax2.text(
            4.3,
            0.35,
            'ZTF',
            color='#ffb703',
            fontsize=12,
            weight='bold'
        )

        ax2.text(
            5.0,
            1.2,
            'LSST',
            color='#d62828',
            fontsize=12,
            weight='bold'
        )

        ax2.set_title(
            rf'TDE Population ($N={np.sum(generated_TDEs):.0f}$)',
            fontsize=15
        )

        ax2.set_xlabel(r'$\log_{10}(M_{\rm BH}/M_\odot)$')
        ax2.set_ylabel('Redshift $z$')

        cbar2 = fig.colorbar(pcm2, ax=ax2, pad=0.02)
        cbar2.set_label('Generated TDE Events')

        # ============================================================
        # FINAL FORMATTING
        # ============================================================

        for a in axes:
            a.set_xlim(4.0, 8.5)
            a.set_ylim(0, 8)
            a.grid(True, linestyle='--', alpha=0.25)

        plt.tight_layout()

        plt.savefig(filename, dpi=300)

    @staticmethod
    def plot_MBH_vs_waterfall(lgMBH, z_gal, generated_EMRIs, generated_TDEs, T_obs, filename):

        from matplotlib.colors import Normalize
        from matplotlib.colors import LogNorm

        Mbh = np.logspace(4, 8.5, 500)
        z = np.linspace(0.001, 8.0, 500)
        M, Z = np.meshgrid(Mbh, z)
        logM = np.log10(M)

        # EMRI SNR MODEL (TOP PANEL)
        snr_emri = (
            500
            * np.exp(-((logM - 5.7)**2) / 0.45)
            / (1 + Z)**1.2
        )

        EMRI_THRESHOLD = 20

        # TDE MODELS (BOTTOM PANEL)
        # TDE luminosity envelope

        tde_lum = (
            np.exp(-((logM - 6.0)**2) / 1.2)
        )

        # Hills mass suppression
        hills_cutoff = 1 / (1 + np.exp((logM - 7.7) * 12))

        # Cosmological dimming
        tde_signal = tde_lum * hills_cutoff / (1 + Z)**2

        distance_factor = np.exp(-Z / 0.35)

        tde_signal = tde_lum * hills_cutoff * distance_factor

        # Toy survey SNRs
        ztf_snr = 100 * tde_signal
        lsst_snr = 500 * tde_signal

        # Detection threshold
        TDE_THRESHOLD = 5

        ztf_detect = ztf_snr >= TDE_THRESHOLD
        lsst_detect = lsst_snr >= TDE_THRESHOLD

        # ============================================================
        # FIGURE — SINGLE AXES: overlay EMRI SNR contours and TDE detectability
        # ============================================================

        fig, ax = plt.subplots(figsize=(12, 7))

        # EMRI SNR contours (base layer)
        levels = [20, 30, 50, 100, 200]

        cf = ax.contourf(
            M,
            Z,
            snr_emri,
            levels=levels,
            cmap="YlGnBu",
            norm=LogNorm(vmin=20, vmax=300),
        )

        # Contour lines
        cs = ax.contour(
            M,
            Z,
            snr_emri,
            levels=levels,
            colors='black',
            linewidths=1.2,
        )
        ax.clabel(cs, inline=True, fontsize=10, fmt='%d')

        # EMRI threshold contour
        ax.contour(
            M,
            Z,
            snr_emri,
            levels=[EMRI_THRESHOLD],
            colors='red',
            linewidths=2.5,
        )

        # Scatter MBH locations
        cmap_values = generated_TDEs + generated_EMRIs
        norm = Normalize(vmin=np.min(cmap_values), vmax=np.max(cmap_values)) 
        sc = ax.scatter(10**lgMBH, z_gal, s=10, c=cmap_values, cmap='viridis', norm=norm, alpha=0.7)

        # Colorbar for the normalized combined events
        cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.045)
        cbar.set_label("Normalized Number of Events (TDE + EMRI)", fontsize=14)

        # Overlay TDE detectability regions (transparent)
        # ZTF region
        ax.contourf(
            M,
            Z,
            ztf_detect,
            levels=[0.5, 1],
            colors=['orange'],
            alpha=0.35,
        )
        ax.contour(
            M,
            Z,
            ztf_detect,
            levels=[0.5],
            colors=['darkorange'],
            linewidths=2,
        )

        # LSST region
        ax.contourf(
            M,
            Z,
            lsst_detect,
            levels=[0.5, 1],
            colors=['red'],
            alpha=0.22,
        )
        ax.contour(
            M,
            Z,
            lsst_detect,
            levels=[0.5],
            colors=['darkred'],
            linewidths=2,
        )

        # Labels inside plot
        ax.text(
            2e4,
            0.4,
            "ZTF",
            fontsize=15,
            color='darkorange',
            weight='bold'
        )

        ax.text(
            2e5,
            1.5,
            "LSST",
            fontsize=15,
            color='darkred',
            weight='bold'
        )

        # Formatting
        ax.set_xscale('log')
        ax.set_xlim(1e4, 3e8)
        ax.set_ylim(0, 8)

        ax.set_xlabel(r"SMBH Mass [$M_\odot$]", fontsize=16)
        ax.set_ylabel("Redshift", fontsize=15)

        ax.grid(True, which='both', linestyle='--', alpha=0.4)

        # Colorbar for EMRI SNR
        cbar = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.045)
        cbar.set_label("EMRI SNR", fontsize=14)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
            
    @staticmethod
    def plot_comparison_histogram(key, loc='.', x_label=None, **kwargs_sources):
        """
        Plot comparison histograms for different datasets.

        Parameters:
            key (str): The key/column to plot from each dataset.
            loc (str): Directory path to save the figure.
            **kwargs_sources: Arbitrary keyword arguments of the form name=dataframe,
                            where `name` is the label and `dataframe` contains the key.
        """
        plt.figure(figsize=(8,6))
        
        combined_data = []
        
        for label, df in kwargs_sources.items():
            if key not in df:
                print(f"Key '{key}' not found in dataframe '{label}'")
                continue
            data = df[key].mean(axis=1)
            combined_data.append(data)
            plt.hist(data, bins=40, histtype='step', label=label)
        
        # Plot combined histogram
        combined_array = np.concatenate(combined_data)
        plt.hist(combined_array, bins=40, color='black', histtype='step', linestyle='--', linewidth=2, label='Combined')
        
        # Labels and styling
        plt.xlabel(x_label, fontsize=16)
        plt.ylabel("Counts", fontsize=16)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f'{loc}/{key}_comparison_histogram.png')
        plt.close()

    @staticmethod
    def make_waterfall_plot_EMRIs():
        pass

    @staticmethod    
    def make_waterfall_plot_TDEs():
        pass