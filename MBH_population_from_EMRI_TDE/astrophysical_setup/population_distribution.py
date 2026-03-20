import numpy as np
import torch

from scipy.interpolate import RegularGridInterpolator

from galaxy import *
from nsc import NSC
from density import DehnenProfile
from relaxation import RelaxationModel
from rate import RateModel
from evolution import CuspEvolution
from cosmology import LastMajorMerger, GalaxyStellarMassFunction, MBHMassFunction, CosmologyModel

import matplotlib.pyplot as plt
class Distribution2D:
    def __init__(self, limits_z, limits_theta, npoints=200, grid_spacing="linear", device="cpu"):
        self.device = device

        # --- Build grid ---
        if grid_spacing == "linear":
            self.z_grid = torch.linspace(limits_z[0], limits_z[1], npoints, device=device)
            self.theta_grid = torch.linspace(limits_theta[0], limits_theta[1], npoints, device=device)
        else:
            self.z_grid = torch.logspace(torch.log10(limits_z[0]), torch.log10(limits_z[1]), npoints, device=device)
            self.theta_grid = torch.logspace(torch.log10(limits_theta[0]), torch.log10(limits_theta[1]), npoints, device=device)

        # Store numpy for SciPy interpolation
        self.z_np = self.z_grid.cpu().numpy()
        self.theta_np = self.theta_grid.cpu().numpy()

        self.pdf_grid = None
        self.interp = None
    
    def pdf(self):
        raise NotImplementedError

    def interpolate(self, pdf_2d):
        """
        Build 2D interpolator using SciPy (continuous PDF evaluation)
        """
        pdf_2d = pdf_2d / np.sum(pdf_2d)  # normalize

        self.pdf_grid = pdf_2d
        self.interp = RegularGridInterpolator(
            (self.z_np, self.theta_np),
            pdf_2d,
            bounds_error=False,
            fill_value=0.0
        )

    def cdf(self, **pdf_kwargs):
        """
        Compute CDF from PDF grid.
        """
        if self.pdf_grid is None:
            self.pdf_grid = self.pdf(**pdf_kwargs)

        flat_pdf = self.pdf_grid.ravel()
        cdf = np.cumsum(flat_pdf)
        cdf /= cdf[-1]  # normalize
        return cdf

    def draw_samples(self, size, **pdf_kwargs):
        """
        Sample from 2D PDF using flatten → CDF → inverse transform.
        """
        cdf = self.cdf(**pdf_kwargs)

        # Draw uniform samples and find corresponding indices in the CDF
        u = np.random.rand(size)

        # searchsorted returns the indices where elements should be inserted to maintain order
        idx = np.searchsorted(cdf, u)
        print(f"Sampled indices in flattened PDF: {idx}")  # Debug: print first 10 indices

        # Convert flat indices back to 2D grid indices
        Nz = len(self.z_np)
        Ntheta = len(self.theta_np)

        # Compute 2D indices
        iz = idx // Ntheta # jump by rows (z-axis)
        it = idx % Ntheta # jump by columns (theta-axis)

        return self.z_np[iz], self.theta_np[it]


class dN_dlgMBH_dz(Distribution2D):

    def __init__(self, limits_z, limits_theta, npoints=200, grid_spacing='linear', device="cpu"):
        super().__init__(limits_z, limits_theta, npoints, grid_spacing, device)

        self.cosmo = CosmologyModel()
        self.MBHMF = MBHMassFunction(gsmf=GalaxyStellarMassFunction())
        self.T_obs_det = 4  # years
        self.Ntot_EMRI = 1e6
        self.component_masses_sBH = np.random.uniform(1., 100, 100000)
        self.component_masses_stars = np.random.uniform(1., 100, 100000)


    def pdf(self):
        """
        Computes the 2D PDF grid d^2N / (dz dlogM)
        """

        Nz, Nm = len(self.z_grid), len(self.theta_grid)
        pdf = np.zeros((Nz, Nm))

        for i, z in enumerate(self.z_grid.cpu().numpy()):

            dVc_dz = self.cosmo.dVc_dz(z)
            phi = 10**self.MBHMF.get_mbhmf(self.theta_grid.cpu().numpy(), z)

            for j, lgMBH in enumerate(self.theta_grid.cpu().numpy()):

                lgMgal = Galaxy.lgMgal_from_lgMBH(lgMBH)
                gal = Galaxy(lgMgal, z)
                nsc = NSC(gal, lgMBH)
                profile = DehnenProfile(nsc)
                relax = RelaxationModel(nsc, profile)
                rate = RateModel(nsc)
                evol = CuspEvolution(nsc, relax, rate, LastMajorMerger(self.cosmo))

                T_c = evol.cusp_age(
                    Ntot=self.Ntot_EMRI,
                    component_masses=self.component_masses_sBH,
                    kvir=1.0,
                    kind='EMRI',
                    mbar=10.,
                    unit='Gyr'
                )

                t_EMRI = evol.t_EMRI
                Gamma_hat = evol.Gamma_hat_EMRI
                Gamma = Gamma_hat * rate.universal_EMRI_rate(evol.evaluate_tau(T_c, t_EMRI))

                T_obs = self.T_obs_det / (1 + z)

                pdf[i, j] = dVc_dz * phi[j] * Gamma * T_obs
                print(i, j, z, lgMBH, dVc_dz, phi[j], pdf[i, j])

        self.interpolate(pdf)
        return pdf


dist = dN_dlgMBH_dz(limits_z=(0.001, 10), limits_theta=(4, 8.5), npoints=10)
pdf = dist.pdf()

z_samp, mbh_samp = dist.draw_samples(200)

breakpoint()


plt.figure(figsize=(7,6))
plt.imshow(pdf.T, origin='lower',
           extent=[dist.z_np[0], dist.z_np[-1], dist.theta_np[0], dist.theta_np[-1]],
           aspect='auto', cmap='viridis')
plt.colorbar(label=r'$d^2N/d\log M\, dz$')
plt.xlabel('z')
plt.ylabel(r'$\log_{10} M_{\rm BH}$')
plt.title('EMRI 2D PDF')
plt.show()


