import numpy as np
import torch
import math 

from scipy.interpolate import RegularGridInterpolator

from galaxy import *
from nsc import NSC
from density import DehnenProfile
from relaxation import RelaxationModel
from rate import RateModel
from evolution import CuspEvolution
from cosmology import LastMajorMerger, GalaxyStellarMassFunction, MBHMassFunction, CosmologyModel, CO_mass_function

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
        self.r_grid = torch.logspace(-5, 3, 500, device=device)  # pc
        
        self.pdf_grid = None
        self.interp = None

        self.cosmo = CosmologyModel()
        self.MBHMF = MBHMassFunction(gsmf=GalaxyStellarMassFunction())
        self.T_obs_det = 4  # years
        self.Ntot_EMRI = 1e6
        self.component_masses_sBH = np.random.uniform(1., 100, 100000)
        self.component_masses_stars = np.random.uniform(1., 100, 100000)

    def _sanitise_inputs(self, **inputs):
        outs = {}
        for a, b in inputs.items():
            b_tensor = torch.as_tensor(b, device=self.device)
            if b_tensor.ndim == 0:
                b_tensor = b_tensor[None]
            outs[a] = b_tensor
        return outs
    
    def pdf(self):
        raise NotImplementedError

    def evaluate_at_z_theta(self, z, theta):
        """
        Continuous PDF evaluation using 2D interpolation.
        z, theta can be scalars or arrays.
        """
        pts = np.array([z, theta]).T  # shape (N, 2)
        return self.interp(pts)

    def interpolate(self, pdf_2d_normalized):
        """
        Build 2D interpolator using SciPy (continuous PDF evaluation)
        """

        self.pdf_grid = pdf_2d_normalized
        self.interp = RegularGridInterpolator(
            (self.z_np, self.theta_np),
            pdf_2d_normalized,
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


    def pdf(self, **hypers):

        Nz, Nm = len(self.z_grid), len(self.theta_grid)
        pdf = np.zeros((Nz, Nm))

        hypers = self._sanitise_inputs(**hypers)
        gamma = float(hypers["gamma"])

        for i, z in enumerate(self.z_grid.cpu().numpy()):

            dVc_dz = self.cosmo.dVc_dz(z)
            phi = 10**self.MBHMF.get_mbhmf(self.theta_grid.cpu().numpy(), z)

            for j, lgMBH in enumerate(self.theta_grid.cpu().numpy()):

                lgMgal = Galaxy.lgMgal_from_lgMBH(lgMBH)
                gal = Galaxy(lgMgal, z)
                nsc = NSC(gal, lgMBH)
                profile = DehnenProfile(nsc, gamma)
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
                # print(i, j, z, lgMBH, dVc_dz, phi[j], pdf[i, j])
        
        pdf /= np.sum(pdf)
        self.interpolate(pdf)
        return pdf

class dN_da_dz(Distribution2D):

    def __init__(self, limits_z, limits_theta, limits_MBH, npoints=200, grid_spacing='linear', device="cpu"):
        super().__init__(limits_z, limits_theta, npoints, grid_spacing, device)
        
        self.dN_dlgMBH_dz = dN_dlgMBH_dz(limits_z, limits_MBH, npoints, grid_spacing, device)
        self.a_grid = self.theta_grid 

        self.logMBH_grid = self.dN_dlgMBH_dz.theta_grid.cpu().numpy()
        # self.logMBH_grid_max  = self.logMBH_grid.max()

        self.dlogM = np.diff(self.logMBH_grid, prepend=self.logMBH_grid[0])

    def pdf(self, **hypers):

        Nz, Nm = len(self.z_grid), len(self.theta_grid)
        pdf = np.zeros((Nz, Nm))

        hypers = self._sanitise_inputs(**hypers)
        beta = float(hypers["beta"])
        lambda_alpha = float(hypers["lambda_alpha"])
        gamma = float(hypers["gamma"])


        lgalpha_M = ( beta + lambda_alpha * (self.logMBH_grid - 6) ) # shape (Nm,)

        pdf_dN_dlgMBH_dz = self.dN_dlgMBH_dz.pdf(gamma=gamma) # shape (Nz, Nm)

        for i, z in enumerate(self.z_grid.cpu().numpy()):

            # pdf_mass[iz,:] is 1D array over M
            fM = pdf_dN_dlgMBH_dz[i, :]                # shape (Nm,)

            # compute beta PDF for all a values for each mass point
            a = self.a_grid.cpu().numpy()       # shape (Na,)

            # matrix of shape (Nm, Na)
            beta_matrix = np.zeros((len(self.logMBH_grid), len(a)))

            for j, lgalpha_j in enumerate(lgalpha_M):
                # Beta(alpha_j, beta) normalization:

                B = math.gamma(lgalpha_j) * math.gamma(beta) / math.gamma(lgalpha_j + beta)

                beta_matrix[j,:] = (a**(lgalpha_j - 1)) * ((1-a)**(beta - 1)) / B

            # Convolution in mass dimension:
            # sum_j [ pdf_dN_dlgMBH_dz[z,j] * beta_pdf[j,a] * dlogM[j] ]
            
            pdf[i,:] = np.sum(fM[:,None] * beta_matrix * self.dlogM[:,None], axis=0)

        pdf/=np.sum(pdf)
        breakpoint()
        self.interpolate(pdf)
        return pdf

class dN_dCO_dz(Distribution2D):

    def __init__(self, limits_z, limits_theta, limits_MBH, npoints=200, grid_spacing='linear', device="cpu"):
        super().__init__(limits_z, limits_theta, npoints, grid_spacing, device)
        
        self.dN_dlgMBH_dz = dN_dlgMBH_dz(limits_z, limits_MBH, npoints, grid_spacing, device)
        self.CO_mass_grid = self.theta_grid 

        self.logMBH_grid = self.dN_dlgMBH_dz.theta_grid.cpu().numpy()
        self.dlogM = np.diff(self.logMBH_grid, prepend=self.logMBH_grid[0])

    def pdf(self, **hypers):

        Nz, Nm = len(self.z_grid), len(self.theta_grid)
        pdf = np.zeros((Nz, Nm))

        hypers = self._sanitise_inputs(**hypers)
        gamma = float(hypers["gamma"])

        pdf_dN_dlgMBH_dz = self.dN_dlgMBH_dz.pdf(gamma=gamma) # shape (Nz, Nm)

        for i, z in enumerate(self.z_grid.cpu().numpy()):

            # pdf_mass[iz,:] is 1D array over M
            fM = pdf_dN_dlgMBH_dz[i, :]                # shape (Nm,)

            # compute beta PDF for all a values for each mass point
            mu = self.CO_mass_grid.cpu().numpy()       # shape (Na,)

            p_mu_given_M = np.zeros((len(self.logMBH_grid), len(mu)))

            for j, lgMBH in enumerate(self.logMBH_grid):

                lgMgal = Galaxy.lgMgal_from_lgMBH(lgMBH)
                gal = Galaxy(lgMgal, z)
                nsc = NSC(gal, lgMBH)
                
                r_max = nsc.r_influence(unit='pc')
                r_min = nsc.r_capture(unit='pc')

                profile = DehnenProfile(nsc, gamma)

                I_r = profile.number_of_CO_within_shell(r_min=r_min, r_max=r_max, Ntot=self.Ntot_EMRI, kind='EMRI', npts=len(self.r_grid))

                psi = CO_mass_function().delta_distribution(m=self.CO_mass_grid, M_CO=10.0)
                # breakpoint()

                p_mu_given_M[j, :] = 1 # psi * I_r / (np.trapezoid(psi, self.CO_mass_grid) * I_r) 

            # Convolution in mass dimension:
            # sum_j [ pdf_dN_dlgMBH_dz[z,j] * pdf_mu[j,mu] * dlogM[j] ]
            pdf[i,:] = np.sum(fM[:,None] * p_mu_given_M * self.dlogM[:,None], axis=0)
        pdf/=np.sum(pdf)
        breakpoint()
        self.interpolate(pdf)
        return pdf



# dist_dN_dlgMBH_dz = dN_dlgMBH_dz(limits_z=(0.001, 10), limits_theta=(4, 8.5), npoints=5, grid_spacing='linear', device="cpu")
# pdf_dN_dlgMBH_dz = dist_dN_dlgMBH_dz.pdf(gamma=1.5)

# z_samp, mbh_samp = dist_dN_dlgMBH_dz.draw_samples(50)

# plt.figure(figsize=(7,6))
# plt.imshow(pdf_dN_dlgMBH_dz.T, origin='lower',
#            extent=[dist_dN_dlgMBH_dz.z_np[0], dist_dN_dlgMBH_dz.z_np[-1], dist_dN_dlgMBH_dz.theta_np[0], dist_dN_dlgMBH_dz.theta_np[-1]],
#            aspect='auto', cmap='viridis')
# plt.colorbar(label=r'$d^2N/d\log M\, dz$')
# plt.xlabel('z')
# plt.ylabel(r'$\log_{10} M_{\rm BH}$')
# plt.title('EMRI 2D PDF')
# plt.savefig('dN_dlgMBH_dz.pdf', dpi=200)
# plt.show()

# dist_dN_da_dz = dN_da_dz(limits_z=(0.001, 10), limits_theta=(0.1, 0.998), limits_MBH=(4, 8.5), npoints=5, grid_spacing='linear', device="cpu")
# pdf_dN_da_dz = dist_dN_da_dz.pdf(beta=6.0, lambda_alpha=2.7, gamma=1.5)
# z_samp, a_samp = dist_dN_da_dz.draw_samples(50)

# plt.figure(figsize=(7,6))
# plt.imshow(pdf_dN_da_dz.T, origin='lower',
#            extent=[dist_dN_da_dz.z_np[0], dist_dN_da_dz.z_np[-1], dist_dN_da_dz.theta_np[0], dist_dN_da_dz.theta_np[-1]],
#            aspect='auto', cmap='viridis')
# plt.colorbar(label=r'$d^2N/da\,dz$')
# plt.xlabel('z')
# plt.ylabel(r'$a$')
# plt.title('EMRI 2D PDF')
# plt.savefig('dN_da_dz.pdf', dpi=200)
# plt.show()

# dN_dCO_dz = dN_dCO_dz(limits_z=(0.001, 10), limits_theta=(10-1E-7, 10+1E-7), limits_MBH=(4, 8.5), npoints=5, grid_spacing='linear', device="cpu")
# pdf_dN_dCO_dz = dN_dCO_dz.pdf(gamma=1.5)
