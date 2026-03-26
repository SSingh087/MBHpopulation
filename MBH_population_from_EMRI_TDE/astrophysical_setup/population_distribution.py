import torch
import numpy as np
import warnings
import matplotlib.pyplot as plt

from cosmology import CosmologyModel, GalaxyStellarMassFunction, MBHMassFunction, LastMajorMerger
from galaxy import Galaxy
from nsc import NSC
from density import DehnenProfile
from relaxation import RelaxationModel
from rate import RateModel
from evolution import CuspEvolution

from config import (MBH_A, MBH_B, MBH_sigma0, MBH_scatter)


class PopulationDistribution:
    """
    Torch-first base: builds grid and offers helpers (mesh/flatten/reshape),
    cosmology terms, MBH MF, EMRI rate, interpolation + sampling.
    """
    def __init__(self, limits_z, limits_theta, limits_MBH, npoints=1000, grid_spacing="linear", device="cpu", dtype=torch.float64):
        self.device = device
        self.dtype = dtype


        # z-grid is always linear.
        self.z_grid = torch.linspace(limits_z[0], limits_z[1], npoints, device=device, dtype=dtype)


        # The theta grid can be either linear or logarithmic depending on the user's choice.
        if grid_spacing == "linear":
            self.theta_grid = torch.linspace(limits_theta[0], limits_theta[1], npoints, device=device, dtype=dtype)
        else:
            self.theta_grid = torch.logspace(
                torch.log10(torch.tensor(limits_theta[0], device=device, dtype=dtype)),
                torch.log10(torch.tensor(limits_theta[1], device=device, dtype=dtype)),
                npoints, device=device, dtype=dtype
            )

        self.cosmo = CosmologyModel()
        self._gsmf = GalaxyStellarMassFunction()
        self._mbhmf = MBHMassFunction(gsmf=self._gsmf)

        # here we need to evaluate if the galaxies corresponding to the MBHs 
        # actually exisits when sampled from the GalaxyMassFunction

        self.lgMgal, _ = self._gsmf.get_gsmf(z_gal=self.z_grid, n_points_mass=len(self.theta_grid))
        self.nucleation_indices = Galaxy.check_nucleation(self.lgMgal, self.z_grid)

        self.galaxies_on_grid = Galaxy(lgMgal=self.lgMgal, z_gal=self.z_grid, nucleation_occurs=self.nucleation_indices)

        print(f"Number of galaxies in grid: {len(self.z_grid)}, Number of nucleated galaxies: {self.nucleation_indices.sum()}")

        # calculate sigma and lgMBH
        self.lgMgal = self.lgMgal[self.nucleation_indices]
        self.sigma_pc_yr = torch.tensor(self.galaxies_on_grid.sigma_pc_yr)
        self.lgMBH_mass_from_galaxy_object = torch.tensor(self.galaxies_on_grid.lgMBH_mass, device=self.device, dtype=self.dtype)
        
        self.z_grid = self.z_grid[self.nucleation_indices]
        
        # sort by lgMBH, z and other parameters accordingly to ensure that the grid is consistent with the MBH mass function and the galaxy properties.
        # This is important because the MBH mass function is derived from the galaxy properties, and we want to make sure that the grid points are ordered in a way that reflects this relationship
        self.lgMBH_sorted, sorted_indices = torch.sort(self.lgMBH_mass_from_galaxy_object)
        self.lgMgal_sorted = self.lgMgal[sorted_indices]
        self.z_grid_sorted = self.z_grid[sorted_indices]
        self.theta_grid_sorted = self.theta_grid[sorted_indices] 
        self.sigma_pc_yr_grid_sorted = self.sigma_pc_yr[sorted_indices]

        print(f"Number of galaxies after sorting: {len(self.z_grid)}")

        # at this stage we have a pair of lgMBH_sorted and z_grid that are consistent with
        # each other, and we can use them to construct the 2D mesh after checking
        # if max and min of MBH grid are within the limits provided by the user

        self.Ntot_EMRI = 1E5 * torch.ones_like(self.z_grid) # this is the total number of COs in the NSC, can be scaled with galaxy properties in the future
        self.component_masses_sBH = torch.full_like(self.z_grid, 10.0) # this is the mass of the sBHs in the NSC, can be scaled with galaxy properties in the future

        if limits_MBH[-1] < max(self.lgMBH_sorted) and limits_MBH[0] > min(self.lgMBH_sorted):
            print("MBH grid is consistent with limits_MBH.")
        else:

            # this part is to ensure that the MBH grid is consistent with the limits provided by the user
            # if not we adjust the grid to be within the limits and issue a warning. 
            # This is important because the MBH grid is derived from the 
            # galaxy properties and may not always align perfectly with the user's specified 
            # limits, especially if the limits are narrow or if the GalaxyMassFunction 
            # parameters lead to a different distribution of MBHs.

            warnings.warn("MBH grid from GalaxyMassFunction is not consistent with limits_MBH. Please adjust limits or check GalaxyMassFunction parameters.")

            mask = (self.lgMBH_sorted >= limits_MBH[0]) & (self.lgMBH_sorted <= limits_MBH[1])

            self.lgMgal_sorted = self.lgMgal_sorted[mask]

            self.lgMBH_sorted = self.lgMBH_sorted[mask]
            self.sigma_pc_yr_grid_sorted = self.sigma_pc_yr_grid_sorted[mask]

            self.z_grid_sorted = self.z_grid_sorted[mask]
            self.theta_grid_sorted = self.theta_grid_sorted[mask]

            # nucleation check is already applied before, so we can set nucleation_occurs=True for all of them.
            self.galaxies_on_grid = Galaxy(lgMgal=self.lgMgal_sorted, lgMBH=self.lgMBH_sorted, z_gal=self.z_grid_sorted, nucleation_occurs=True)

            self.Ntot_EMRI = 1E5 * torch.ones_like(self.z_grid_sorted) # this is the total number of COs in the NSC, can be scaled with galaxy properties in the future
            self.component_masses_sBH = torch.full_like(self.z_grid_sorted, 10.0) # this is the mass of the sBHs in the NSC, can be scaled with galaxy properties in the future

            print(f"Number of galaxies after applying MBH limits: {len(self.z_grid_sorted)}")

        self.T_obs_det = torch.tensor(4.0)

        self.N = self.z_grid_sorted.numel() # this is the number of valid (z, lgMBH) pairs after nucleation check and sorting
    
    def _sanitise_inputs(self, **inputs):
        outs = {}
        for a, b in inputs.items():
            b_tensor = torch.as_tensor(b, device=self.device)
            if b_tensor.ndim == 0:
                b_tensor = b_tensor[None]
            outs[a] = b_tensor
        return outs

    def pdf(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def plot_marginal_theta(self, theta, pdf, bins=40, ax=None):
        """
        Plot weighted 1D marginal distribution over log10(M_BH).
        """
        theta = self._to_numpy(theta)
        pdf   = self._to_numpy(pdf)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6,4))

        ax.hist(theta, bins=bins, weights=pdf, density=True, alpha=0.75)
        ax.set_xlabel(r'$\log_{10}(M_{\rm BH}/M_\odot)$')
        ax.set_ylabel(r'$p(\log M_{\rm BH})$')
        ax.set_title("Marginal MBH Distribution")

        return ax

    def plot_marginal_z(self, z, pdf, bins=40, ax=None):
        """
        Plot weighted 1D marginal distribution over redshift.
        """
        z = self._to_numpy(z)
        pdf = self._to_numpy(pdf)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6,4))

        ax.hist(z, bins=bins, weights=pdf, density=True, alpha=0.75)
        ax.set_xlabel(r'$z$')
        ax.set_ylabel(r'$p(z)$')
        ax.set_title("Redshift Marginal Distribution")

        return ax

    def plot_joint_2D(self, z, theta, pdf, bins=40, ax=None):
        """
        Weighted 2D joint distribution p(z, logMBH)
        """
        z = self._to_numpy(z)
        theta = self._to_numpy(theta)
        pdf = self._to_numpy(pdf)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6,5))

        h = ax.hist2d(z, theta, bins=bins, weights=pdf,
                      density=True, cmap='viridis')
        plt.colorbar(h[3], ax=ax, label=r'$p(z, \log M_{\rm BH})$')

        ax.set_xlabel(r'$z$')
        ax.set_ylabel(r'$\log_{10}(M_{\rm BH}/M_\odot)$')
        ax.set_title("Joint Distribution p(z, log M_BH)")

        return ax

    def plot_joint_2D_smooth(self, z, theta, pdf, bins=100, ax=None):
        """
        Kernel-smoothed 2D density estimator for nicer publication-quality plots.
        """
        from scipy.stats import gaussian_kde

        z = self._to_numpy(z)
        theta = self._to_numpy(theta)
        pdf = self._to_numpy(pdf)

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

        cf = ax.contourf(Zg, Tg, density, levels=40, cmap="inferno")
        plt.colorbar(cf, ax=ax, label="Density")

        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"$\log_{10}(M_{\rm BH}/M_\odot)$")
        ax.set_title("Smoothed Joint PDF (KDE)")

        return ax

    def _to_numpy(self, x):
        """Convert torch tensor or numpy array to numpy array."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)


class dN_dlgMBH_dz(PopulationDistribution):
    def __init__(self, limits_z, limits_theta, limits_MBH, npoints=200, grid_spacing='linear', device="cpu", dtype=torch.float64):
        super().__init__(limits_z, limits_theta, limits_MBH, npoints, grid_spacing, device, dtype)

    @torch.no_grad()
    def pdf(self, X, **hypers) -> torch.Tensor:
        
        z_gal, theta = X
        
        hypers = self._sanitise_inputs(**hypers)
        gamma = float(hypers["gamma"])
        
        
        z_gal = torch.as_tensor(z_gal, device=self.device, dtype=self.dtype)
        theta = torch.as_tensor(theta, device=self.device, dtype=self.dtype)
        
        # compare closest MBH in the grid to the input theta (lgMBH) and get the corresponding galaxy properties (lgMgal, sigma, etc.) for that grid point.
        # This is necessary because the input theta may not exactly match the grid points, so we need to find the closest one and use its properties for the calculations.
        idx = torch.searchsorted(self.theta_grid_sorted, theta)
        idx = torch.clamp(idx, 1, len(self.theta_grid_sorted) - 1)
        left = self.theta_grid_sorted[idx - 1]
        right = self.theta_grid_sorted[idx]
        idx -= (torch.abs(theta - left) < torch.abs(theta - right)).long()

        # galactic parmeters corresponding to the input theta (lgMBH) after checking for nucleation and sorting
        lgMgal_derived = self.lgMgal_sorted[idx]
        sigma_pc_yr_derived = self.sigma_pc_yr_grid_sorted[idx]

        # NSC and EMRI parameters corresponding to the input theta (lgMBH) after checking for nucleation and sorting
        Ntot_EMRI = self.Ntot_EMRI[idx]
        component_masses_sBH = self.component_masses_sBH[idx]

        __galaxies__ = Galaxy(lgMgal=lgMgal_derived, lgMBH=theta, sigma_pc_yr=sigma_pc_yr_derived, z_gal=z_gal, nucleation_occurs=True)

        nscs = NSC(__galaxies__, theta)
        profiles = DehnenProfile(nscs, gamma)
        relax_models = RelaxationModel(nscs, profiles)
        rates = RateModel(nscs)
        evol = CuspEvolution(nscs, relax_models, rates, LastMajorMerger(self.cosmo))
        tau, T_c, t_EMRI, Gamma_hat_EMRI = evol.evaluate_tau(Ntot=Ntot_EMRI, component_masses=component_masses_sBH, kvir=1.0, kind='EMRI', mbar=10., unit='Gyr', A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter)

        Gamma = torch.tensor(Gamma_hat_EMRI * rates.universal_EMRI_rate(tau))

        T_obs = self.T_obs_det / (1 + z_gal)  

        phi_linear = torch.tensor(self._mbhmf.eval_mbhmf(z=z_gal, logMBH=theta, n_points_mass=len(theta), return_log10=False))
        phi_linear = torch.nan_to_num(phi_linear, nan=0.0, posinf=0.0, neginf=0.0) # 
        
        dVc_dz = torch.tensor(self.cosmo.dVc_dz(z_gal))

        pdf = dVc_dz * phi_linear * Gamma * T_obs

        total = torch.sum(pdf)

        if (not torch.isfinite(total)) or (total <= 0):
            raise RuntimeError("Computed PDF is zero everywhere. Check inputs/units.")

        pdf = pdf / total
        # self.interpolate(pdf)
        return pdf


N_objs = 100

z_gal = torch.tensor(np.random.uniform(0.01, 5, size=N_objs))

GSMF = GalaxyStellarMassFunction()
lgMgal_samples = GSMF.sample_gsmf(z_gal=z_gal, size=N_objs)
nucleation_indices = Galaxy.check_nucleation(lgMgal_samples, z_gal)


galaxies = Galaxy(lgMgal=lgMgal_samples, z_gal=z_gal, nucleation_occurs=nucleation_indices)
lgMBH_mass_from_galaxies = torch.tensor(galaxies.lgMBH_mass)
z_gal = z_gal[nucleation_indices]

print(f"passing {nucleation_indices.sum()} nucleated galaxies")

# in this instance we fix the z_gals and MBHs and all its properties

# now for all the nucleated galaxies we will simulate EMRIs and TDEs
# for all the similated TDEs and EMRI check the SNR 
# if SNR > threshhold the pass it to Fisher Matrix code 
# assuming Fisher matrix returns values as gaussian around the injected values
# so z_Gal and lgMBH should be gaussian around the mean for testing purposes

dist_dN_dlgMBH_dz = dN_dlgMBH_dz(limits_z=(z_gal[0], z_gal[-1]), limits_theta=(lgMBH_mass_from_galaxies.min(), lgMBH_mass_from_galaxies.max()), limits_MBH=(lgMBH_mass_from_galaxies.min(), lgMBH_mass_from_galaxies.max()), npoints=10, grid_spacing='linear', device="cpu")
pdf_dN_dlgMBH_dz = dist_dN_dlgMBH_dz.pdf(X=(z_gal, lgMBH_mass_from_galaxies), gamma=1.5)


# dist_dN_dlgMBH_dz.plot_marginal_theta(lgMBH_mass_from_galaxies, pdf_dN_dlgMBH_dz.cpu(), bins=20)
# plt.show()  

# dist_dN_dlgMBH_dz.plot_marginal_z(z_gal, pdf_dN_dlgMBH_dz.cpu(), bins=20)
# plt.show()

dist_dN_dlgMBH_dz.plot_joint_2D_smooth(z_gal, lgMBH_mass_from_galaxies, pdf_dN_dlgMBH_dz.cpu(), bins=50)
plt.show()


# distribution = Distribution2D(limits_z=(0.01, 10.0), limits_theta=(4.0, 8.5), limits_MBH=(4.0, 8.5), npoints=500, device='cpu', dtype=torch.float64)

# dist_dN_dlgMBH_dz = dN_dlgMBH_dz(limits_z=(0.001, 10), limits_theta=(4, 8.5), limits_MBH=(4, 8.5), npoints=1000, grid_spacing='log', device="cpu")

# pdf_dN_dlgMBH_dz = dist_dN_dlgMBH_dz.pdf( , gamma=1.5)
# # breakpoint()
# plt.scatter(dist_dN_dlgMBH_dz.theta_grid.cpu().numpy(), dist_dN_dlgMBH_dz.lgMBH_sorted.cpu().numpy(),
#     c=pdf_dN_dlgMBH_dz.cpu().numpy(), s=5, cmap='viridis')
# plt.xlabel(r'$z$')
# plt.ylabel(r'$dN/dzd\log_{10}M_{\rm BH}$')
# plt.colorbar(label='PDF')
# plt.show()

# class dN_da_dz(Distribution2D):

#     def __init__(self, limits_z, limits_theta, limits_MBH, npoints=200, grid_spacing='linear', device="cpu"):
#         super().__init__(limits_z, limits_theta, npoints, grid_spacing, device)
        
#         self.dN_dlgMBH_dz = dN_dlgMBH_dz(limits_z, limits_MBH, npoints, grid_spacing, device)
#         self.a_grid = self.theta_grid 

#         self.logMBH_grid = self.dN_dlgMBH_dz.theta_grid.cpu().numpy()
#         # self.logMBH_grid_max  = self.logMBH_grid.max()

#         self.dlogM = np.diff(self.logMBH_grid, prepend=self.logMBH_grid[0])

#     def pdf(self, **hypers):

#         Nz, Nm = len(self.z_grid), len(self.theta_grid)
#         pdf = np.zeros((Nz, Nm))

#         hypers = self._sanitise_inputs(**hypers)
#         beta = float(hypers["beta"])
#         lambda_alpha = float(hypers["lambda_alpha"])
#         gamma = float(hypers["gamma"])


#         lgalpha_M = ( beta + lambda_alpha * (self.logMBH_grid - 6) ) # shape (Nm,)

#         pdf_dN_dlgMBH_dz = self.dN_dlgMBH_dz.pdf(gamma=gamma) # shape (Nz, Nm)

#         for i, z in enumerate(self.z_grid.cpu().numpy()):

#             # pdf_mass[iz,:] is 1D array over M
#             fM = pdf_dN_dlgMBH_dz[i, :]                # shape (Nm,)

#             # compute beta PDF for all a values for each mass point
#             a = self.a_grid.cpu().numpy()       # shape (Na,)

#             # matrix of shape (Nm, Na)
#             beta_matrix = np.zeros((len(self.logMBH_grid), len(a)))

#             for j, lgalpha_j in enumerate(lgalpha_M):
#                 # Beta(alpha_j, beta) normalization:

#                 B = math.gamma(lgalpha_j) * math.gamma(beta) / math.gamma(lgalpha_j + beta)

#                 beta_matrix[j,:] = (a**(lgalpha_j - 1)) * ((1-a)**(beta - 1)) / B

#             # Convolution in mass dimension:
#             # sum_j [ pdf_dN_dlgMBH_dz[z,j] * beta_pdf[j,a] * dlogM[j] ]
            
#             pdf[i,:] = np.sum(fM[:,None] * beta_matrix * self.dlogM[:,None], axis=0)

#         pdf/=np.sum(pdf)
#         breakpoint()
#         self.interpolate(pdf)
#         return pdf

# class dN_dCO_dz(Distribution2D):

#     def __init__(self, limits_z, limits_theta, limits_MBH, npoints=200, grid_spacing='linear', device="cpu"):
#         super().__init__(limits_z, limits_theta, npoints, grid_spacing, device)
        
#         self.dN_dlgMBH_dz = dN_dlgMBH_dz(limits_z, limits_MBH, npoints, grid_spacing, device)
#         self.CO_mass_grid = self.theta_grid 

#         self.logMBH_grid = self.dN_dlgMBH_dz.theta_grid.cpu().numpy()
#         self.dlogM = np.diff(self.logMBH_grid, prepend=self.logMBH_grid[0])

#     def pdf(self, **hypers):

#         Nz, Nm = len(self.z_grid), len(self.theta_grid)
#         pdf = np.zeros((Nz, Nm))

#         hypers = self._sanitise_inputs(**hypers)
#         gamma = float(hypers["gamma"])

#         pdf_dN_dlgMBH_dz = self.dN_dlgMBH_dz.pdf(gamma=gamma) # shape (Nz, Nm)

#         for i, z in enumerate(self.z_grid.cpu().numpy()):

#             # pdf_mass[iz,:] is 1D array over M
#             fM = pdf_dN_dlgMBH_dz[i, :]                # shape (Nm,)

#             # compute beta PDF for all a values for each mass point
#             mu = self.CO_mass_grid.cpu().numpy()       # shape (Na,)

#             p_mu_given_M = np.zeros((len(self.logMBH_grid), len(mu)))

#             for j, lgMBH in enumerate(self.logMBH_grid):

#                 lgMgal = Galaxy.lgMgal_from_lgMBH(lgMBH)
#                 gal = Galaxy(lgMgal, z)
#                 nsc = NSC(gal, lgMBH)
                
#                 r_max = nsc.r_influence(unit='pc')
#                 r_min = nsc.r_capture(unit='pc')

#                 profile = DehnenProfile(nsc, gamma)

#                 I_r = profile.number_of_CO_within_shell(r_min=r_min, r_max=r_max, Ntot=self.Ntot_EMRI, kind='EMRI', npts=len(self.r_grid))

#                 psi = CO_mass_function().delta_distribution(m=self.CO_mass_grid, M_CO=10.0)
#                 # breakpoint()

#                 p_mu_given_M[j, :] = 1 # psi * I_r / (np.trapezoid(psi, self.CO_mass_grid) * I_r) 

#             # Convolution in mass dimension:
#             # sum_j [ pdf_dN_dlgMBH_dz[z,j] * pdf_mu[j,mu] * dlogM[j] ]
#             pdf[i,:] = np.sum(fM[:,None] * p_mu_given_M * self.dlogM[:,None], axis=0)
#         pdf/=np.sum(pdf)
#         breakpoint()
#         self.interpolate(pdf)
#         return pdf





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

