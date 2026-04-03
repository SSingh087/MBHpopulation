import torch
import numpy as np
import warnings
import matplotlib.pyplot as plt

from cosmology import CosmologyModel, GalaxyStellarMassFunction, MBHMassFunction, LastMajorMerger, CO_mass_function
from galaxy import Galaxy
from nsc import NSC
from density import DehnenProfile
from relaxation import RelaxationModel
from rate import RateModel
from evolution import CuspEvolution

from config import (MBH_A, MBH_B, MBH_sigma0, MBH_scatter)

from utils import Plotting



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
        self.sigma_pc_yr_grid_sorted = self.sigma_pc_yr[sorted_indices]

        print(f"Number of galaxies after sorting: {len(self.z_grid)}")

        # at this stage we have a pair of lgMBH_sorted and z_grid that are consistent with
        # each other, and we can use them to construct the 2D mesh after checking
        # if max and min of MBH grid are within the limits provided by the user

        self.Ntot_EMRI = 1E5 * torch.ones_like(self.z_grid) # this is the total number of COs in the NSC, can be scaled with galaxy properties in the future
        self.component_masses_sBH = torch.full_like(self.z_grid, 10.0) # this is the mass of the sBHs in the NSC, can be scaled with galaxy properties in the future

        self.Ntot_EMRI = self.Ntot_EMRI[sorted_indices]
        self.component_masses_sBH = self.component_masses_sBH[sorted_indices]

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

            # nucleation check is already applied before, so we can set nucleation_occurs=True for all of them.
            self.galaxies_on_grid = Galaxy(lgMgal=self.lgMgal_sorted, lgMBH=self.lgMBH_sorted, z_gal=self.z_grid_sorted, nucleation_occurs=True)

            self.Ntot_EMRI = self.Ntot_EMRI[mask]
            self.component_masses_sBH = self.component_masses_sBH[mask]

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

    def cdf(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError
    
    def interpolate(self, pdf):
        # this method can be used to create an interpolator for the pdf, which can then be used for sampling or evaluating the pdf at arbitrary points.
        # we can use scipy's griddata or RegularGridInterpolator for this purpose, depending on the structure of the grid and the desired interpolation method.
        pass

    def draw_sample(self, num_samples, **kwargs):
        # this method can be used to draw samples from the distribution defined by the pdf. 
        # We can use inverse transform sampling, rejection sampling, or any other appropriate method depending on the structure of the pdf and the grid.
        pass

class dN_dlgMBH_dz(PopulationDistribution):
    def __init__(self, limits_z, limits_theta, limits_MBH, npoints=200, grid_spacing='linear', device="cpu", dtype=torch.float64):
        super().__init__(limits_z, limits_theta, limits_MBH, npoints, grid_spacing, device, dtype)

    @torch.no_grad()
    def pdf(self, X, **hypers) -> torch.Tensor:
        
        z_gal, mbhmass = X
        
        hypers = self._sanitise_inputs(**hypers)
        gamma = float(hypers["gamma"])
        
        z_gal = torch.as_tensor(z_gal, device=self.device, dtype=self.dtype)
        mbhmass = torch.as_tensor(mbhmass, device=self.device, dtype=self.dtype)
        
        # compare closest MBH in the grid to the input theta (lgMBH) and get the corresponding galaxy properties (lgMgal, sigma, etc.) for that grid point.
        # This is necessary because the input theta may not exactly match the grid points, so we need to find the closest one and use its properties for the calculations.
        
        idx = torch.searchsorted(self.lgMBH_sorted, mbhmass)
        idx = torch.clamp(idx, 1, len(self.lgMBH_sorted) - 1)
        
        left = self.lgMBH_sorted[idx - 1]
        right = self.lgMBH_sorted[idx]
        
        choose_left = (torch.abs(mbhmass - left) <= torch.abs(mbhmass - right))
        idx = idx - choose_left.long()

        # galactic parmeters corresponding to the input theta (lgMBH) after checking for nucleation and sorting
        
        lgMgal_derived = self.lgMgal_sorted[idx]
        sigma_pc_yr_derived = self.sigma_pc_yr_grid_sorted[idx]

        # NSC and EMRI parameters corresponding to the input theta (lgMBH) after checking for nucleation and sorting
        Ntot_EMRI = self.Ntot_EMRI[idx]
        component_masses_sBH = self.component_masses_sBH[idx]

        __galaxies__ = Galaxy(lgMgal=lgMgal_derived, lgMBH=mbhmass, sigma_pc_yr=sigma_pc_yr_derived, z_gal=z_gal, nucleation_occurs=True)

        nscs = NSC(__galaxies__, mbhmass)
        profiles = DehnenProfile(nscs, gamma)
        relax_models = RelaxationModel(nscs, profiles)
        rates = RateModel(nscs)
        evol = CuspEvolution(nscs, relax_models, rates, LastMajorMerger(self.cosmo))
        tau, T_c, t_EMRI, Gamma_hat_EMRI = evol.evaluate_tau(Ntot=Ntot_EMRI, component_masses=component_masses_sBH, kvir=1.0, kind='EMRI', mbar=10., unit='Gyr', A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter)

        Gamma = torch.tensor(Gamma_hat_EMRI * rates.universal_EMRI_rate(tau))

        T_obs = self.T_obs_det / (1 + z_gal)  

        phi_linear = torch.tensor(self._mbhmf.eval_mbhmf(z=z_gal, logMBH=mbhmass, n_points_mass=len(mbhmass), return_log10=False))
        phi_linear = torch.nan_to_num(phi_linear, nan=0.0, posinf=0.0, neginf=0.0) # 
        
        dVc_dz = torch.tensor(self.cosmo.dVc_dz(z_gal))

        pdf = dVc_dz * phi_linear * Gamma * T_obs

        total = torch.sum(pdf)

        if (not torch.isfinite(total)) or (total <= 0):
            raise RuntimeError("Computed PDF is zero everywhere. Check inputs/units.")

        pdf = pdf / total
        # self.interpolate(pdf)
        return pdf


class dN_da_dz(PopulationDistribution):

    def __init__(self, limits_z, limits_theta, limits_MBH, npoints=200, grid_spacing='linear', device="cpu"):
        super().__init__(limits_z, limits_theta, limits_MBH, npoints, grid_spacing, device)
        self._dN_dlgMBH_dz = dN_dlgMBH_dz(limits_z, limits_theta, limits_MBH, npoints, grid_spacing, device)

    @torch.no_grad()
    def pdf(self, X, **hypers) -> torch.Tensor:

        z_gal, mbhspin, mbhmass = X

        z_gal = torch.as_tensor(z_gal, device=self.device, dtype=self.dtype)
        mbhspin = torch.as_tensor(mbhspin, device=self.device, dtype=self.dtype)
        mbhmass = torch.as_tensor(mbhmass, device=self.device, dtype=self.dtype)

        hypers = self._sanitise_inputs(**hypers)
        beta = hypers["beta"]
        if "alpha" not in hypers:
            lambda_alpha = hypers["lambda_alpha"]
            alpha_M = ( beta + lambda_alpha * (mbhmass - 6) )
            # alpha = 10**lgalpha_M
        else:
            alpha_M = hypers["alpha"] * torch.ones_like(mbhmass)

        gamma = hypers["gamma"]

        B = torch.exp(torch.lgamma(alpha_M) + torch.lgamma(beta) - torch.lgamma(alpha_M + beta))
        beta_matrix = (mbhspin[:, None] ** (alpha_M[None, :] - 1)) * ((1 - mbhspin[:, None]) ** (beta - 1)) / B[None, :]

        pdf_dN_dlgMBH_dz = self._dN_dlgMBH_dz.pdf(X=(z_gal, mbhmass), gamma=gamma)

        dlogM = torch.diff(mbhmass, prepend=mbhmass[:1].clone())

        # Final contraction over mass dimension
        pdf = torch.sum(pdf_dN_dlgMBH_dz[None, :] * beta_matrix * dlogM[None, :], dim=1)
        pdf/=torch.sum(pdf)

        return pdf


class dN_dCO_dz(PopulationDistribution):

    def __init__(self, limits_z, limits_theta, limits_MBH, npoints=200, grid_spacing='linear', device="cpu"):
        super().__init__(limits_z, limits_theta, limits_MBH, npoints, grid_spacing, device)
        self._dN_dlgMBH_dz = dN_dlgMBH_dz(limits_z, limits_theta, limits_MBH, npoints, grid_spacing, device)
        self.CO_mass_function = CO_mass_function()
        
    @torch.no_grad()
    def pdf(self, X, **hypers) -> torch.Tensor:

        z_gal, co_mass, mbhmass = X

        z_gal = torch.as_tensor(z_gal, device=self.device, dtype=self.dtype)
        co_mass = torch.as_tensor(co_mass, device=self.device, dtype=self.dtype)
        mbhmass = torch.as_tensor(mbhmass, device=self.device, dtype=self.dtype)

        hypers = self._sanitise_inputs(**hypers)
        gamma = hypers["gamma"]
        
        pdf_dN_dlgMBH_dz = self._dN_dlgMBH_dz.pdf(X=(z_gal, mbhmass), gamma=gamma)

        idx = torch.searchsorted(self.lgMBH_sorted, mbhmass)
        idx = torch.clamp(idx, 1, len(self.lgMBH_sorted) - 1)
        
        left = self.lgMBH_sorted[idx - 1]
        right = self.lgMBH_sorted[idx]
        
        choose_left = (torch.abs(mbhmass - left) <= torch.abs(mbhmass - right))
        idx = idx - choose_left.long()

        # galactic parmeters corresponding to the input theta (lgMBH) after checking for nucleation and sorting
        
        lgMgal_derived = self.lgMgal_sorted[idx]
        sigma_pc_yr_derived = self.sigma_pc_yr_grid_sorted[idx]

        # NSC and EMRI parameters corresponding to the input theta (lgMBH) after checking for nucleation and sorting
        Ntot_EMRI = self.Ntot_EMRI[idx]
        component_masses_sBH = self.component_masses_sBH[idx]

        __galaxies__ = Galaxy(lgMgal=lgMgal_derived, lgMBH=mbhmass, sigma_pc_yr=sigma_pc_yr_derived, z_gal=z_gal, nucleation_occurs=True)

        nscs = NSC(__galaxies__, mbhmass)

        r_maxs = nscs.r_influence(unit='pc')
        r_mins = nscs.r_capture(unit='pc')

        profiles = DehnenProfile(nscs, gamma)

        I_rs = profiles.number_of_CO_within_shell(r_min=r_mins, r_max=r_maxs, Ntot=Ntot_EMRI, kind='EMRI', npts=2000)
        
        if torch.all(torch.diff(co_mass) == 0):
            co_mf = torch.ones_like(co_mass, device=self.device, dtype=self.dtype)
            pdf_mu = co_mf / co_mf.sum()     # trivial delta
        else:
            # Continuous mass grid
            psi_np = self.CO_mass_function.mass_distribution(
                m=co_mass.cpu().numpy(),
                kind="kroupa"
            )
            psi = torch.tensor(psi_np, device=self.device, dtype=self.dtype)

            psi_I = psi[None, :] * I_rs[:, None]
            norm = torch.trapz(psi_I, co_mass[None, :], dim=1)
            pdf_mu = psi_I / norm[:, None]


        pdf_dN_dlgMBH_dz = self._dN_dlgMBH_dz.pdf(X=(z_gal, mbhmass), gamma=gamma)
        dlogM = torch.diff(mbhmass, prepend=mbhmass[:1].clone())

        # Final contraction over mass dimension
        pdf = torch.sum(pdf_dN_dlgMBH_dz[None, :] * pdf_mu[:, None] * dlogM[None, :], dim=1)
        pdf/=torch.sum(pdf)
        return pdf


N_objs = 50

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

dist_dN_dlgMBH_dz = dN_dlgMBH_dz(limits_z=(z_gal[0], z_gal[-1]), limits_theta=(lgMBH_mass_from_galaxies.min(), lgMBH_mass_from_galaxies.max()), limits_MBH=(lgMBH_mass_from_galaxies.min(), lgMBH_mass_from_galaxies.max()), npoints=500, grid_spacing='linear', device="cpu")
pdf_dN_dlgMBH_dz = dist_dN_dlgMBH_dz.pdf(X=(z_gal, lgMBH_mass_from_galaxies), gamma=1.5)

MBHspins = torch.tensor(np.random.uniform(0, 1, size=N_objs)[nucleation_indices])
dist_dN_da_dz = dN_da_dz(limits_z=(z_gal[0], z_gal[-1]), limits_theta=(MBHspins.min(), MBHspins.max()), limits_MBH=(lgMBH_mass_from_galaxies.min(), lgMBH_mass_from_galaxies.max()), npoints=100, grid_spacing='linear', device="cpu")
pdf_dN_da_dz = dist_dN_da_dz.pdf(X=(z_gal, MBHspins, lgMBH_mass_from_galaxies), gamma=1.5, lambda_alpha=0.5, beta=12.0)

CO_masses = torch.tensor(np.full_like(z_gal, 10))
dist_dN_dCO_dz = dN_dCO_dz(limits_z=(z_gal[0], z_gal[-1]), limits_theta=(CO_masses.min(), CO_masses.max()), limits_MBH=(lgMBH_mass_from_galaxies.min(), lgMBH_mass_from_galaxies.max()), npoints=100, grid_spacing='linear', device="cpu")
pdf_dN_dCO_dz = dist_dN_dCO_dz.pdf(X=(z_gal, CO_masses, lgMBH_mass_from_galaxies), gamma=1.5)


Plotting.plot_joint_with_marginals(z_gal, lgMBH_mass_from_galaxies, pdf_dN_dlgMBH_dz, theta_label=r"\log_{10}(M_{\rm BH}/M_\odot)", smooth=True, cmap="magma")
plt.tight_layout()
plt.savefig("dN_dlgMBH_dz.pdf", dpi=300)
plt.show()


Plotting.plot_joint_with_marginals(z_gal, MBHspins, pdf_dN_da_dz, theta_label=r"a", smooth=True, cmap="magma")
plt.tight_layout()
plt.savefig("dN_da_dz.pdf", dpi=300)
plt.show()

Plotting.plot_joint_with_marginals(z_gal, CO_masses, pdf_dN_dCO_dz, theta_label=r"\mu", smooth=True, cmap="magma")
plt.tight_layout()
plt.savefig("dN_dCO_dz.pdf", dpi=300)
plt.show()

# Plotting.plot_joint_with_marginals(z_gal, lgMBH_mass_from_galaxies, pdf_dN_dlgMBH_dz, smooth=True, cmap="magma")
# plt.tight_layout()
# plt.savefig("dN_dlgMBH_dz.pdf", dpi=300)
# plt.show()