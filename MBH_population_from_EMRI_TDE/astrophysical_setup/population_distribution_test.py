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


class Distribution2D:

    def __init__(self, device="cpu", dtype=torch.float64):
        self.device = device
        self.dtype = dtype

        # The theta grid can be either linear or logarithmic depending on the user's choice.
        # if grid_spacing == "linear":
        #     self.theta_grid = torch.linspace(limits_theta[0], limits_theta[1], npoints, device=device, dtype=dtype)
        # else:
        #     self.theta_grid = torch.logspace(
        #         torch.log10(torch.tensor(limits_theta[0], device=device, dtype=dtype)),
        #         torch.log10(torch.tensor(limits_theta[1], device=device, dtype=dtype)),
        #         npoints, device=device, dtype=dtype
        #     )

        self.cosmo = CosmologyModel()
        self._gsmf = GalaxyStellarMassFunction()
        self._mbhmf = MBHMassFunction(gsmf=self._gsmf)

        # self.Ntot_EMRI = 1E5 * torch.ones_like(self.theta_grid) # this is the total number of COs in the NSC, can be scaled with galaxy properties in the future
        # self.component_masses_sBH = torch.full_like(self.theta_grid, 10.0) # this is the mass of the sBHs in the NSC, can be scaled with galaxy properties in the future

        self.T_obs_det = torch.tensor(4.0)

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


class dN_dlgMBH_dz(Distribution2D):
    def __init__(self, device="cpu", dtype=torch.float64):
        super().__init__(device, dtype)

    @torch.no_grad()
    def pdf(self, X, **hypers) -> torch.Tensor:
        
        z_gal, theta = X

        self.z_gal = torch.as_tensor(z_gal, device=self.device, dtype=self.dtype)
        self.theta = torch.as_tensor(theta, device=self.device, dtype=self.dtype)

        hypers = self._sanitise_inputs(**hypers)
        gamma = float(hypers["gamma"])


        # print(self.z_gal, self.theta)

        lgMgal = Galaxy.lgMgal_from_lgMBH(self.theta, A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0)

        # all MBHs passed here should have a nucleated galaxy,
        # since the data is generated from the galaxy population
        # with nucleation flags. So we can set
        # nucleation_occurs=True for all of them.
        galaxies = Galaxy(lgMgal, self.z_gal, nucleation_occurs=True) 
        print(galaxies.lgMgal)
        breakpoint()



N_objs = 10

z_gal = torch.tensor(np.random.uniform(0.01, 10, size=N_objs))

GSMF = GalaxyStellarMassFunction()
lgMgal_samples = GSMF.sample_gsmf(z_gal=z_gal, size=N_objs)
nucleation_indices = Galaxy.check_nucleation(lgMgal_samples, z_gal)

galaxies = Galaxy(lgMgal=lgMgal_samples, z_gal=z_gal, nucleation_occurs=nucleation_indices)
lgMBH_mass_from_galaxies = torch.tensor(galaxies.lgMBH_mass(A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=0.0)[nucleation_indices])

print(lgMgal_samples[nucleation_indices])

dist_dN_dlgMBH_dz = dN_dlgMBH_dz(device="cpu")
pdf_dN_dlgMBH_dz = dist_dN_dlgMBH_dz.pdf(X=(z_gal, lgMBH_mass_from_galaxies), gamma=1.5)


breakpoint()









nscs = NSC(galaxies, lgMBH_sorted)
profiles = DehnenProfile(nscs, gamma)
relax_models = RelaxationModel(nscs, profiles)
rates = RateModel(nscs)
evol = CuspEvolution(nscs, relax_models, rates, LastMajorMerger(self.cosmo))
tau, T_c, t_EMRI, Gamma_hat_EMRI = evol.evaluate_tau(Ntot=self.Ntot_EMRI, component_masses=self.component_masses_sBH, kvir=1.0, kind='EMRI', mbar=10., unit='Gyr', A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter)

Gamma = torch.tensor(Gamma_hat_EMRI * rates.universal_EMRI_rate(tau)) # (Nz,) array of EMRI rates at the current time for each (z, lgMBH) pair

T_obs = self.T_obs_det / (1 + self.z_grid)  # (Nz,)
NSC_obj = NSC(Galaxy(lgMgal=self.theta, z_gal=self.z_gal, nucleation_occurs=True), self.theta)
