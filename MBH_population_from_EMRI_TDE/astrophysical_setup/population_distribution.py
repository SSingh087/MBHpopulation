import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

from cosmology import CosmologyModel, GalaxyStellarMassFunction, MBHMassFunction, LastMajorMerger
from galaxy import Galaxy
from nsc import NSC
from density import DehnenProfile
from relaxation import RelaxationModel
from rate import RateModel
from evolution import CuspEvolution

# (Place above Distribution2D in the same file)
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

ArrayLike = Union[float, np.ndarray]

@dataclass
class PopulationCache:
    Nz: int
    Nm: int
    seed: Optional[int] = None
    rng: np.random.Generator = field(init=False)

    # cached arrays for EMRI and TDE populations; set via setters, used in pdf evaluation
    Ntot_emri: Optional[np.ndarray] = None        # (Nz,Nm)
    masses_emri: Optional[np.ndarray] = None      # (Nz,Nm,S)
    Ntot_tde: Optional[np.ndarray] = None         # (Nz,Nm)
    masses_tde: Optional[np.ndarray] = None       # (Nz,Nm,S)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def _broadcast_N(self, N: ArrayLike) -> np.ndarray:
        N = np.asarray(N, dtype=float)
        if N.ndim == 0:
            # If N is a scalar (e.g., 1e5), fill the full (Nz, Nm) grid with that value.
            # np.full((Nz, Nm), N) → each redshift and galaxy bin gets the same total number
            return np.full((self.Nz, self.Nm), N, dtype=float)
        
        if N.shape == (self.Nm,):
            # If N is 1D with shape (Nm,) → one value per galaxy, same across all redshifts.
            # N[None, :] adds a new axis for redshift → (1, Nm)
            # np.tile(..., (Nz, 1)) repeats this along the redshift axis → (Nz, Nm)
            return np.tile(N[None, :], (self.Nz, 1))
        
        if N.shape == (self.Nz,):
            # If N is 1D with shape (Nz,) → one value per redshift, same across all galaxies.
            # N[:, None] adds a new axis for galaxy → (Nz, 1)
            # np.tile(..., (1, Nm)) repeats this along the galaxy axis → (Nz, Nm)
            return np.tile(N[:, None], (1, self.Nm))
        
        if N.shape == (self.Nz, self.Nm):
            # If N is already 2D with shape (Nz, Nm), we can use it directly.
            return N
        raise ValueError(f"Ntot has incompatible shape {N.shape}; expected scalar, (Nm,), (Nz,), or (Nz,Nm).")

    def _broadcast_masses(self, M: Optional[ArrayLike], n_species: int, mass_range: Tuple[float, float]) -> np.ndarray:
        
        S = max(1, int(n_species)) # number of species (e.g. MS, WD, NS, sBH, etc.)
        
        if M is None:
            # If no mass array is provided, sample random masses uniformly over mass_range for every redshift, galaxy, and species
            low, high = mass_range
            return self.rng.uniform(low, high, size=(self.Nz, self.Nm, S))

        M = np.asarray(M, dtype=float)
        
        if M.ndim == 0:
            # Scalar mass → broadcast to all redshifts, galaxies, species
            return np.full((self.Nz, self.Nm, 1), float(M))
        if M.ndim == 1:
            # (S,)
            if M.size != S:
                # if size differs, just tile provided vector (more flexible)
                return np.tile(M[None, None, :], (self.Nz, self.Nm, 1))
            return np.tile(M[None, None, :], (self.Nz, self.Nm, 1))
        if M.ndim == 2:
            # (Nm,S) or (Nz,S)
            if M.shape[0] == self.Nm:
                return np.tile(M[None, :, :], (self.Nz, 1, 1))
            if M.shape[0] == self.Nz:
                return np.tile(M[:, None, :], (1, self.Nm, 1))
            raise ValueError(f"2-D masses must be (Nm,S) or (Nz,S); got {M.shape}.")
        if M.ndim == 3:
            if M.shape[:2] != (self.Nz, self.Nm):
                raise ValueError(f"3-D masses must start with (Nz,Nm, S); got {M.shape}.")
            return M
        raise ValueError(f"component_masses has incompatible ndim={M.ndim}.")

    def set_emri(self, Ntot: ArrayLike = 1.0e5, component_masses: Optional[ArrayLike] = None,
                 n_species: int = 1, mass_range: Tuple[float, float] = (10.0, 10.0)):
        self.Ntot_emri = self._broadcast_N(Ntot)
        self.masses_emri = self._broadcast_masses(component_masses, n_species, mass_range)
        return self

    def set_tde(self, Ntot: ArrayLike = 1.0e5, component_masses: Optional[ArrayLike] = None,
                n_species: int = 1, mass_range: Tuple[float, float] = (1.0, 1.0)):
        self.Ntot_tde = self._broadcast_N(Ntot)
        self.masses_tde = self._broadcast_masses(component_masses, n_species, mass_range)
        return self

    def get_emri_flat(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.Ntot_emri is None or self.masses_emri is None:
            self.set_emri()
        N_flat = self.Ntot_emri.reshape(-1)               # (N_flat,)
        M_flat = self.masses_emri.reshape(self.Nz*self.Nm, -1)  # (N_flat,S)
        mbar_flat = M_flat.mean(axis=1)                    # (N_flat,)
        return N_flat, M_flat, mbar_flat

    def get_tde_flat(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.Ntot_tde is None or self.masses_tde is None:
            self.set_tde()
        N_flat = self.Ntot_tde.reshape(-1)
        M_flat = self.masses_tde.reshape(self.Nz*self.Nm, -1)
        mbar_flat = M_flat.mean(axis=1)
        return N_flat, M_flat, mbar_flat

    def reseed(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)


class Distribution2D:
    """
    Base class: builds grid, provides vectorized helpers for:
      - mesh/flatten/reshape
      - MBH mass function φ(z, logMBH) on the current grid
      - EMRI rate Γ(z, logMBH) on the current grid
      - cosmology terms (dVc/dz, T_obs)
      - interpolation + sampling
    Derived classes should only combine these pieces to produce a PDF on (z, θ).
    """

    def __init__(self, limits_z, limits_theta, npoints=200, grid_spacing="linear", device="cpu"):
        self.device = device

        if grid_spacing == "linear":
            self.z_grid = torch.linspace(limits_z[0], limits_z[1], npoints, device=device)
            self.theta_grid = torch.linspace(limits_theta[0], limits_theta[1], npoints, device=device)
        else:
            self.z_grid = torch.logspace(torch.log10(torch.tensor(limits_z[0], device=device)),
                                         torch.log10(torch.tensor(limits_z[1], device=device)),
                                         npoints, device=device)
            self.theta_grid = torch.logspace(torch.log10(torch.tensor(limits_theta[0], device=device)),
                                             torch.log10(torch.tensor(limits_theta[1], device=device)),
                                             npoints, device=device)

        self.z_grid = self.z_grid.detach()
        self.theta_grid = self.theta_grid.detach()


        # ---- Torch mesh ----
        self.Z, self.TH = torch.meshgrid(self.z_grid, self.theta_grid, indexing='ij')  # (Nz,Nm)
        self.Nz, self.Nm = self.Z.shape
        self.N_flat = self.Nz * self.Nm

        # ---- NumPy views ----
        self.z_np = self.z_grid.cpu().numpy()
        self.theta_np = self.theta_grid.cpu().numpy()
        self.Z_np = self.Z.cpu().numpy()
        self.TH_np = self.TH.cpu().numpy()

        self.cosmo = CosmologyModel()
        self._gsmf = GalaxyStellarMassFunction()
        self._mbhmf = MBHMassFunction(gsmf=self._gsmf)

        self.T_obs_det = 4.0

        # we will need to cache the population inputs since the calculation is
        # expensive and we want to allow flexible input shapes (scalar, vector, or full grid)
        self.population_cache = PopulationCache(self.Nz, self.Nm, seed=None)

        # Interpolator cache for any derived PDF (e.g. dN/dlogMBH/dz or dN/da/dz) that we want to evaluate continuously
        self.pdf_grid = None
        self.interp = None

    def set_emri_population(self, Ntot=1.0e5, component_masses=None, n_species=1, mass_range=(10.0, 10.0), seed=None):
        if seed is not None:
            self.population_cache.reseed(seed)
        self.population_cache.set_emri(Ntot=Ntot, component_masses=component_masses, n_species=n_species, mass_range=mass_range)
        return self

    def set_tde_population(self, Ntot=1.0e5, component_masses=None, n_species=1, mass_range=(1.0, 1.0), seed=None):
        if seed is not None:
            self.population_cache.reseed(seed)
        self.population_cache.set_tde(Ntot=Ntot, component_masses=component_masses, n_species=n_species, mass_range=mass_range)
        return self

    def mesh(self):
        """Return (Z, TH) mesh with shape (Nz, Nm)."""
        return self.Z, self.TH

    def mesh_np(self):     # NumPy
        return self.Z_np, self.TH_np

    def flatten(self, A2):
        """(Nz, Nm) → (Nz*Nm,)"""
        return np.asarray(A2).reshape(-1)

    def unflatten(self, A1):
        """(Nz*Nm,) → (Nz, Nm)"""
        return np.asarray(A1).reshape(self.Nz, self.Nm)

    def dVc_dz_on_grid(self):
        """Return dVc/dz(z) broadcast to (Nz, 1) for safe multiplication."""
        return self.cosmo.dVc_dz(self.z_grid)[:, None]  # (Nz,1)

    def T_obs_on_grid(self):
        """Return observer-frame time T_obs(z) on the mesh (Nz, Nm)."""
        Z, _ = self.mesh()
        return self.T_obs_det / (1.0 + Z)

    def phi_MBH_on_grid(self, n_points_mass=512):
        """NumPy array (Nz,Nm): φ_MBH(z,logMBH) in linear space."""
        Z, TH = self.mesh_np()
        return self._mbhmf.evaluate_on_mesh(Z, TH, n_points_mass=n_points_mass)

    def emri_rate_on_grid(self, gamma=1.5, kind='EMRI', kvir=1.0, mbar=None,
                          Ntot=None, component_masses=None, unit='Gyr'):
        Z, TH = self.mesh_np()
        z_flat = Z.reshape(-1)          # (N_flat,)
        lgMBH_flat = TH.reshape(-1)     # (N_flat,)
        lgMgal_flat = Galaxy.lgMgal_from_lgMBH(lgMBH_flat)

        gal = Galaxy(lgMgal_flat, z_flat)
        nsc = NSC(gal, lgMBH_flat)
        prof = DehnenProfile(nsc, gamma)
        relax = RelaxationModel(nsc, prof)
        rate = RateModel(nsc)
        evol = CuspEvolution(nsc, relax, rate, LastMajorMerger(self.cosmo))

        # ---- population inputs (prefer explicit overrides; else cache) ----
        if (Ntot is None) or (component_masses is None):
            Ntot_flat, masses_flat, mbar_flat_from_masses = self.population_cache.get_emri_flat()  # (N_flat,), (N_flat,S), (N_flat,)
        else:
            # normalize explicit overrides
            Ntot_flat = np.asarray(Ntot).reshape(-1)
            if Ntot_flat.size == 1:
                Ntot_flat = np.full(self.N_flat, float(Ntot_flat))
            masses = np.asarray(component_masses)

            if masses.ndim == 0:
                masses_flat = np.full((self.N_flat, 1), float(masses))
            elif masses.ndim == 1:
                masses_flat = np.tile(masses[None, :], (self.N_flat, 1))
            elif masses.ndim == 2 and masses.shape[0] == self.N_flat:
                masses_flat = masses
            else:
                raise ValueError("component_masses must be scalar, (S,), or (N_flat,S).")
            mbar_flat_from_masses = masses_flat.mean(axis=1)

        # choose mbar: explicit scalar/array overrides take precedence
        if mbar is None:
            mbar_vec = mbar_flat_from_masses            # (N_flat,)
        else:
            mbar_arr = np.asarray(mbar)
            if mbar_arr.ndim == 0:
                mbar_vec = np.full(self.N_flat, float(mbar_arr))
            else:
                if mbar_arr.size != self.N_flat:
                    raise ValueError("mbar array must be length N_flat when provided.")
                mbar_vec = mbar_arr

        # ---- evaluate τ, T_c, t_EMRI, Γ_hat with full population ----
        tau, T_c, t_EMRI, Gamma_hat = evol.evaluate_tau(
            Ntot=Ntot_flat,
            component_masses=masses_flat,  # (N_flat,S) or (N_flat,1)
            kvir=kvir, kind=kind, mbar=mbar_vec, unit=unit
        )

        if kind.upper() == 'EMRI':
            Gamma_flat = Gamma_hat * rate.universal_EMRI_rate(tau)
        else:
            Gamma_flat = Gamma_hat * rate.universal_TDE_rate(tau)

        return Gamma_flat.reshape(self.Nz, self.Nm)

    def interpolate(self, pdf_2d_normalized: np.ndarray):
        self.pdf_grid = pdf_2d_normalized
        self.interp = RegularGridInterpolator(
            (self.z_np, self.theta_np),
            pdf_2d_normalized,
            bounds_error=False,
            fill_value=0.0
        )

    def evaluate_at_z_theta(self, z, theta):
        if self.interp is None:
            raise RuntimeError("Call .pdf(...) once before evaluating.")
        z = np.atleast_1d(np.asarray(z, dtype=float))
        th = np.atleast_1d(np.asarray(theta, dtype=float))
        pts = np.stack([z, th], axis=1)
        return self.interp(pts)

    def cdf(self, **pdf_kwargs):
        if self.pdf_grid is None:
            self.pdf_grid = self.pdf(**pdf_kwargs)
        flat = self.pdf_grid.ravel()
        c = np.cumsum(flat)
        total = c[-1]
        if not np.isfinite(total) or total <= 0:
            raise RuntimeError("PDF integrates to zero; cannot sample.")
        return c / total

    def draw_samples(self, size, **pdf_kwargs):
        cdf = self.cdf(**pdf_kwargs)
        u = np.random.rand(size)
        idx = np.searchsorted(cdf, u)
        iz = idx // self.Nm
        it = idx % self.Nm
        z_s = torch.as_tensor(self.z_np[iz], device=self.device)
        th_s = torch.as_tensor(self.theta_np[it], device=self.device)
        return z_s, th_s

    def pdf(self, **kwargs):
        raise NotImplementedError


class dN_dlgMBH_dz(Distribution2D):
    def __init__(self, limits_z, limits_theta, npoints=200, grid_spacing='linear', device="cpu"):
        super().__init__(limits_z, limits_theta, npoints, grid_spacing, device)

    def pdf(self, gamma=1.5, kind='EMRI', kvir=1.0, mbar=None, n_points_mass=512):
        dVc  = self.dVc_dz_on_grid()                      # (Nz,1)
        phi  = self.phi_MBH_on_grid(n_points_mass)        # (Nz,Nm)
        Gamma = self.emri_rate_on_grid(gamma=gamma, kind=kind, kvir=kvir, mbar=mbar)  # (Nz,Nm)
        Tobs = self.T_obs_on_grid()                       # (Nz,Nm)
        breakpoint()
        pdf = dVc * phi * Gamma * Tobs
        total = np.sum(pdf)
        if not np.isfinite(total) or total <= 0:
            raise RuntimeError("Computed PDF is zero everywhere. Check inputs/units.")
        pdf /= total
        self.interpolate(pdf)
        return pdf

distribution = Distribution2D((0.01, 10.0), (4.0, 8.5), npoints=10, device='cpu')


# One species everywhere
dist = dN_dlgMBH_dz((0.01, 10.0), (4.0, 8.5), npoints=10, device='cpu')
dist.set_emri_population(Ntot=1.0e5, component_masses=10.0, n_species=1)  # (Nz,Nm,1) after broadcast
pdf = dist.pdf(gamma=1.5)

# # Three species everywhere, per‑species masses fixed
# species = np.array([5.0, 10.0, 30.0])   # (S,)
# dist.set_emri_population(Ntot=1.0e5, component_masses=species, n_species=species.size)

# # Fully per‑grid masses with 3 species (different at each cell)
# Nz, Nm = dist.Nz, dist.Nm
# Ntot_grid = 1.0e5 * np.ones((Nz, Nm))
# masses_grid = np.random.uniform(1.0, 50.0, size=(Nz, Nm, 3))
# dist.set_emri_population(Ntot=Ntot_grid, component_masses=masses_grid)

# # Override mbar explicitly
# mbar_vec = np.full(dist.N_flat, 8.5)  # (N_flat,)
# pdf = dist.pdf(gamma=1.5, mbar=mbar_vec)


breakpoint()



# class dN_dlgMBH_dz(Distribution2D):

#     def __init__(self, limits_z, limits_theta, npoints=200, grid_spacing='linear', device="cpu"):
#         super().__init__(limits_z, limits_theta, npoints, grid_spacing, device)


#     def pdf(self, **hypers):

#         lgMBH = self.theta_grid.cpu().numpy()
#         Nz, Nm = len(self.z_grid), len(self.theta_grid)
#         pdf = np.zeros((Nz, Nm))

#         hypers = self._sanitise_inputs(**hypers)
#         gamma = float(hypers["gamma"])

#         dVc_dz = self.cosmo.dVc_dz(self.z_grid)

#         _, dlogMBH_dlogMgal = self.MBHMF.get_mbhmf(z_gal=self.z_grid)

#         dlogMBH_dlogMgal_linear = 10.0**dlogMBH_dlogMgal

#         lgMgal = Galaxy.lgMgal_from_lgMBH(lgMBH)

#         gal = Galaxy(lgMgal, self.z_grid)
#         nsc = NSC(gal, lgMBH)

#         profile = DehnenProfile(nsc, gamma)
#         relax = RelaxationModel(nsc, profile)
#         rate = RateModel(nsc)
#         evol = CuspEvolution(nsc, relax, rate, LastMajorMerger(self.cosmo))

#         tau, T_c, t_EMRI, Gamma_hat_EMRI = self.evaluate_tau(
#             Ntot=self.Ntot_EMRI, component_masses=evol.component_masses_sBH, kvir=kvir,
#             kind=kind, mbar=mbar, unit=unit, A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter
#         )

#         Gamma_EMRI = Gamma_hat_EMRI * rate.universal_EMRI_rate(tau)

#         T_obs = self.T_obs_det / (1 + z)

#         breakpoint()

#         for i, z in enumerate(self.z_grid.cpu().numpy()):

            

#             for j, lgMBH in enumerate(self.theta_grid.cpu().numpy()):

#                 lgMgal = Galaxy.lgMgal_from_lgMBH(lgMBH)

#                 pdf[i, j] = dVc_dz * phi[j] * Gamma * T_obs
#                 # print(i, j, z, lgMBH, dVc_dz, phi[j], pdf[i, j])
        
#         pdf /= np.sum(pdf)
#         self.interpolate(pdf)
#         return pdf

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





# dist_dN_dlgMBH_dz = dN_dlgMBH_dz(limits_z=(0.001, 10), limits_theta=(4, 8.5), npoints=5, grid_spacing='linear', device="cpu")
# pdf_dN_dlgMBH_dz = dist_dN_dlgMBH_dz.pdf(gamma=1.5)

# z_samp, mbh_samp = dist_dN_dlgMBH_dz.draw_samples(50)

# plt.figure(figsize=(7,6))
# plt.imshow(pdf_dN_dlgMBH_dz.T, origin='lower',
#            extent=[dist_dN_dlgMBH_dz.z_grid[0], dist_dN_dlgMBH_dz.z_grid[-1], dist_dN_dlgMBH_dz.theta_np[0], dist_dN_dlgMBH_dz.theta_np[-1]],
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
#            extent=[dist_dN_da_dz.z_grid[0], dist_dN_da_dz.z_grid[-1], dist_dN_da_dz.theta_np[0], dist_dN_da_dz.theta_np[-1]],
#            aspect='auto', cmap='viridis')
# plt.colorbar(label=r'$d^2N/da\,dz$')
# plt.xlabel('z')
# plt.ylabel(r'$a$')
# plt.title('EMRI 2D PDF')
# plt.savefig('dN_da_dz.pdf', dpi=200)
# plt.show()

# dN_dCO_dz = dN_dCO_dz(limits_z=(0.001, 10), limits_theta=(10-1E-7, 10+1E-7), limits_MBH=(4, 8.5), npoints=5, grid_spacing='linear', device="cpu")
# pdf_dN_dCO_dz = dN_dCO_dz.pdf(gamma=1.5)
