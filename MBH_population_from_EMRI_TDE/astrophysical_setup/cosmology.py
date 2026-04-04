import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Dict, Any
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy.interpolate import RegularGridInterpolator
from galaxy import Galaxy
from config import (MBH_A, MBH_B, MBH_sigma0)

class HaloMassFunction:
    pass

class GalaxyStellarMassFunction:
    # https://arxiv.org/abs/2509.07960

    def __init__(self):
        self.lgMgal_data = np.array([
                6.5,6.7,6.9,7.1,7.3,7.5,7.7,7.9,8.1,8.3,
                8.5,8.7,8.9,9.1,9.3,9.5,9.7,9.9,10.1,10.3,
                10.5,10.7,10.9,11.1,11.3,11.5,11.7,11.9,12.1
            ])

        self.gsmf_data = {}
    
        # -------- TABLE 1 (z = 0 → 7) --------
        self.gsmf_data[0.0] = self.clean([-0.28,-0.53,-0.73,-0.90,-1.06,-1.21,-1.33,-1.43,-1.53,-1.63,-1.72,-1.80,-1.86,-1.91,-1.94,-1.99,-2.02,-2.05,-2.07,-2.13,-2.24,-2.43,-2.70,-3.00,-3.32,-3.72,-4.23,-4.82,-6.20])
        self.gsmf_data[0.1] = self.clean([-0.26,-0.51,-0.71,-0.88,-1.04,-1.19,-1.32,-1.42,-1.52,-1.62,-1.71,-1.79,-1.85,-1.90,-1.95,-1.98,-2.02,-2.05,-2.08,-2.13,-2.25,-2.44,-2.70,-2.99,-3.35,-3.73,-4.26,-4.88,-5.73])
        self.gsmf_data[0.5] = self.clean([-0.18,-0.43,-0.64,-0.82,-0.98,-1.13,-1.26,-1.37,-1.48,-1.58,-1.68,-1.77,-1.83,-1.88,-1.91,-1.96,-2.00,-2.05,-2.12,-2.19,-2.29,-2.46,-2.72,-3.05,-3.44,-3.87,-4.45,-5.36,-6.20])
        self.gsmf_data[1]   = self.clean([-0.05,-0.33,-0.55,-0.74,-0.91,-1.06,-1.20,-1.32,-1.44,-1.55,-1.66,-1.75,-1.82,-1.86,-1.91,-1.96,-2.01,-2.09,-2.17,-2.26,-2.38,-2.54,-2.81,-3.16,-3.58,-4.07,-4.66,-5.30,-5.90])
        self.gsmf_data[2]   = self.clean([0.19,-0.08,-0.35,-0.59,-0.79,-0.97,-1.12,-1.27,-1.40,-1.53,-1.65,-1.74,-1.82,-1.89,-1.96,-2.05,-2.15,-2.26,-2.38,-2.51,-2.65,-2.85,-3.09,-3.43,-3.81,-4.22,-4.81,-5.36,-6.20])
        self.gsmf_data[3]   = self.clean([0.29,0.01,-0.30,-0.58,-0.81,-1.00,-1.18,-1.34,-1.48,-1.63,-1.74,-1.85,-1.94,-2.04,-2.16,-2.28,-2.43,-2.56,-2.70,-2.87,-3.05,-3.29,-3.55,-3.93,-4.38,-4.88,-5.36,-6.20,-6.20])
        self.gsmf_data[4]   = self.clean([0.28,-0.01,-0.34,-0.65,-0.90,-1.12,-1.31,-1.48,-1.65,-1.79,-1.92,-2.05,-2.17,-2.31,-2.46,-2.62,-2.76,-2.94,-3.10,-3.30,-3.53,-3.77,-4.10,-4.51,-4.88,-5.73,-6.20,-6.20,None])
        self.gsmf_data[5]   = self.clean([0.19,-0.09,-0.44,-0.77,-1.06,-1.30,-1.51,-1.70,-1.87,-2.03,-2.17,-2.32,-2.47,-2.64,-2.81,-3.00,-3.18,-3.37,-3.58,-3.76,-4.01,-4.31,-4.67,-4.88,-5.43,-6.20,None,None,None])
        self.gsmf_data[6]   = self.clean([0.06,-0.21,-0.56,-0.92,-1.24,-1.52,-1.75,-1.96,-2.14,-2.33,-2.48,-2.67,-2.84,-3.04,-3.22,-3.41,-3.61,-3.85,-4.04,-4.28,-4.62,-4.90,-5.36,None,-6.20,None,None,None,None])
        self.gsmf_data[7]   = self.clean([-0.09,-0.36,-0.71,-1.09,-1.45,-1.77,-2.03,-2.26,-2.46,-2.67,-2.86,-3.03,-3.25,-3.44,-3.67,-3.92,-4.17,-4.26,-4.65,-4.97,-5.20,-5.73,None,-5.90,None,None,None,None,None])

        # -------- TABLE 2 (z = 8 → 17) --------
        self.gsmf_data[8]  = self.clean([-0.30,-0.55,-0.89,-1.28,-1.68,-2.04,-2.32,-2.59,-2.81,-3.06,-3.25,-3.48,-3.66,-3.92,-4.23,-4.35,-4.61,-5.12,-5.60,-5.51,-5.90,-6.20,None,None,None,None,None,None,None])
        self.gsmf_data[9]  = self.clean([-0.53,-0.77,-1.11,-1.51,-1.93,-2.33,-2.66,-2.95,-3.25,-3.45,-3.70,-3.93,-4.21,-4.52,-4.76,-4.93,-5.36,-5.51,None,-6.20,-6.20,None,None,None,None,None,None,None,None])
        self.gsmf_data[10] = self.clean([-0.80,-1.04,-1.37,-1.76,-2.21,-2.64,-3.03,-3.39,-3.64,-3.91,-4.22,-4.42,-4.74,-5.03,-5.25,-5.51,-6.20,-6.20,None,None,None,None,None,None,None,None,None,None,None])
        self.gsmf_data[11] = self.clean([-1.11,-1.33,-1.65,-2.04,-2.49,-2.98,-3.39,-3.77,-4.05,-4.43,-4.65,-4.90,-5.30,-5.90,-5.90,-6.20,None,-6.20,None,None,None,None,None,None,None,None,None,None,None])
        self.gsmf_data[12] = self.clean([-1.45,-1.67,-1.98,-2.36,-2.80,-3.31,-3.81,-4.21,-4.59,-5.06,-5.20,-5.51,-5.73,None,-6.20,-6.20,None,None,None,None,None,None,None,None,None,None,None,None,None])
        self.gsmf_data[13] = self.clean([-1.83,-2.03,-2.32,-2.68,-3.13,-3.64,-4.10,-4.62,-4.97,-5.73,-5.90,-5.73,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None])
        self.gsmf_data[14] = self.clean([-2.24,-2.44,-2.72,-3.08,-3.53,-3.99,-4.57,-5.16,-5.43,-6.20,-6.20,None,None,-6.20,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None])
        self.gsmf_data[15] = self.clean([-2.69,-2.90,-3.15,-3.52,-3.95,-4.52,-5.16,-5.36,-6.20,-6.20,-6.20,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None])
        self.gsmf_data[16] = self.clean([-3.18,-3.36,-3.66,-3.98,-4.42,-5.09,-5.30,-5.90,-5.90,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None])
        self.gsmf_data[17] = self.clean([-3.68,-3.89,-4.10,-4.55,-4.88,-5.12,None,None,-6.20,None,-6.20,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None])

        self.z_vals = np.array(sorted(self.gsmf_data.keys()))
        self.gsmf_grid = np.array([self.gsmf_data[z] for z in self.z_vals])

        self.gsmf_grid = np.nan_to_num(self.gsmf_grid, nan=np.nan, posinf=np.nan, neginf=np.nan)

        # have a single 2D interpolator for z and lgMgal, which will be used for extrapolation as well
        self.interp = RegularGridInterpolator((self.z_vals, self.lgMgal_data), self.gsmf_grid, bounds_error=False, fill_value=np.nan)
    
    def clean(self, arr):
        return np.array([np.nan if x is None else x for x in arr], dtype=float)

    def build_mass_grid(self, n_points_mass=1000):
        self.lgMgal_grid = np.linspace(self.lgMgal_data.min(), self.lgMgal_data.max(), n_points_mass)
        return self.lgMgal_grid

    def get_gsmf(self, z_gal, n_points_mass=1000):
        """
        Continuous GSMF:
        input: logMBH (scalar or array), z
        output: log10(phi)
        """

        # this should have at least 1D shape for interpolation, even if it's just a single value
        z_gal = np.atleast_1d(z_gal)

        # this one should be built from the mass grid of the data, not the input mass grid, since the interpolation is based on the data grid
        lgMgal = self.build_mass_grid(n_points_mass=n_points_mass)  # (N_M,) 

        Z, M = np.meshgrid(z_gal, lgMgal, indexing='ij')  # (N_z, N_M) same z repeated across rows, same lgMgal repeated across columns

        # it goes like pts[len(M[0]):len(M[0])+len(M[1]),1] == pts[:len(M[0]),1]
        # and for z it repeats for all M values for the same z, then moves to the next z
        pts = np.stack([Z.ravel(), M.ravel()], axis=1)  # (N_z * N_M, 2) 
        # interpolate in mass

        f_mass = self.interp(pts).reshape(Z.shape)  # (N_z, N_M)

        return lgMgal, f_mass

    def sample_gsmf(self, z_gal, size=1000):
        lgMgal, phi = self.get_gsmf(z_gal, n_points_mass=size)
        return lgMgal

class MBHMassFunction:

    def __init__(self, gsmf):
        self.gsmf = gsmf

    @staticmethod
    def _compute_phi_MBH(lgMgal_grid, log10phi_gsmf):
        """
        Convert GSMF φ(Mgal,z) into MBH MF φ(MBH,z).
        Shared by both get_mbhmf() and _build_mbhmf_grid().
        """

        logMBH_grid = Galaxy.lgMBH_from_Mgal(lgMgal_grid)  # (NM,)

        dlogMBH_dlogMgal = np.gradient(logMBH_grid, lgMgal_grid)
        dlogMgal_dlogMBH = 1.0 / dlogMBH_dlogMgal  # (NM,)

        phi_gsmf_linear = 10**log10phi_gsmf  # (Nz, NM)
        dndlogMBH_linear = phi_gsmf_linear * dlogMgal_dlogMBH[None, :]  # (Nz, NM)
        log10phi_MBH = np.log10(dndlogMBH_linear + 1e-300)

        return logMBH_grid, log10phi_MBH

    def get_mbhmf(self, z_gal, n_points_mass=1000):
        """
        Return MBH MF φ(log MBH,z) on the GSMF mass grid
        for user-specified redshifts z_gal.
        """

        lgMgal_grid, log10phi_gsmf = self.gsmf.get_gsmf(z_gal, n_points_mass=n_points_mass)  # (NM,), (Nz,NM)
        logMBH_grid, log10phi_MBH = self._compute_phi_MBH(lgMgal_grid, log10phi_gsmf)

        return logMBH_grid, log10phi_MBH

    def _build_mbhmf_grid(self, n_points_mass=2000):
        """
        Build the full 2D MBHMF grid at all GSMF z-values.
        """
        z_vals = self.gsmf.z_vals  # (Nz,)
        lgMgal_grid = self.gsmf.build_mass_grid(n_points_mass=n_points_mass)

        _, log10phi_gsmf = self.gsmf.get_gsmf(z_vals, n_points_mass=n_points_mass)  # (Nz,NM)
        logMBH_grid, log10phi_MBH = self._compute_phi_MBH(lgMgal_grid, log10phi_gsmf)

        return z_vals, logMBH_grid, log10phi_MBH

    def eval_mbhmf(self, z, logMBH, n_points_mass=2000, return_log10=False):
        """
        Evaluate φ_MB(z, logMBH) at arbitrary coordinates.
        """

        z = np.atleast_1d(z)
        logMBH = np.atleast_1d(logMBH)

        if z.shape != logMBH.shape:
            raise ValueError("z and logMBH must have same shape.")

        z_vals, logMBH_grid, log10phi_MBH = self._build_mbhmf_grid(n_points_mass=n_points_mass)

        interp = RegularGridInterpolator(
            (z_vals, logMBH_grid),
            log10phi_MBH,
            bounds_error=False,
            fill_value=-np.inf  # make it zero outside the grid in linear space, which is -inf in log space
        )

        points = np.column_stack([z, logMBH])  # (N,2)
        log10phi_vals = interp(points)

        if return_log10:
            return log10phi_vals

        return 10**log10phi_vals

@dataclass
class CO_mass_function:
    def delta_distribution(self, m, M_CO):
        m = np.asarray(m)
        # exact discrete delta at nearest bin
        psi = np.zeros_like(m, dtype=float)
        idx = np.argmin(np.abs(m - M_CO))
        psi[idx] = 1.0
        return psi

    def power_law(self, m, alpha=2.35):
        """Return power-law distribution ψ(m) ∝ m^{-α}."""
        m = np.asarray(m, float)
        return m**(-alpha)

    def kroupa_imf(self, m):
        """Return Kroupa IMF ψ(m) ∝ m^{-1.3} for m < 0.5 Msun and ψ(m) ∝ m^{-2.3} for m ≥ 0.5 Msun."""
        m = np.asarray(m, float)
        psi = np.zeros_like(m)
        
        # m < 0.5
        mask1 = m < 0.5
        psi[mask1] = m[mask1]**(-1.3)

        # m ≥ 0.5
        mask2 = ~mask1
        psi[mask2] = m[mask2]**(-2.3)

        return psi

    def salpeter(self, m):
        """Return Salpeter IMF ψ(m) ∝ m^{-2.35}."""
        return self.power_law(m, alpha=2.35)

    def normalize(self, func, m):
        """Return func(m) normalized so ∫func(m) dm = 1."""
        psi = np.asarray(func, float)
        m = np.asarray(m, float)

        integral = np.trapz(psi, m)
        if integral == 0:
            raise ValueError("CO mass function integral is zero; check grid or function.")

        return psi / integral

    # ------------------------------------------------------
    # 6. Unified interface
    # ------------------------------------------------------
    def mass_distribution(self, m, kind="kroupa", M_CO=10.0, alpha=2.35):
        m = np.asarray(m, float)

        if kind == "delta":
            psi = self.delta_distribution(m, M_CO)

        elif kind == "power_law":
            psi = self.power_law(m, alpha=alpha)

        elif kind == "kroupa":
            psi = self.kroupa_imf(m)

        elif kind == "salpeter":
            psi = self.salpeter(m)

        else:
            raise ValueError(f"Unknown mass function kind: {kind}")

        return self.normalize(psi, m)

@dataclass
class CosmologyModel:
    """
    Sampler for the LAST major merger redshift z_LMM using an
    inhomogeneous Poisson process and the cumulative hazard method.
    """

    # Cosmology parameters
    H0: float = 70.0      # km/s/Mpc
    Om0: float = 0.3
    Tcmb0: float = 2.725  # K

    def __post_init__(self):
        self.cosmo = FlatLambdaCDM(H0=self.H0, Om0=self.Om0, Tcmb0=self.Tcmb0)

    def H_sinv(self, z):
        """H(z) in s^-1."""
        return self.cosmo.H(z).to(u.s**-1).value

    def dt_dz(self, z):
        """Return |dt/dz| in seconds."""
        return 1.0 / ((1.0 + np.asarray(z)) * self.H_sinv(z))

    def age_Gyr(self, z):
        """Cosmic age t(z) in Gyr."""
        return self.cosmo.age(z).to_value(u.Gyr)
    
    def dVc_dz(self, z):
        """Comoving volume element dV_c/dz in Mpc^3."""
        dVc_dz_dOmega = self.cosmo.differential_comoving_volume(z).to(u.Mpc**3 / u.sr).value
        return 4 * np.pi * dVc_dz_dOmega  # Mpc^3 per unit z assuming whole sky

    def luminosity_distance(self, z):
        """Luminosity distance D_L(z) in Mpc."""
        return self.cosmo.luminosity_distance(z).to(u.Mpc)

class LastMajorMerger:

    def __init__(self, cosmology_model: CosmologyModel):
        self.cosmo_model = cosmology_model if cosmology_model is not None else CosmologyModel()
        self.z_max: float = 12.0
        # Resolution of hazard grid
        self.n_grid: int = 10


    def lambda_MM(self, z):
        """
        Major-merger rate model: λ(z | M) = λ0(M) (1+z)^{m(M)}
        We can replace λ0(M) or m(M) with any mass model.

        Example simple model:
            λ0 = 0.25 Gyr^-1
            m(M) = 2 + 0.3 * log10(M/1e10)
        """
        z = np.asarray(z)

        m = 2.0
        # For now, we can keep λ0 constant and m fixed, but in the future we can make them mass-dependent.
        lambda0 = 0.25  # Gyr^-1

        return lambda0 * (1 + z)**m  # Gyr^-1

    def hazard_integrand(self, z):
        """
        Return hazard integrand for LMM sampling: g(z | M) = λ(z | M) * |dt/dz|
        https://web.stanford.edu/~lutian/coursepdf/unit1.pdf
        """

        lambda_z_Gyr = self.lambda_MM(z)

        lambda_z_s = lambda_z_Gyr / (3.15576e16)  # convert to s^-1
        return lambda_z_s * self.cosmo_model.dt_dz(z)   # events per redshift (dimensionless)

    def build_hazard_grid(self, z_obs_array):
        z_obs_array = np.asarray(z_obs_array)
        N = z_obs_array.shape[0]

        # Create linear grid per galaxy
        z_grid_lin = np.linspace(0, 1, self.n_grid)  # normalized 0→1
        z_grid = z_obs_array[:, None] + (self.z_max - z_obs_array[:, None]) * z_grid_lin

        g = self.hazard_integrand(z_grid)
        dz = np.diff(z_grid, axis=1)

        # Cumulative hazard Λ(z_obs → z)
        Lambda = np.zeros_like(z_grid)
        Lambda[:, 1:] = np.cumsum(0.5 * (g[:, 1:] + g[:, :-1]) * dz, axis=1)

        return z_grid, Lambda


    def sample_z_LMM(self, z_obs_array, size=1):
        """
        size is the number of samples per galaxy. If size=1, returns shape (N,). If size>1, returns shape (N, size).
        """
        z_obs_array = np.asarray(z_obs_array)
        N = len(z_obs_array)

        z_grid, Lambda = self.build_hazard_grid(z_obs_array)  # shapes (N, n_grid)

        U = np.random.random((N, size))
        H_star = -np.log1p(-U)  # shape (N, size)

        z_samples = np.zeros((N, size))

        # Loop over galaxies
        # this needs to be worked on for vectorization, but for now we can just loop since n_grid is small
        #also np.searchsorted does not support 3D arrays for and we get  got ValueError: object too deep for desired array ERROR
        
        for i in range(N):
            # vectorized over 'size'
            z_samples[i] = np.interp(H_star[i], Lambda[i], z_grid[i])

        return z_samples.squeeze()  # (N,) if size==1 else (N, size)

    # -----------------------------------
    def sample_LMM_times(self, z_obs_array, size=1):
        z_LMM = self.sample_z_LMM(z_obs_array, size=size)
        t_LMM = self.cosmo_model.age_Gyr(z_LMM)
        t_obs = self.cosmo_model.age_Gyr(z_obs_array)
        return z_LMM, t_LMM, t_obs