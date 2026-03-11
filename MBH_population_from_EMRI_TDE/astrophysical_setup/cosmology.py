# cosmology.py
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u


@dataclass
class CosmologyModel:
    H0: float = 70.0     # km/s/Mpc
    Om0: float = 0.3     # Matter density parameter at z=0 (Omega_matter)
    Tcmb0: float = 2.725  # CMB temperature at z=0 in K
    z_max: float = 10.0   # max z for LMM sampling (can be overridden in methods)

    def __post_init__(self):
        self.cosmo = FlatLambdaCDM(H0=self.H0, Om0=self.Om0, Tcmb0=self.Tcmb0)

    # H(z) in s^-1 (plain float)
    def H_sinv(self, z: np.ndarray | float) -> np.ndarray | float:
        return self.cosmo.H(z).to(u.s**-1).value

    # Cosmic age t(z) in Gyr (plain float)
    def age_Gyr(self, z: np.ndarray | float) -> np.ndarray | float:
        return self.cosmo.age(z).to_value(u.Gyr)

    # dt/dz = -1 / [(1+z) H(z)]  in seconds per redshift (negative)
    def dt_dz_seconds(self, z: np.ndarray | float) -> np.ndarray | float:
        Hz = self.H_sinv(z)  # s^-1
        return -1.0 / ((1.0 + np.asarray(z)) * Hz)

    # Unnormalized p(z_lmm|z_obs) weight: w(z) ∝ (1+z)^m * |dt/dz|
    # = (1+z)^(m-1) / H(z). (lambda0 cancels)
    def lmm_weight(self, z: np.ndarray | float, m: float) -> np.ndarray | float:
        z_arr = np.asarray(z)
        return (1.0 + z_arr)**(m - 1.0) / self.H_sinv(z_arr)

    # Build normalized PDF and CDF on [z_obs, z_max] grid
    def lmm_pdf_cdf(self, z_obs: float, m: float, n_grid: int = 4096,
                    z_max: Optional[float] = None
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z_hi = float(self.z_max if z_max is None else z_max)
        if not (z_hi > z_obs):
            raise ValueError("z_max must be > z_obs for LMM sampling.")

        z_grid = np.linspace(z_obs, z_hi, n_grid)
        w = self.lmm_weight(z_grid, m)              # unnormalized
        
        # Normalize to a proper PDF
        area = np.trapezoid(w, z_grid)
        if area <= 0.0 or not np.isfinite(area):
            raise RuntimeError("Invalid normalization for LMM PDF (check cosmology or grid).")
        pdf = w / area

        # CDF via cumulative trapezoid (monotone)
        # Build CDF aligned to z_grid (cdf[0]=0)
        cdf = np.empty_like(pdf)
        cdf[0] = 0.0
        cdf[1:] = np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(z_grid))
        # Numerical guard: enforce last exactly 1
        cdf[-1] = 1.0

        return z_grid, pdf, cdf

    # Draw samples from p(z_LMM|z_obs) using inverse-CDF on the prebuilt grid
    def sample_lmm_redshift(self,z_obs: float, m: float, size: int = 1, n_grid: int = 4096,
                            z_max: Optional[float] = None) -> np.ndarray:
        z_grid, pdf, cdf = self.lmm_pdf_cdf(z_obs, m, n_grid=n_grid, z_max=z_max)
        u = np.random.random(size)
        
        # np.interp expects strictly increasing cdf; it's monotone by construction.
        z_samples = np.interp(u, cdf, z_grid)
        return z_samples

    # Convenience: sample one and return times in Gyr
    def sample_lmm_times_Gyr(self, z_obs: float, m: float,
                             z_max: Optional[float] = None) -> Tuple[float, float, float]:
        """
        Returns (z_LMM, t_LMM[Gyr], t_obs[Gyr]).
        """
        z_LMM = float(self.sample_lmm_redshift(z_obs, m, size=1, z_max=z_max)[0])
        t_LMM = float(self.age_Gyr(z_LMM))
        t_obs = float(self.age_Gyr(z_obs))
        return z_LMM, t_LMM, t_obs
