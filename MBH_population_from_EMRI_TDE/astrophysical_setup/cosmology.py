import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u


@dataclass
class LastMajorMerger:
    """
    Sampler for the LAST major merger redshift z_LMM using an
    inhomogeneous Poisson process and the cumulative hazard method.
    """

    # Cosmology parameters
    H0: float = 70.0      # km/s/Mpc
    Om0: float = 0.3
    Tcmb0: float = 2.725  # K

    # Sampling grid upper limit
    z_max: float = 12.0

    # Resolution of hazard grid
    n_grid: int = 6000

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
        lambda0 = 0.25  # Gyr^-1

        return lambda0 * (1 + z)**m  # Gyr^-1

    def hazard_integrand(self, z):
        """
        Return hazard integrand for LMM sampling: g(z | M) = λ(z | M) * |dt/dz|
        https://web.stanford.edu/~lutian/coursepdf/unit1.pdf
        """

        lambda_z_Gyr = self.lambda_MM(z)

        lambda_z_s = lambda_z_Gyr / (3.15576e16)  # convert to s^-1
        return lambda_z_s * self.dt_dz(z)   # events per redshift (dimensionless)

    def build_hazard_grid(self, z_obs):
        """
        Precompute:
            z_grid
            Λ(z_obs → z_grid)
        """
        z_grid = np.linspace(z_obs, self.z_max, self.n_grid)

        g = self.hazard_integrand(z_grid)

        # Cumulative hazard Λ(z_obs → z)
        Lambda = np.zeros_like(z_grid)
        Lambda[1:] = np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(z_grid))

        return z_grid, Lambda

    def sample_z_LMM(self, z_obs, size=1):
        """
        Sample 'size' last-major-merger redshifts.

        Steps:
            1. Compute hazard grid
            2. Draw U ~ Uniform(0,1)
            3. H* = -ln(1 - U)
            4. Invert cumulative hazard to find z_LMM
        """
        z_grid, Lambda = self.build_hazard_grid(z_obs)

        # Draw hazards
        U = np.random.random(size)
        H_star = -np.log1p(-U)  # numerically stable

        # Invert Λ -> z_LMM
        z_samples = np.interp(H_star, Lambda, z_grid)
        return z_samples.squeeze()

    def sample_LMM_times(self, z_obs, size=1):
        """
        Returns:
            z_LMM, t_LMM[Gyr], t_obs[Gyr]
        """
        z_LMM = self.sample_z_LMM(z_obs, size=size)
        t_LMM = self.age_Gyr(z_LMM)
        t_obs = self.age_Gyr(z_obs)
        return z_LMM, t_LMM, t_obs


sampler = LastMajorMerger()

# print(z_LMM)
import matplotlib.pyplot as plt

for z_obs in [.5, 5, 10]:
    z_LMM = sampler.sample_z_LMM(z_obs, size=100)
    print(z_LMM)
    plt.hist(z_LMM, bins=10, label=z_obs)

plt.xlabel('z')
plt.ylabel('# z_LMM')
plt.legend()
plt.show()