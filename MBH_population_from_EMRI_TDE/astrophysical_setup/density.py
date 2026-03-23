
from typing import Sequence
import numpy as np

from config import (Msun_to_grams, gamma_dehnen_initial)
from nsc import NSC

class DehnenProfile:

    def __init__(self, nsc: NSC, gamma: float = gamma_dehnen_initial):
        self.nsc = nsc
        self.gamma_dehnen_initial = float(gamma)

    def dehnen_number_density(self, r, Ntot, kind='TDE', unit='1/pc^3'):
        """
        Dehnen 3D number-density profile n_i(r) for species i (stars or sBHs)
        n_i(r) = (3-$\gamma$)/(4π) * Ntot * r_a / [ r^$\gamma$ (r + r_a)^(4-$\gamma$) ] 

        Parameters
        ----------
        r         : array-like, radii [pc] at which to evaluate
        Ntot      : float, total number of objects (dimensionless)
        gamma     : float, inner slope
        kind      : str, 'TDE' or 'EMRI' (for potential future use in different truncations)

        Returns
        -------
        n_i : ndarray, number density [1/pc^3] same shape as r
        """

        r = np.asarray(r, float)

        r_a = self.nsc.scale_radius(kvir=1.0, factor=4.0, unit="pc")
        r_k = (self.nsc.r_capture(unit="pc")
               if kind.upper() == "EMRI"
               else self.nsc.r_tidal(unit="pc"))

        coef = (3 - self.gamma_dehnen_initial) * Ntot * r_a / (4 * np.pi)

        n = coef / (
            np.power(r, self.gamma_dehnen_initial) * np.power(r + r_a, 4 - self.gamma_dehnen_initial)
        )

        # apply cutoff (TDE or EMRI)
        n = np.where(r >= r_k, n, 0.0)
        n = np.where(np.isfinite(n), n, 0.0)
        return n  # [1/pc^3]

    def radial_number_distribution(self, r, Ntot, kind='TDE', unit='1/pc'):
        """
        Shell number distribution:
            n_r(r) = 4π r^2 n_i(r)    [units: 1/pc]
        """
        r = np.asarray(r, float)
        return 4 * np.pi * r**2 * self.dehnen_number_density(r, Ntot, kind)

    def cumulative_number(self, r, Ntot, kind='TDE'):
        r = np.asarray(r)
        nr = self.radial_number_distribution(r, Ntot=Ntot, kind=kind)
        return np.cumsum(0.5 * (nr[1:] + nr[:-1]) * np.diff(r))

    def number_of_CO_within_shell(self, r_min, r_max, Ntot, kind='TDE', npts=2000):
        """
        Number of objects between r_min and r_max:
            N = ∫_{r_min}^{r_max} 4π r^2 n(r) dr
        """
        r = np.logspace(np.log10(r_min), np.log10(r_max), npts)
        nr = self.radial_number_distribution(r, Ntot=Ntot, kind=kind)
        return np.trapezoid(nr, r)

    def mass_density(self, r, Ntot, component_masses, kind='TDE', renormalize=False, unit='Msun/pc^3'):
        """
        Total 3D mass density: rho(r) = sum_i m_i * n_i(r)

        Parameters
        ----------
        r       : array-like radii (pc)
        gamma   :
        Ntot    :
        component_masses : in units of solar mass
        renormalize: if True, rescale n_i so that ∫ 4π r^2 n_i dr (over the provided grid) = Ntot
        """
        r = np.asarray(r, dtype=float)
        comp_mass = np.asarray(component_masses, dtype=float) # in units of 

        n_i = self.dehnen_number_density(r, Ntot=Ntot, kind=kind)  # 1/pc^3

        if renormalize:
            nr = 4.0 * np.pi * r**2 * n_i  # 1/pc
            N_calc = np.trapezoid(nr, r)
            if np.isfinite(N_calc) and N_calc > 0.0:
                n_i = n_i * (Ntot / N_calc)
            # Add component mass density (Msun/pc^3)
        rho_Msun_pc3 = np.sum(comp_mass[:, None] * n_i, axis=0) if comp_mass.ndim == 1 else comp_mass * n_i

        if unit == 'Msun/pc^3':
            return rho_Msun_pc3
        elif unit == 'g/pc^3':
            return rho_Msun_pc3 * Msun_to_grams
        else:
            raise ValueError("unit must be 'Msun/pc^3' or 'g/pc^3'.")

