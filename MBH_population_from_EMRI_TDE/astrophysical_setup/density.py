
from typing import Sequence
import numpy as np

from config import (Msun_to_grams, gamma_dehnen_initial)
from nsc import NSC

class DehnenProfile:

    def __init__(self, nsc: NSC, gamma: float = gamma_dehnen_initial):
        self.nsc = nsc
        self.gamma_dehnen_initial = float(gamma)

    def dehnen_number_density(self, r_grid, Ntot, kind='TDE', unit='1/pc^3'):
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

        r_grid = np.asarray(r_grid, float)

        r_a = self.nsc.scale_radius(kvir=1.0, factor=4.0, unit="pc")
        r_k = (self.nsc.r_capture(unit="pc")
               if kind.upper() == "EMRI"
               else self.nsc.r_tidal(unit="pc"))

        coef = (3 - self.gamma_dehnen_initial) * Ntot * r_a / (4 * np.pi)
        
        n = coef[:, None] / (
            np.power(r_grid, self.gamma_dehnen_initial) * np.power(r_grid[None,:] + r_a[:, None], 4 - self.gamma_dehnen_initial)
        )
        # apply cutoff (TDE or EMRI)
        # n = np.where(r_grid >= r_k, n, 0.0)

        mask = (r_grid[None,:] >= r_k[:,None])
        n = np.where(mask, n, 0.0)

        return n  # [1/pc^3]  with dimension (N, Nr)

    def radial_number_distribution(self, r_grid, Ntot, kind='TDE', unit='1/pc'):
        """
        Shell number distribution:
            n_r(r) = 4π r^2 n_i(r)    [units: 1/pc]
        """
        r_grid = np.asarray(r_grid, float)
        return 4 * np.pi * r_grid[None, :]**2 * self.dehnen_number_density(r_grid, Ntot, kind)

    def cumulative_number(self, r_grid, Ntot, kind='TDE'):
        r_grid = np.asarray(r_grid)
        nr = self.radial_number_distribution(r_grid, Ntot=Ntot, kind=kind)
        dr = np.diff(r_grid)[None, :]   # (1, Nr-1)
        return np.cumsum(0.5 * (nr[:,1:] + nr[:,:-1]) * dr, axis=1)


    def number_of_CO_within_shell(self, r_min, r_max, Ntot, kind='TDE', npts=2000):
        """
        Number of objects between r_min and r_max:
            N = ∫_{r_min}^{r_max} 4π r^2 n(r) dr
        """
        r_grid = np.logspace(np.log10(r_min), np.log10(r_max), npts)
        nr = self.radial_number_distribution(r_grid, Ntot, kind)
        return np.trapezoid(nr, r_grid, axis=1)   # <-- (N,)


    def mass_density(self, r_grid, Ntot, component_masses, kind='TDE', renormalize=False, unit='Msun/pc^3'):
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
        r_grid = np.asarray(r_grid, dtype=float)
        comp_mass = np.asarray(component_masses, dtype=float) # in units of 

        n_i = self.dehnen_number_density(r_grid, Ntot=Ntot, kind=kind)  # 1/pc^3

        if renormalize:
            nr = 4.0 * np.pi * r_grid**2 * n_i  # 1/pc
            N_calc = np.trapezoid(nr, r_grid)
            if np.isfinite(N_calc) and N_calc > 0.0:
                n_i = n_i * (Ntot / N_calc)
        
        # Add component mass density (Msun/pc^3)
        if comp_mass.ndim == 1:
            rho_Msun_pc3 = np.sum(comp_mass[:, None] * n_i, axis=0)  # For 1D array of masses
        else:
            rho_Msun_pc3 = np.sum(comp_mass[:,:,None] * n_i[:,None,:], axis=1)  # Adjust indexing if comp_mass is 2D


        if unit == 'Msun/pc^3':
            return rho_Msun_pc3
        elif unit == 'g/pc^3':
            return rho_Msun_pc3 * Msun_to_grams
        else:
            raise ValueError("unit must be 'Msun/pc^3' or 'g/pc^3'.")

