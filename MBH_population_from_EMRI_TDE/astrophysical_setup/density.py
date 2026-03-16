
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
        r_a = self.nsc.scale_radius(kvir=1.0, factor=4.0, unit='pc')

        coef = (3.0 - self.gamma_dehnen_initial) * float(Ntot) / (4.0 * np.pi)

        r_k = self.nsc.r_capture(unit='pc') if kind.upper() == 'EMRI' else self.tidal_radius_star(unit='pc')

        n = coef * (r_a / (np.power(r, self.gamma_dehnen_initial) * np.power(r + r_a, 4.0 - self.gamma_dehnen_initial))) * np.heaviside(r - r_k, r_k)  # 1/pc^3
        n = np.where(np.isfinite(n), n, 0.0)

        return n  # 1/pc^3

    def radial_number_distribution(self, r, Ntot, kind='TDE', unit='1/pc'):
        """
        Shell number distribution:
            n_r(r) = 4π r^2 n_i(r)    [units: 1/pc]
        """
        return 4.0 * np.pi * r**2 * self.dehnen_number_density(r, Ntot=Ntot, kind=kind)  # 1/pc

    def cumulative_number_within_radius(self, r, Ntot, kind='TDE'):
        """
        Cumulative number:
            N(<r) = ∫_0^r 4π r'^2 n_i(r') dr'
        """
        nr = self.radial_number_distribution(r, Ntot=Ntot, kind=kind)  # 1/pc

        sort = np.argsort(r)
        r_s = r[sort]
        nr_s = nr[sort]

        # trapezoid cumulative
        if r_s.size > 1:
            partial = np.concatenate(([0.0], np.cumsum(0.5 * (nr_s[1:] + nr_s[:-1]) * np.diff(r_s))))
        else:
            partial = np.zeros_like(r_s)

        # unsort
        inv = np.argsort(sort)
        return partial[inv]

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
        m_i = np.asarray(component_masses, dtype=float) # in units of solar mass
        rho_Msun_pc3 = np.zeros_like(r, dtype=float)  # accumulate in Msun/pc^3

        n_i = self.dehnen_number_density(r, Ntot=Ntot, kind=kind)  # 1/pc^3

        for i in range(len(m_i)):
            m_Msun = m_i[i]


            if renormalize:
                nr = 4.0 * np.pi * r**2 * n_i  # 1/pc
                sort = np.argsort(r)
                r_s, nr_s = r[sort], nr[sort]
                N_calc = np.trapz(nr_s, r_s) if r_s.size > 1 else 0.0
                if np.isfinite(N_calc) and N_calc > 0.0:
                    n_i = n_i * (Ntot / N_calc)
            # Add component mass density (Msun/pc^3)
            rho_Msun_pc3 += m_Msun * n_i

        if unit == 'Msun/pc^3':
            return rho_Msun_pc3
        elif unit == 'g/pc^3':
            return rho_Msun_pc3 * Msun_to_grams
        else:
            raise ValueError("unit must be 'Msun/pc^3' or 'g/pc^3'.")

