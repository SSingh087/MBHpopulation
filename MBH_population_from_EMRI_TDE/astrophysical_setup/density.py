"""Dehnen density profile used for the nuclear star cluster (NSC) cusp."""
import numpy as np
from config import Msun_to_grams, gamma_dehnen_initial
from nsc import NSC, CompactObject

class DehnenProfile:
    """
    All methods return arrays of shape (N_galaxies, N_radii_for_each_galaxy),
    even when N_galaxies = 1, to preserve clean vectorization.

    For users who want 1D output when N_galaxies=1,
    use helper functions at the bottom of the file.
    """

    def __init__(self, nsc: NSC, compact_object: CompactObject, gamma: float = gamma_dehnen_initial):
        self.nsc = nsc
        self.compact_object = compact_object
        self.gamma_dehnen_initial = float(gamma)

        self.sBH_mass = self.compact_object.masses['sBH']
        self.star_mass = self.compact_object.masses['star']
        self.star_radius = self.compact_object.CO_radius['star']

    def dehnen_number_density(self, r_grid, kind='TDE', unit='1/pc^3'):
        """
        Returns n(r) with shape (N_galaxies, N_radii_for_each_galaxy)
        """
        r_grid = np.asarray(r_grid, float)         # expect shape (N_galaxies, N_radii_for_each_galaxy)

        Ntot = self.compact_object.total_number['sBH'] if kind.upper() == 'EMRI' else self.compact_object.total_number['star']        # (N_galaxies,)
        N_galaxies, N_radii_for_each_galaxy = r_grid.shape
        r_a = self.nsc.scale_radius(kvir=1.0, factor=4.0, unit="pc")   # (N_galaxies,)
        r_k = (self.nsc.r_capture(unit='pc') 
               if kind.upper() == 'EMRI'
               else self.nsc.r_tidal(m_star_solar=self.star_mass, R_star_solar=self.star_radius, unit='pc'))                      # (N_galaxies,)

        gamma = self.gamma_dehnen_initial

        # prefactor → shape (N_galaxies, 1)
        coef = ((3 - gamma) * Ntot * r_a / (4 * np.pi))[:, None]

        # denominator → broadcast to (N_galaxies, N_radii_for_each_galaxy)
        denom = (r_grid ** gamma) * (r_grid + r_a[:, None]) ** (4 - gamma)

        n = coef / denom

        # cutoff mask: r >= r_k
        mask = (r_grid >= r_k[:, None])
        n = np.where(mask, n, 0.0)

        return n

    def radial_number_distribution(self, r_grid, kind='TDE', unit='1/pc'):
        r_grid = np.asarray(r_grid, float)
        n = self.dehnen_number_density(r_grid, kind)   # (N_galaxies, N_radii_for_each_galaxy)
        return 4.0 * np.pi * (r_grid ** 2) * n

    def cumulative_number(self, r_grid, kind='TDE'):
        """
        Returns N_galaxies(<r) with shape (N_galaxies, N_radii_for_each_galaxy-1)
        """
        r_grid = np.asarray(r_grid, float)          # (N_galaxies, N_radii_for_each_galaxy)
        nr = self.radial_number_distribution(r_grid, kind)  # (N_galaxies, N_radii_for_each_galaxy)

        dr = np.diff(r_grid, axis=1)               # (N_galaxies, N_radii_for_each_galaxy-1)

        # trapezoidal integration over r
        Ncum = np.cumsum(0.5 * (nr[:, 1:] + nr[:, :-1]) * dr, axis=1)
        return Ncum

    def number_of_CO_within_shell(self, r_min, r_max, kind='TDE', npts=2000):
        """
        Computes the number of objects between r_min and r_max for each galaxy.
        Accepts scalar or array radii. Returns array of shape (N_galaxies,).
        """

        r_min = np.asarray(r_min, float)
        r_max = np.asarray(r_max, float)
        Ntot = self.compact_object.total_number['sBH'] if kind.upper() == 'EMRI' else self.compact_object.total_number['star'] # (N_galaxies,)

        # Scalar → broadcast to arrays of shape (N_galaxies,)
        N_galaxies = Ntot.size
        if r_min.ndim == 0:
            r_min = np.full(N_galaxies, r_min)
        if r_max.ndim == 0:
            r_max = np.full(N_galaxies, r_max)

        # Safety checks
        if r_min.shape != (N_galaxies,) or r_max.shape != (N_galaxies,):
            raise ValueError("r_min, r_max, and Ntot must all have shape (N_galaxies,)")

        # log r_min, log r_max shapes: (N_galaxies,1)
        log_rmin = np.log10(r_min)[:, None]
        log_rmax = np.log10(r_max)[:, None]

        # base grid in [0,1]
        u = np.linspace(0, 1, npts)[None, :]        # (1, npts)

        # final radius grid: (N_galaxies, npts)
        r_grid = 10**(log_rmin + u * (log_rmax - log_rmin))

        nr = self.radial_number_distribution(r_grid, kind)  # (N_galaxies, npts)
        N_objects = np.trapezoid(nr, r_grid, axis=1)              # (N_galaxies,)

        return N_objects

    def dehnen_number_density_at_radius(self, r, kind='TDE'):
        r = np.asarray(r, float)           # (N_galaxies,)
        Ntot = self.compact_object.total_number['sBH'] if kind.upper() == 'EMRI' else self.compact_object.total_number['star']    

        r_a = self.nsc.scale_radius(kvir=1.0, factor=4.0, unit="pc")
        r_k = (self.nsc.r_capture(unit="pc") if kind.upper() == "EMRI"
               else self.nsc.r_tidal(m_star_solar=self.star_mass, R_star_solar=self.star_radius, unit='pc'))

        gamma = self.gamma_dehnen_initial

        coef = (3 - gamma) * Ntot * r_a / (4 * np.pi)
        denom = (r ** gamma) * (r + r_a) ** (4 - gamma)

        n = coef / denom
        n = np.where(r >= r_k, n, 0.0)

        return n                              # (N_galaxies,)

    def mass_density(self, r_grid, unit='Msun/pc^3'):
        r_grid = np.asarray(r_grid, float)       # (N_galaxies, N_radii_for_each_galaxy)
        N_galaxies, N_radii_for_each_galaxy = r_grid.shape

        if self.compact_object.types_masses == 'same_mass':
            sBH_total_mass = self.compact_object.total_mass_in_MBH_mass_units['sBH']  # scalar
            star_total_mass = self.compact_object.total_mass_in_MBH_mass_units['star']  # scalar
            rho = sBH_total_mass * self.dehnen_number_density(r_grid, kind='EMRI') + star_total_mass * self.dehnen_number_density(r_grid, kind='TDE')  # (N_galaxies, N_radii_for_each_galaxy)
        else:
            raise NotImplementedError("Random mass sampling for COs is not implemented yet. Please use 'same_mass' for now.")

        if unit == 'Msun/pc^3':
            return rho
        elif unit == 'g/pc^3':
            return rho * Msun_to_grams
        else:
            raise ValueError("unit must be 'Msun/pc^3' or 'g/pc^3'.")

    def mass_density_at_rinfl(self, kvir=1.0, unit='Msun/pc^3'):
        
        r_inf = self.nsc.r_influence(kvir=kvir, unit='pc')   # (N_galaxies,)

        if self.compact_object.types_masses == 'same_mass':
            rho = (self.compact_object.total_mass_in_MBH_mass_units['sBH'] * self.dehnen_number_density_at_radius(r_inf, kind='EMRI')) + (self.compact_object.total_mass_in_MBH_mass_units['star'] * self.dehnen_number_density_at_radius(r_inf, kind='TDE'))  # (N_galaxies,)
        else:
            raise NotImplementedError("Random mass sampling for COs is not implemented yet. Please use 'same_mass' for now.")

        if unit == 'Msun/pc^3':
            return rho
        elif unit == 'g/pc^3':
            return rho * Msun_to_grams
        else:
            raise ValueError("Invalid unit for mass density.")