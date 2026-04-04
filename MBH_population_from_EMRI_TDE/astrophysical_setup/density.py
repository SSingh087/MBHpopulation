import numpy as np
from config import Msun_to_grams, gamma_dehnen_initial
from nsc import NSC, CompactObject

class DehnenProfile:
    """
    All methods return arrays of shape (N, Nr),
    even when N = 1, to preserve clean vectorization.

    For users who want 1D output when N=1,
    use helper functions at the bottom of the file.
    """

    def __init__(self, nsc: NSC, compact_object: CompactObject, gamma: float = gamma_dehnen_initial):
        self.nsc = nsc
        self.compact_object = compact_object
        self.gamma_dehnen_initial = float(gamma)

    def dehnen_number_density(self, r_grid, kind='TDE', unit='1/pc^3'):
        """
        Returns n(r) with shape (N, Nr)
        """
        r_grid = np.asarray(r_grid, float)         # expect shape (N, Nr)
        
        Ntot = self.compact_object.total_number['sBH'] if kind.upper() == 'EMRI' else self.compact_object.total_number['star']        # (N,)

        N, Nr = r_grid.shape

        r_a = self.nsc.scale_radius(kvir=1.0, factor=4.0, unit="pc")   # (N,)
        r_k = (self.nsc.r_capture(unit='pc') 
               if kind.upper() == 'EMRI'
               else self.nsc.r_tidal(unit='pc'))                      # (N,)

        gamma = self.gamma_dehnen_initial

        # prefactor → shape (N, 1)
        coef = ((3 - gamma) * Ntot * r_a / (4 * np.pi))[:, None]

        # denominator → broadcast to (N, Nr)
        denom = (r_grid ** gamma) * (r_grid + r_a[:, None]) ** (4 - gamma)

        n = coef / denom

        # cutoff mask: r >= r_k
        mask = (r_grid >= r_k[:, None])
        n = np.where(mask, n, 0.0)

        return n

    def radial_number_distribution(self, r_grid, kind='TDE', unit='1/pc'):
        r_grid = np.asarray(r_grid, float)
        n = self.dehnen_number_density(r_grid, kind)   # (N, Nr)
        return 4.0 * np.pi * (r_grid ** 2) * n

    def cumulative_number(self, r_grid, kind='TDE'):
        """
        Returns N(<r) with shape (N, Nr-1)
        """
        r_grid = np.asarray(r_grid, float)          # (N, Nr)
        nr = self.radial_number_distribution(r_grid, kind)  # (N, Nr)

        dr = np.diff(r_grid, axis=1)               # (N, Nr-1)

        # trapezoidal integration over r
        Ncum = np.cumsum(0.5 * (nr[:, 1:] + nr[:, :-1]) * dr, axis=1)
        return Ncum

    def number_of_CO_within_shell(self, r_min, r_max, kind='TDE', npts=2000):
        """
        Computes the number of objects between r_min and r_max for each galaxy.
        Accepts scalar or array radii. Returns array of shape (N,).
        """

        r_min = np.asarray(r_min, float)
        r_max = np.asarray(r_max, float)
        Ntot = self.compact_object.total_number['sBH'] if kind.upper() == 'EMRI' else self.compact_object.total_number['star'] # (N,)

        # Scalar → broadcast to arrays of shape (N,)
        N = Ntot.size
        if r_min.ndim == 0:
            r_min = np.full(N, r_min)
        if r_max.ndim == 0:
            r_max = np.full(N, r_max)

        # Safety checks
        if r_min.shape != (N,) or r_max.shape != (N,):
            raise ValueError("r_min, r_max, and Ntot must all have shape (N,)")

        # log r_min, log r_max shapes: (N,1)
        log_rmin = np.log10(r_min)[:, None]
        log_rmax = np.log10(r_max)[:, None]

        # base grid in [0,1]
        u = np.linspace(0, 1, npts)[None, :]        # (1, npts)

        # final radius grid: (N, npts)
        r_grid = 10**(log_rmin + u * (log_rmax - log_rmin))

        nr = self.radial_number_distribution(r_grid, kind)  # (N, npts)
        N_objects = np.trapezoid(nr, r_grid, axis=1)              # (N,)

        return N_objects

    def dehnen_n_at_radius(self, r, kind='TDE'):
        r = np.asarray(r, float)           # (N,)
        Ntot = self.compact_object.total_number['sBH'] if kind.upper() == 'EMRI' else self.compact_object.total_number['star']    

        r_a = self.nsc.scale_radius(kvir=1.0, factor=4.0, unit="pc")
        r_k = (self.nsc.r_capture(unit="pc") if kind.upper() == "EMRI"
               else self.nsc.r_tidal(unit="pc"))

        gamma = self.gamma_dehnen_initial

        coef = (3 - gamma) * Ntot * r_a / (4 * np.pi)
        denom = (r ** gamma) * (r + r_a) ** (4 - gamma)

        n = coef / denom
        n = np.where(r >= r_k, n, 0.0)

        return n                              # (N,)

    def mass_density(self, r_grid, unit='Msun/pc^3'):
        r_grid = np.asarray(r_grid, float)       # (N, Nr)
        N, Nr = r_grid.shape

        if self.compact_object.types_masses == 'same_mass':
            sBH_mass = self.compact_object.total_mass['sBH']  # scalar
            star_mass = self.compact_object.total_mass['star']  # scalar
            rho = sBH_mass * self.dehnen_number_density(r_grid, kind='EMRI') + star_mass * self.dehnen_number_density(r_grid, kind='TDE')  # (N, Nr)
        else:
            raise NotImplementedError("Random mass sampling for COs is not implemented yet. Please use 'same_mass' for now.")

            # comp_mass = np.asarray(self.compact_object.component_masses, float)
            # n_i = self.dehnen_number_density(r_grid, kind)   # (N, Nr)

            # if comp_mass.ndim == 1:
            #     rho = np.sum(comp_mass) * n_i                      # (N, Nr)
            # else:
            #     rho = np.sum(comp_mass[:, :, None] * n_i[:, None, :], axis=1)

        if unit == 'Msun/pc^3':
            return rho
        elif unit == 'g/pc^3':
            return rho * Msun_to_grams
        else:
            raise ValueError("unit must be 'Msun/pc^3' or 'g/pc^3'.")

    def mass_density_at_rinfl(self, kvir=1.0, unit='Msun/pc^3'):
        
        r_inf = self.nsc.r_influence(kvir=kvir, unit='pc')   # (N,)

        if self.compact_object.types_masses == 'same_mass':
            rho = (self.compact_object.total_mass['sBH'] * self.dehnen_n_at_radius(r_inf, kind='EMRI')) + (self.compact_object.total_mass['star'] * self.dehnen_n_at_radius(r_inf, kind='TDE'))  # (N,)
        else:
            raise NotImplementedError("Random mass sampling for COs is not implemented yet. Please use 'same_mass' for now.")

            # if comp_mass.ndim == 1:
            #     rho = n * np.sum(comp_mass)
            # else:
            #     rho = np.sum(comp_mass * n[:, None], axis=1)

        if unit == 'Msun/pc^3':
            return rho
        elif unit == 'g/pc^3':
            return rho * Msun_to_grams
        else:
            raise ValueError("Invalid unit for mass density.")