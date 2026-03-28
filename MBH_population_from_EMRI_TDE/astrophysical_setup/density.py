import numpy as np
from config import Msun_to_grams, gamma_dehnen_initial
from nsc import NSC

class DehnenProfile:
    """
    All methods return arrays of shape (N, Nr),
    even when N = 1, to preserve clean vectorization.

    For users who want 1D output when N=1,
    use helper functions at the bottom of the file.
    """

    def __init__(self, nsc: NSC, gamma: float = gamma_dehnen_initial):
        self.nsc = nsc
        self.gamma_dehnen_initial = float(gamma)


    # ---------------------------------------------------------------
    # 3D number density
    # ---------------------------------------------------------------
    def dehnen_number_density(self, r_grid, Ntot, kind='TDE', unit='1/pc^3'):
        """
        Returns n(r) with shape (N, Nr)
        """
        r_grid = np.asarray(r_grid, float)         # expect shape (N, Nr)
        Ntot   = np.asarray(Ntot, float)           # (N,)

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


    # ---------------------------------------------------------------
    # Radial shell distribution 4πr^2 n(r)
    # ---------------------------------------------------------------
    def radial_number_distribution(self, r_grid, Ntot, kind='TDE', unit='1/pc'):
        r_grid = np.asarray(r_grid, float)
        n = self.dehnen_number_density(r_grid, Ntot, kind)   # (N, Nr)
        return 4.0 * np.pi * (r_grid ** 2) * n


    # ---------------------------------------------------------------
    # Cumulative number
    # ---------------------------------------------------------------
    def cumulative_number(self, r_grid, Ntot, kind='TDE'):
        """
        Returns N(<r) with shape (N, Nr-1)
        """
        r_grid = np.asarray(r_grid, float)          # (N, Nr)
        nr = self.radial_number_distribution(r_grid, Ntot, kind)  # (N, Nr)

        dr = np.diff(r_grid, axis=1)               # (N, Nr-1)

        # trapezoidal integration over r
        Ncum = np.cumsum(0.5 * (nr[:, 1:] + nr[:, :-1]) * dr, axis=1)
        return Ncum


    # ---------------------------------------------------------------
    # Number of objects within [r_min, r_max]
    # ---------------------------------------------------------------
    def number_of_CO_within_shell(self, r_min, r_max, Ntot, kind='TDE', npts=2000):
        """
        Computes the number of objects between r_min and r_max for each galaxy.
        Accepts scalar or array radii. Returns array of shape (N,).
        """

        r_min = np.asarray(r_min, float)
        r_max = np.asarray(r_max, float)
        Ntot  = np.asarray(Ntot, float)

        # Scalar → broadcast to arrays of shape (N,)
        N = Ntot.size
        if r_min.ndim == 0:
            r_min = np.full(N, r_min)
        if r_max.ndim == 0:
            r_max = np.full(N, r_max)

        # Safety checks
        if r_min.shape != (N,) or r_max.shape != (N,):
            raise ValueError("r_min, r_max, and Ntot must all have shape (N,)")

        # ----------------------------------------------
        # Build per-galaxy radius grids: shape (N, npts)
        # ----------------------------------------------
        # log r_min, log r_max shapes: (N,1)
        log_rmin = np.log10(r_min)[:, None]
        log_rmax = np.log10(r_max)[:, None]

        # base grid in [0,1]
        u = np.linspace(0, 1, npts)[None, :]        # (1, npts)

        # final radius grid: (N, npts)
        r_grid = 10**(log_rmin + u * (log_rmax - log_rmin))

        # ----------------------------------------------
        # Integrate 4π r^2 n(r)
        # ----------------------------------------------
        nr = self.radial_number_distribution(r_grid, Ntot, kind)  # (N, npts)
        N_objects = np.trapezoid(nr, r_grid, axis=1)              # (N,)

        return N_objects

    # ---------------------------------------------------------------
    # n(r) at a single radius per galaxy
    # ---------------------------------------------------------------
    def dehnen_n_at_radius(self, r, Ntot, kind='TDE'):
        r = np.asarray(r, float)           # (N,)
        Ntot = np.asarray(Ntot, float)

        r_a = self.nsc.scale_radius(kvir=1.0, factor=4.0, unit="pc")
        r_k = (self.nsc.r_capture(unit="pc") if kind.upper() == "EMRI"
               else self.nsc.r_tidal(unit="pc"))

        gamma = self.gamma_dehnen_initial

        coef = (3 - gamma) * Ntot * r_a / (4 * np.pi)
        denom = (r ** gamma) * (r + r_a) ** (4 - gamma)

        n = coef / denom
        n = np.where(r >= r_k, n, 0.0)

        return n                              # (N,)


    # ---------------------------------------------------------------
    # Mass density ρ(r)
    # ---------------------------------------------------------------
    def mass_density(self, r_grid, Ntot, component_masses, kind='TDE', unit='Msun/pc^3'):
        r_grid = np.asarray(r_grid, float)       # (N, Nr)
        N, Nr = r_grid.shape

        comp_mass = np.asarray(component_masses, float)
        n_i = self.dehnen_number_density(r_grid, Ntot, kind)   # (N, Nr)

        if comp_mass.ndim == 1:
            rho = np.sum(comp_mass) * n_i                      # (N, Nr)
        else:
            rho = np.sum(comp_mass[:, :, None] * n_i[:, None, :], axis=1)

        if unit == 'Msun/pc^3':
            return rho
        elif unit == 'g/pc^3':
            return rho * Msun_to_grams
        else:
            raise ValueError("unit must be 'Msun/pc^3' or 'g/pc^3'.")


    # ---------------------------------------------------------------
    # Mass density at r_inf
    # ---------------------------------------------------------------
    def mass_density_at_rinfl(self, Ntot, component_masses, kvir=1.0,
                              kind='TDE', unit='Msun/pc^3'):
        r_inf = self.nsc.r_influence(kvir=kvir, unit='pc')   # (N,)
        n = self.dehnen_n_at_radius(r_inf, Ntot, kind)       # (N,)

        comp_mass = np.asarray(component_masses)

        if comp_mass.ndim == 1:
            rho = n * np.sum(comp_mass)
        else:
            rho = np.sum(comp_mass * n[:, None], axis=1)

        if unit == 'Msun/pc^3':
            return rho
        elif unit == 'g/pc^3':
            return rho * Msun_to_grams
        else:
            raise ValueError("Invalid unit for mass density.")

# ===============================================================
# Broadcasting Helpers (User utilities)
# ===============================================================

def ensure_2d(arr):
    """
    Ensures arr has shape (N, Nr).
    If arr is (Nr,), returns (1, Nr).
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def squeeze_if_single(arr):
    """
    If arr has shape (1, Nr), return (Nr,).
    Otherwise return arr unchanged.
    """
    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr[0]
    return arr


def vectorize_r_grid(r_grid, N):
    """
    Ensures r_grid has shape (N, Nr).
    Accepts (Nr,) or (1, Nr).
    """
    r_grid = np.asarray(r_grid)
    if r_grid.ndim == 1:
        return np.broadcast_to(r_grid, (N, len(r_grid)))
    if r_grid.shape[0] == 1 and N > 1:
        return np.broadcast_to(r_grid, (N, r_grid.shape[1]))
    return r_grid