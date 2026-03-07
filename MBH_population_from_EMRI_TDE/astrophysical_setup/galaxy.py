
from utils import *

_f_NSC_Hannah_   = lambda lgMgal: np.exp(-((lgMgal - 7.6)**4) / 400.0)  # arXiv:2407.10911 Fig 4
_f_NSC_Neumayer_ = lambda lgMgal: np.exp(-((lgMgal - 9.0)**4) /  50.0)  # arXiv:2001.03626 Fig 3

class Galaxy:
    """
    Galaxy with simple size-mass, virial sigma, and M-sigma MBH.
    """

    def __init__(self, lgMgal, nucleation_occurs=True):
        """
        Parameters
        ----------
        lgMgal : float
            log10(Mstar/Msun)
        nucleation_occurs : bool
            Whether nucleation occurs for this galaxy (flag decided externally).
        """
        self.lgMgal = float(lgMgal)
        self.nucleation_occurs = bool(nucleation_occurs)

    @classmethod
    def check_nucleation(cls, lgMgal):
        """
        Factory that returns a Galaxy instance if nucleation occurs, else None.
        This mirrors your procedural sampling logic.
        """
        f = np.random.random()  # scalar in [0,1)

        if f < _f_NSC_Hannah_(lgMgal):
            # print(f"nucleation will occur for {lgMgal} (Hannah)")
            return cls(lgMgal, nucleation_occurs=True)
        elif f < _f_NSC_Neumayer_(lgMgal):
            # print(f"nucleation will occur for {lgMgal} (Neumayer)")
            return cls(lgMgal, nucleation_occurs=True)
        else:
            # print(f"NO nucleation for {lgMgal}")
            return None

    def R_eff(self, lg_A=0.82, B=0.24, Re_scatter=0.20, unit='kpc'):
        """
        Half-light (effective) radius from size-mass relation:
        log10(Re/kpc) = B * (lgMgal - 10.7) + lg_A + N(0, Re_scatter)
        """
        lgreff = B * (self.lgMgal - 10.7) + lg_A
        lgreff += np.random.normal(0.0, Re_scatter)
        Re_kpc = 10.0**lgreff

        if unit == 'kpc':
            return Re_kpc
        elif unit == 'cm':
            return Re_kpc * kpc_to_cm  # utils should define `kpc` in cm
        else:
            raise ValueError("Invalid unit. Must be 'kpc' or 'cm'")

    def sigma(self, kvir=1.0, unit='km/s'):
        """
        Virial sigma ~ sqrt( G * M_enclosed / (kvir * Re) ).
        Here use M_enclosed = 0.5 * Mstar (in grams), Re in cm.
        """
        Re_cm = self.R_eff(unit='cm')
        Mstar_linear = 10.0**self.lgMgal
        M_enclosed_grams = 0.5 * Mstar_linear * Msun

        sig_cgs = np.sqrt(G * M_enclosed_grams / (kvir * Re_cm))  # cm/s

        if unit == 'km/s':
            return sig_cgs / 1e5
        elif unit == 'cm/s':
            return sig_cgs
        else:
            raise ValueError("Invalid unit. Must be 'km/s' or 'cm/s'")

    def lgMBH_mass(self, A=7.87, B=4.55, sigma_0=160.0, MBH_scatter=0.53):
        """
        log10(MBH/Msun) from M-sigma; add Gaussian scatter.
        Defaults match your Greene+20-like parameters.
        """
        s = self.sigma(unit='km/s')
        lgMBH = A + B * np.log10(s / sigma_0)
        lgMBH += np.random.normal(0.0, MBH_scatter)
        return lgMBH


# class NSC(Galaxy):
#     """
#     Nuclear Star Cluster wrapper that extends a galaxy with BH + NSC-related scales.

#     """

#     def __init__(self, lgMgal, lgMBH=None, MBH_params=None):
#         super().__init__(lgMgal)  # sets self.lgMgal and self.nucleation_occurs

#         # Store a fixed BH mass for this NSC instance
#         if MBH_params is None:
#             MBH_params = dict(A=7.87, B=4.55, sigma_0=160.0, MBH_scatter=0.53)

#         if lgMBH is None:
#             # Sample only once, keep it fixed on the object
#             self._lgMBH = super().lgMBH_mass(
#                 A=MBH_params.get('A', 7.87),
#                 B=MBH_params.get('B', 4.55),
#                 sigma_0=MBH_params.get('sigma_0', 160.0),
#                 MBH_scatter=MBH_params.get('MBH_scatter', 0.53)
#             )
#         else:
#             self._lgMBH = float(lgMBH)

#     # ----------------- Convenience properties -----------------
#     @property
#     def lgMBH(self):
#         """log10(M_BH/Msun), fixed for this NSC instance."""
#         return self._lgMBH

#     @property
#     def Mbh_grams(self):
#         """BH mass in grams."""
#         return (10.0**self._lgMBH) * Msun

#     def sigma_cms(self, kvir=1.0):
#         """
#         Host velocity dispersion in cm/s. Reuses galaxies.sigma().
#         Falls back to km/s -> cm/s conversion if your galaxies.sigma only supports 'km/s'.
#         """
#         try:
#             s = super().sigma(kvir=kvir, unit='cm/s')
#             return s
#         except Exception:
#             s_kms = super().sigma(kvir=kvir, unit='km/s')
#             return s_kms * 1e5

#     # ----------------- NSC / BH characteristic radii -----------------
#     def influence_radius(self, kvir=1.0, unit='pc'):
#         """
#         BH influence radius:
#             r_h = G * M_bh / sigma^2
#         Returns
#         -------
#         float
#             r_h in requested unit ('pc' or 'cm').
#         """
#         M = self.Mbh_grams
#         sigma = self.sigma_cms(kvir=kvir)  # cm/s
#         rh_cm = G * M / (sigma**2)         # cm
#         if unit == 'pc':
#             return rh_cm / pc
#         elif unit == 'cm':
#             return rh_cm
#         else:
#             raise ValueError("unit must be 'pc' or 'cm'")

#     def scale_radius_ra(self, kvir=1.0, factor=4.0, unit='pc'):
#         """
#         Dehnen scale radius used by Broggi+:
#             r_a = factor * r_h   (default factor=4)
#         Returns r_a in 'pc' or 'cm'.
#         """
#         rh = self.influence_radius(kvir=kvir, unit=unit)
#         return factor * rh

#     def capture_radius_compact(self, unit='pc'):
#         """
#         Direct-capture (compact object) radius (Newtonian proxy):
#             r_BH = 8 * G * M_bh / c^2
#         Returns r_BH in 'pc' or 'cm'.
#         """
#         M = self.Mbh_grams
#         r_cm = 8.0 * G * M / (c**2)
#         if unit == 'pc':
#             return r_cm / pc
#         elif unit == 'cm':
#             return r_cm
#         else:
#             raise ValueError("unit must be 'pc' or 'cm'")

#     def tidal_radius_star(self, m_star=1.0*Msun, R_star=1.0*Rsun, unit='pc'):
#         """
#         Stellar tidal disruption radius:
#             r_t = R_star * (M_bh / m_star)^(1/3)
#         Returns r_t in 'pc' or 'cm'.
#         """
#         M = self.Mbh_grams
#         rt_cm = R_star * (M / m_star)**(1.0/3.0)
#         if unit == 'pc':
#             return rt_cm / pc
#         elif unit == 'cm':
#             return rt_cm
#         else:
#             raise ValueError("unit must be 'pc' or 'cm'")