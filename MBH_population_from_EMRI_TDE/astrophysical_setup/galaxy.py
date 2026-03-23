import numpy as np
from typing import Optional

from config import *

_f_NSC_Hannah_   = lambda lgMgal: np.exp(-((lgMgal - 7.6)**4) / 400.0)  # arXiv:2407.10911 Fig 4
_f_NSC_Neumayer_ = lambda lgMgal: np.exp(-((lgMgal - 9.0)**4) /  50.0)  # arXiv:2001.03626 Fig 3

class Galaxy:
    """
    Galaxy with simple size-mass, virial sigma, and M-sigma MBH.
    """

    def __init__(self, lgMgal: float, z_gal: float, rng: Optional[np.random.Generator] = None, nucleation_occurs: bool = True):
        """
        Parameters
        ----------
        lgMgal : float
            log10(Mstar/Msun)
        nucleation_occurs : bool
            Whether nucleation occurs for this galaxy (flag decided externally).
        """
        self.lgMgal = float(lgMgal)
        self.z_gal = float(z_gal)
        self.nucleation_occurs = bool(nucleation_occurs)
        self.rng = rng if rng is not None else np.random.default_rng()

    @classmethod
    def check_nucleation(cls, lgMgal, z_gal):
        """
        returns a Galaxy instance if nucleation occurs, else None.
        """
        f = np.random.random()  # scalar in [0,1)

        if f < _f_NSC_Hannah_(lgMgal):
            # print(f"nucleation will occur for {lgMgal} (Hannah)")
            return cls(lgMgal, z_gal, nucleation_occurs=True)
        elif f < _f_NSC_Neumayer_(lgMgal):
            # print(f"nucleation will occur for {lgMgal} (Neumayer)")
            return cls(lgMgal, z_gal, nucleation_occurs=True)
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
            return Re_kpc * kpc_to_cm
        else:
            raise ValueError("Invalid unit. Must be 'kpc' or 'cm'")

    def sigma(self, kvir=1.0, unit='km/s'):
        """
        Virial sigma ~ sqrt( G * M_enclosed / (kvir * Re) ).
        Here use M_enclosed = 0.5 * Mstar (in grams), Re in cm.
        """
        Re_cm = self.R_eff(unit='cm')
        Mstar_linear = 10.0**self.lgMgal
        M_enclosed_grams = 0.5 * Mstar_linear * Msun_to_grams

        sig_cgs = np.sqrt(G_cgs * M_enclosed_grams / (kvir * Re_cm))  # cm/s

        if unit == 'km/s':
            return sig_cgs / 1e5
        if unit == 'cm/s':
            return sig_cgs
        elif unit == 'pc/year':
            return sig_cgs * sec_per_year / pc_to_cm  # pc/year from m/s
        else:
            raise ValueError("Invalid unit. Must be 'km/s' or 'pc/year'")

    def lgMBH_mass(self, A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter, unit='Msun'):
        """
        log10(MBH/Msun) from M-sigma; add Gaussian scatter.
        Defaults match your Greene+20-like parameters.
        """
        sigma = self.sigma(unit='km/s')
        lgMBH = A + B * np.log10(sigma / sigma_0)
        lgMBH += np.random.normal(0.0, MBH_scatter)
        return lgMBH

    def lgMBH_from_Mgal(self, lgMgal, A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0):
        """
        Compute MBH for an arbitrary galaxy mass (log Mgal).
        Skips the nucleation check and uses the same sigma-Mstar relation
        as the instance method, but does not depend on self.lgMgal.
        """

        lgreff = 0.24*(lgMgal - 10.7) + 0.82
        Re_kpc = 10**lgreff
        Re_cm = Re_kpc * kpc_to_cm

        Mstar = 10**lgMgal
        M_enc = 0.5 * Mstar * Msun_to_grams

        sig_cgs = np.sqrt(G_cgs * M_enc / Re_cm)
        sigma = sig_cgs / 1e5  # km/s

        # donot apply scatter here since this is used for the grid calculation in cosmology.py and we want a deterministic grid. Scatter can be applied later when sampling from the interpolator.
        return A + B * np.log10(sigma / sigma_0)

    @staticmethod
    def lgMgal_from_lgMBH(lgMBH, A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0):
        """
        Analytic inversion of the M-sigma relation used in lgMBH_from_Mgal.
        Returns log10(Mgal/Msun).
        """

        # reproduce sigma(lgMgal) = C * 10^(beta * lgMgal)
        # using same coefficients as R_eff and M_enc relations
        beta = 0.5 - 0.12   # = 0.38
        const_sigma = -0.41 + 1.284  # constants from Re and M_enc pieces

        # from MBH = A + B*(log10(sigma) − log10(sigma0)):
        log_sigma = (lgMBH - A)/B + np.log10(sigma_0)

        # invert log_sigma = const_sigma + beta * lgMgal
        lgMgal = (log_sigma - const_sigma)/beta

        return lgMgal

