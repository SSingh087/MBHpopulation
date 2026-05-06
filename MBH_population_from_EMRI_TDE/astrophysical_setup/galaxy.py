import numpy as np
from typing import Optional
import warnings

from config import *

_f_NSC_Hannah_   = lambda lgMgal: np.exp(-((lgMgal - 7.6)**4) / 400.0)  # arXiv:2407.10911 Fig 4
_f_NSC_Neumayer_ = lambda lgMgal: np.exp(-((lgMgal - 9.0)**4) /  50.0)  # arXiv:2001.03626 Fig 3

class Galaxy:
    """
    Galaxy with simple size-mass, virial sigma, and M-sigma MBH.
    """

    def __init__(self, z_gal: np.ndarray, ra: np.ndarray=None, dec: np.ndarray=None, rng: Optional[np.random.Generator] = None, lgMgal: Optional[np.ndarray] = None, lgMBH: Optional[np.ndarray] = None, sigma_pc_yr: Optional[np.ndarray] = None, nucleation_occurs: bool = True):
        """
        Parameters
        ----------
        lgMgal : float
            log10(Mstar/Msun)
        nucleation_occurs : bool
            Whether nucleation occurs for this galaxy (flag decided externally).
        """

        self.rng = rng if rng is not None else np.random.default_rng()
        self.z_gal = np.asarray(z_gal)

        if lgMgal is None and lgMBH is None:
            raise ValueError("Must specify either lgMgal or lgMBH. They are related by the M-sigma relation. Specify one or the other, or neither to compute lgMBH from lgMgal.")

        # Case 1: MBH → sigma, Mgal   (inverse deterministic mode)
        if lgMgal is not None and lgMBH is not None:
            warnings.warn("Both lgMgal and lgMBH are provided. This will ignore the M-sigma relation and treat them as independent. Ensure this is intended behavior.")
            self.lgMBH_mass = np.asarray(lgMBH)
            self.sigma_pc_yr = np.asarray(sigma_pc_yr)
            self.sigma_km_s = self.sigma_pc_yr * pc_to_cm / sec_per_year / 1e5  # convert to km/s
            self.lgMgal = np.asarray(lgMgal)

            # No nucleation randomness in PDF mode since we are conditioning on lgMBH which implies a nucleated galaxy. So we can set nucleation_occurs=True for all of them.
            self.nucleation_occurs = np.ones_like(self.lgMBH_mass, dtype=bool)

        # Case 2: Mgal → sigma, MBH (forward galaxy model)
        elif lgMgal is not None and lgMBH is None:
            self.lgMgal = np.array(lgMgal)

            # vectorized nucleation flags
            if nucleation_occurs is None:
                self.nucleation_occurs = np.ones(self.lgMgal.shape[0], dtype=bool)
            else:
                self.nucleation_occurs = np.asarray(nucleation_occurs, dtype=bool)

            # this ensures we use the same properties all the time and not apply scatter differently across different method calls. The scatter is applied once here in the constructor and then stored as an attribute for consistency.

            # we also here care for galaxies which are nucleated.
            self.sigma_pc_yr = self.sigma(unit='pc/year')[self.nucleation_occurs]
            self.sigma_km_s = self.sigma_pc_yr * pc_to_cm / sec_per_year / 1e5  # convert to km/s
            self.lgMBH_mass = self.lgMBH(A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter)[self.nucleation_occurs]
            self.z_gal = self.z_gal[self.nucleation_occurs]

        if ra is None and dec is None:
            self.ra, self.dec = self.sky_location().T
        else:
            self.ra = np.asarray(ra)
            self.dec = np.asarray(dec)
        
            # print(f"Number of galaxies: {self.lgMgal.shape[0]} with nucleation_occurs={self.nucleation_occurs.sum()} out of {self.lgMgal.shape[0]} total.")

    @classmethod
    def check_nucleation(cls, lgMgal, z_gal):
        """
        returns a Galaxy instance if nucleation occurs, else None.
        """
        lgMgal = np.array(lgMgal)
        z_gal = np.array(z_gal)

        p1 = _f_NSC_Hannah_(lgMgal)
        p2 = _f_NSC_Neumayer_(lgMgal)

        p = np.maximum(p1, p2)

        f = np.random.uniform(0.0, 1.0, size=lgMgal.shape)

        mask = f < p

        return mask


    def R_eff(self, lg_A=0.82, B=0.24, Re_scatter=0.20, unit='kpc'):
        """
        Half-light (effective) radius from size-mass relation:
        log10(Re/kpc) = B * (lgMgal - 10.7) + lg_A + N(0, Re_scatter)
        """
        lgreff = B * (self.lgMgal - 10.7) + lg_A
        lgreff += np.random.normal(0.0, Re_scatter, size=self.lgMgal.shape)
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

    def lgMBH(self, A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter, unit='Msun'):
        """
        log10(MBH/Msun) from M-sigma; add Gaussian scatter.
        Defaults match your Greene+20-like parameters.
        """
        sigma = self.sigma(unit='km/s')
        lgMBH = A + B * np.log10(sigma / sigma_0)
        lgMBH += np.random.normal(0.0, MBH_scatter, size=sigma.shape)
        return lgMBH

    @staticmethod
    def lgMBH_from_Mgal(lgMgal, A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0):
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

    def sky_location(self):
        """
        Random sky location in RA (0 to 360 degrees) and Dec (-90 to 90 degrees).
        """
        ra = self.rng.uniform(0.0, 360.0, size=len(self.lgMgal))
        dec = self.rng.uniform(-90.0, 90.0, size=len(self.lgMgal))
        return np.column_stack((ra, dec))