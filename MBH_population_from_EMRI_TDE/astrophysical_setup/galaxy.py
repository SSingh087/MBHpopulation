"""
================================================================================
galaxy.py
================================================================================

GALAXY INITIALIZATION AND PROPERTIES
This module handles galaxy stellar mass, velocity dispersion, and black hole
mass calculations using fundamental astrophysical relations.

KEY RELATIONS
─────────────────────────────────────────────────────────────────────────
1. Size-Mass Relation: log10(Re/kpc) = 0.24*(Mgal-10.7) + 0.82 + scatter
   INPUT: lgMgal (log10(M*) in Msun)
   OUTPUT: Re (half-light radius in kpc)
   SCATTER: 0.20 dex

2. Virial Relation: sigma = sqrt(G*M_enclosed / (kvir*Re))
   INPUT: M_enclosed (typically 0.5*Mstar), Re
   OUTPUT: sigma (velocity dispersion in km/s)

3. M-Sigma Relation: log10(MBH) = A + B*log10(sigma/sigma_0) + scatter
   INPUT: sigma (velocity dispersion in km/s)
   OUTPUT: MBH mass in Msun
   REFERENCES: arXiv:2205.06277 (Greene et al. style)
   SCATTER: 0.53 dex

4. Nucleation Fraction: f_NSC(Mgal) depends on galaxy mass
   INPUT: lgMgal
   OUTPUT: probability that galaxy hosts an NSC
   REFERENCES:
     - Hannah et al. arXiv:2407.10911 Fig 4
     - Neumayer et al. arXiv:2001.03626 Fig 3

MODES
─────────────────────────────────────────────────────────────────────────
Three initialization modes:
  Mode 1 (Forward): Provide lgMgal → computes sigma → computes MBH
  Mode 2 (Inverse):  Provide lgMBH + sigma_pc_yr → uses directly
  Mode 3 (Mixed):    Provide both lgMgal + lgMBH → treats independently

================================================================================
"""

import numpy as np
from typing import Optional
import warnings

from config import *


# _f_NSC_Hannah_   = lambda lgMgal: np.exp(-((lgMgal - 7.6)**4) / 400.0)  # arXiv:2407.10911 Fig 4
# _f_NSC_Neumayer_ = lambda lgMgal: np.exp(-((lgMgal - 9.0)**4) /  50.0)  # arXiv:2001.03626 Fig 3

_f_NSC_Hannah_   = lambda lgMgal: np.exp(-((lgMgal - 5.6)**4) / 1000.0)  # arXiv:2407.10911 Fig 4
_f_NSC_Neumayer_ = lambda lgMgal: np.exp(-((lgMgal - 9.0)**2) /  30.0)**2  # arXiv:2001.03626 Fig 3

class Galaxy:
    """
    Galaxy with size-mass, virial sigma, and M-sigma black hole relations.

    This class encapsulates galaxy properties and derived quantities based on
    stellar mass, with automatic computation of velocity dispersion and
    black hole mass using well-calibrated astrophysical relations.

    INITIALIZATION MODES
    ────────────────────────────────────────────────────────────────────────
    The class supports three initialization modes:

    MODE 1: FORWARD (lgMgal → sigma → MBH)
      lgMgal is provided, sigma and MBH are computed
      Inputs: lgMgal, z_gal, nucleation_occurs, ra_deg, dec_deg, rng
      Usage: gal = Galaxy(lgMgal=array, z_gal=z_array, ...)
      Automatically applies nucleation filtering

    MODE 2: INVERSE (lgMBH directly specified)
      lgMBH and sigma are provided externally
      Inputs: lgMBH, sigma_pc_yr, z_gal
      Usage: gal = Galaxy(lgMBH=array, sigma_pc_yr=array, z_gal=z_array)
      Skips M-sigma relation, assumes all objects are nucleated

    MODE 3: MIXED (both lgMgal and lgMBH specified)
      Both quantities provided independently for testing/PDF construction
      Inputs: lgMgal, lgMBH, z_gal
      Usage: gal = Galaxy(lgMgal=array, lgMBH=array, z_gal=z_array)
      Bypasses M-sigma relation; lgMgal and lgMBH are independent

    ATTRIBUTES AFTER INITIALIZATION
    ────────────────────────────────────────────────────────────────────────
    lgMgal : ndarray
        log10(stellar mass / Msun), shape (N_nucleated,)
    sigma_pc_yr : ndarray
        velocity dispersion in pc/year, shape (N_nucleated,)
    sigma_km_s : ndarray
        velocity dispersion in km/s, shape (N_nucleated,)
    lgMBH_mass : ndarray
        log10(MBH mass / Msun), shape (N_nucleated,)
    z_gal : ndarray
        galaxy redshift, shape (N_nucleated,)
    ra_deg : ndarray
        right ascension in degrees, shape (N_nucleated,)
    dec_deg : ndarray
        declination in degrees, shape (N_nucleated,)
    nucleation_occurs : ndarray
        boolean mask of which input galaxies became nucleated

    UNITS
    ────────────────────────────────────────────────────────────────────────
    sigma : pc/year (stored as sigma_pc_yr) or km/s (sigma_km_s)
    lgMgal : log10(Msun)
    lgMBH_mass : log10(Msun)
    Mass: Msun (log10 space)
    Radius: kpc, cm, or pc (depends on method)
    Redshift: dimensionless
    """

    def __init__(self, z_gal: np.ndarray, ra_deg: np.ndarray=None, dec_deg: np.ndarray=None, rng: Optional[np.random.Generator] = None, lgMgal: Optional[np.ndarray] = None, lgMBH: Optional[np.ndarray] = None, sigma_pc_yr: Optional[np.ndarray] = None, nucleation_occurs: bool = True, mbh_mass_method: str = 'use_sigma', nucleation_method: str = 'hannah'):
        """
        Initialize a Galaxy instance with stellar mass and derived properties.

        PARAMETERS
        ──────────────────────────────────────────────────────────────────────
        z_gal : ndarray
            Galaxy redshift (required for all modes)
            DIMENSION: (N_gal,)
            UNITS: dimensionless

        lgMgal : ndarray, optional
            log10(stellar mass / Msun) - used in FORWARD mode
            DIMENSION: (N_gal,)
            UNITS: dex (log10 scale)
            DEFAULT: None (triggers INVERSE mode if lgMgal is None and lgMBH is provided)

        lgMBH : ndarray, optional
            log10(black hole mass / Msun) - used in INVERSE or MIXED modes
            DIMENSION: (N_gal,)
            UNITS: dex (log10 scale)
            DEFAULT: None

        sigma_pc_yr : ndarray, optional
            Velocity dispersion in pc/year (INVERSE mode only)
            DIMENSION: (N_gal,)
            UNITS: pc/year
            DEFAULT: None (computed from M-Sigma in FORWARD mode)

        nucleation_occurs : ndarray or bool, optional
            Boolean mask indicating which galaxies are nucleated
            DIMENSION: (N_gal,) or scalar
            UNITS: dimensionless
            DEFAULT: True (all nucleated)

        ra_deg : ndarray, optional
            Right ascension in degrees
            DIMENSION: (N_gal,)
            UNITS: degrees [0, 360)
            DEFAULT: None (randomly generated)

        dec_deg : ndarray, optional
            Declination in degrees
            DIMENSION: (N_gal,)
            UNITS: degrees [-90, 90]
            DEFAULT: None (randomly generated)

        rng : numpy.random.Generator, optional
            Random number generator for reproducibility
            DEFAULT: numpy.random.default_rng() (system entropy)

        mbh_mass_method : str, optional
            Method to compute MBH: use either 'use_sigma' (compute from sigma) or 'use_lgMgal' 
        nucleation_method : str, optional
            Method to determine nucleation fraction: 'hannah' or 'neumayer'

        INTERNAL DERIVATIONS
        ──────────────────────────────────────────────────────────────────────
        The class automatically:
          1. Computes velocity dispersion from size-mass relation (FORWARD mode)
          2. Derives black hole mass from M-sigma relation (FORWARD mode)
          3. Applies nucleation filtering based on f_NSC(Mgal)
          4. Converts units between pc/year and km/s as needed
          5. Stores scatter as part of object for consistency

        RAISES
        ──────────────────────────────────────────────────────────────────────
        ValueError
            If neither lgMgal nor lgMBH is provided (ambiguous mode)
        """

        self.rng = rng if rng is not None else np.random.default_rng()
        self.z_gal = np.asarray(z_gal)

        # Determine which M-sigma relation to use based on the method parameter
        
        if mbh_mass_method == 'use_sigma':
            self.MBH_A = MBH_A_sigma
            self.MBH_B = MBH_B_sigma
            self.MBH_scatter = MBH_scatter_sigma
        elif mbh_mass_method == 'use_lgMgal':
            self.MBH_A = MBH_A_LGMGAL
            self.MBH_B = MBH_B_LGMGAL
            self.MBH_scatter = MBH_scatter_LGMGAL
        else:
            raise ValueError("Invalid method. Must be 'use_sigma' or 'use_lgMgal'.")

        if lgMgal is None and lgMBH is None:
            raise ValueError("Must specify either lgMgal or lgMBH. They are related by the M-sigma relation. Specify one or the other, or neither to compute lgMBH from lgMgal.")

        # Case 1: MBH → sigma (inverse deterministic mode with both lgMgal and lgMBH)
        if lgMgal is not None and lgMBH is not None:
            warnings.warn("[INFO] : Both lgMgal and lgMBH are provided. This will ignore the M-sigma relation and treat them as independent. Ensure this is intended behavior. Also one should not attempt to use the M-sigma relation in this mode since it will be inconsistent with the provided lgMBH and lgMgal. This mode is primarily for testing and PDF construction where we want to condition on both properties independently.")
            self.lgMBH_mass = np.asarray(lgMBH)
            self.lgMgal = np.asarray(lgMgal)

            # No nucleation randomness in since we are conditioning on lgMBH which implies a nucleated galaxy. So we can set nucleation_occurs=True for all of them.
            self.nucleation_occurs = np.ones_like(self.lgMBH_mass, dtype=bool)

            self.sigma_pc_yr = np.asarray(sigma_pc_yr) if sigma_pc_yr is not None else self.sigma(unit='pc/year')
            self.sigma_km_s = self.sigma_pc_yr * pc_to_cm / sec_per_year / 1e5  # convert to km/s


        # Case 2: Only lgMBH provided (inverse mode - MBH/sigma specified externally)
        elif lgMBH is not None and lgMgal is None and mbh_mass_method == 'use_sigma':
            print("[INFO] : Only lgMBH is provided. This will compute lgMgal and sigma from lgMBH using the M-sigma relation.")
            self.lgMBH_mass = np.asarray(lgMBH)
            self.sigma_pc_yr = np.asarray(sigma_pc_yr)
            self.sigma_km_s = self.sigma_pc_yr * pc_to_cm / sec_per_year / 1e5  # convert to km/s
            self.lgMgal = None  # Not known in this mode

            # All objects are nucleated in this mode since we are conditioning on lgMBH
            self.nucleation_occurs = np.ones_like(self.lgMBH_mass, dtype=bool)
        
        elif lgMBH is not None and lgMgal is None and mbh_mass_method == 'use_lgMgal':
            print("[INFO] Only lgMBH provided (use_Mstar relation): inverting the "
                  "M_BH-M* relation to recover implied lgMgal, then running that "
                  "through the standard size-mass + virial pipeline for sigma.")
            self.lgMBH_mass = np.asarray(lgMBH)
            self.nucleation_occurs = np.ones_like(self.lgMBH_mass, dtype=bool)
            self.lgMgal = (self.lgMBH_mass - self.MBH_A) / self.MBH_B + np.log10(MGAL0)
 

        # Case 3: Mgal → sigma, MBH (forward galaxy model)
        elif lgMgal is not None and lgMBH is None:
            print("[INFO] : Mgal is provided. This will compute sigma and MBH from lgMgal using the {} relation.".format(mbh_mass_method))
            self.lgMgal = np.array(lgMgal)

            # vectorized nucleation flags
            if nucleation_occurs is None:
                print("[INFO] : nuclating galaxies based on the {} nucleation fraction model.".format(nucleation_method))
                nucleation_occurs = Galaxy.check_nucleation(self.lgMgal, method=nucleation_method, rng=self.rng)
            else:
                print("[WARNING] : nucleation_occurs is provided. This will override the default nucleation fraction model and use the provided boolean mask. Ensure this is intended behavior.")
                nucleation_occurs = np.asarray(nucleation_occurs, dtype=bool)
            self.nucleation_occurs = nucleation_occurs

            # this ensures we use the same properties all the time and not apply scatter differently across different method calls. The scatter is applied once here in the constructor and then stored as an attribute for consistency.

            # we also here care for galaxies which are nucleated.
            self.lgMgal = self.lgMgal[self.nucleation_occurs]
            self.sigma_pc_yr = self.sigma(unit='pc/year')
            self.sigma_km_s = self.sigma_pc_yr * pc_to_cm / sec_per_year / 1e5  # convert to km/s
            self.lgMBH_mass = self.lgMBH(sigma_0=MBH_sigma0, unit='Msun', mbh_mass_method=mbh_mass_method)
            self.z_gal = self.z_gal[self.nucleation_occurs]

        if ra_deg is None and dec_deg is None:
            self.ra_deg, self.dec_deg = self.sky_location().T
        else:
            ra_deg = np.asarray(ra_deg)
            dec_deg = np.asarray(dec_deg)
            self.ra_deg = ra_deg
            self.dec_deg = dec_deg
        
        # print(f"Number of galaxies: {self.lgMgal.shape[0]} with nucleation_occurs={self.nucleation_occurs.sum()} out of {self.lgMgal.shape[0]} total.")

    @staticmethod
    def check_nucleation(cls, lgMgal, nucleation_method='hannah'):
        """
        Determine which galaxies are nucleated based on stellar mass.

        This method applies nucleation probability models and stochastically
        determines which galaxies in a sample host nuclear star clusters (NSCs).

        NUCLEATION MODEL
        ────────────────────────────────────────────────────────────────────────
        Probability of NSC is determined by:
            f_NSC(Mgal) = max(f_Hannah(Mgal), f_Neumayer(Mgal))

        where:
            f_Hannah(Mgal) ∝ exp[-((Mgal - 7.6)^4 / 400)]
            f_Neumayer(Mgal) ∝ exp[-((Mgal - 9.0)^4 / 50)]

        Each galaxy is then nucleated with probability f_NSC(Mgal).

        INPUTS
        ────────────────────────────────────────────────────────────────────────
        lgMgal : ndarray
            log10(stellar mass / Msun)
            DIMENSION: (N_gal,)
            UNITS: dex

        nucleation_method : str, optional
            Method to determine nucleation fraction: 'hannah' or 'neumayer'
            DEFAULT: 'hannah'

        OUTPUTS
        ────────────────────────────────────────────────────────────────────────
        mask : ndarray (bool)
            Boolean array indicating nucleated galaxies
            DIMENSION: (N_gal,)
            UNITS: True/False

        EXAMPLE
        ──────────────────────────────────────────────────────────────────────
        >>> lgMgal = np.array([9.0, 10.0, 11.0])
        >>> z_gal = np.array([0.1, 0.1, 0.1])
        >>> gal = Galaxy(lgMgal=lgMgal, z_gal=z_gal)
        >>> nucleated_mask = Galaxy.check_nucleation(lgMgal, nucleation_method='hannah')
        >>> nucleated_lgMgal = lgMgal[nucleated_mask]
        >>> nucleated_z_gal = z_gal[nucleated_mask]
        >>> gal_nucleated = Galaxy(lgMgal=nucleated_lgMgal, z_gal=nucleated_z_gal)
        """
        lgMgal = np.array(lgMgal)

        # Stochastic sampling: draw uniform random numbers
        f = np.random.uniform(0.0, 1.0, size=lgMgal.shape)
        if nucleation_method == 'hannah':
            p = _f_NSC_Hannah_(lgMgal)
        elif nucleation_method == 'neumayer':
            p = _f_NSC_Neumayer_(lgMgal)
        else:
            raise ValueError("Invalid method. Must be 'hannah' or 'neumayer'.")
        
        mask = f < p
        return mask

    def R_eff(self, lg_A=0.82, B=0.24, Re_scatter=0.20, unit='kpc'):
        """
        Half-light (effective) radius from size-mass relation.

        SCALING RELATION
        ────────────────────────────────────────────────────────────────────────
        The relation between galaxy size and stellar mass follows:

            log10(Re / kpc) = B * (lgMgal - 10.7) + lg_A + N(0, Re_scatter)

        where:
          - Re is the effective (half-light) radius
          - Mgal is the stellar mass
          - Scatter is Gaussian in log-space

        INPUTS
        ────────────────────────────────────────────────────────────────────────
        lg_A : float, optional
            Logarithmic intercept (default: 0.82)
            UNITS: dex

        B : float, optional
            Slope parameter (default: 0.24)
            UNITS: dimensionless (dex/dex)

        Re_scatter : float, optional
            Gaussian scatter in log10(Re) (default: 0.20)
            UNITS: dex

        unit : str, optional
            Output unit: 'kpc' or 'cm'
            DEFAULT: 'kpc'

        OUTPUTS
        ────────────────────────────────────────────────────────────────────────
        Re : ndarray
            Half-light radius for each galaxy
            DIMENSION: (N_nucleated,)
            UNITS: kpc or cm (as specified)

        EXAMPLE
        ──────────────────────────────────────────────────────────────────────
        >>> gal = Galaxy(lgMgal=np.array([10.0, 11.0]), z_gal=np.array([0.1, 0.1]))
        >>> Re_kpc = gal.R_eff(unit='kpc')
        >>> Re_cm = gal.R_eff(unit='cm')
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
        Compute velocity dispersion from virial relation.

        PHYSICAL RELATION
        ────────────────────────────────────────────────────────────────────────
        The velocity dispersion is derived from the virial theorem:

            sigma = sqrt(G * M_enclosed / (kvir * Re))

        where:
          - M_enclosed = 0.5 * Mstar (assumed half the stellar mass within Re)
          - kvir is a virial dimensionless factor (typically 1.0 or 2.0)
          - Re is the effective radius from size-mass relation

        INPUTS
        ────────────────────────────────────────────────────────────────────────
        kvir : float, optional
            Virial factor. Typical values:
              kvir = 1.0 → singular isothermal sphere (default)
              kvir = 2.0 → typical galaxy virial models
            DEFAULT: 1.0
            UNITS: dimensionless

        unit : str, optional
            Output unit for velocity dispersion:
              'km/s'    → kilometers per second (most common)
              'cm/s'    → centimeters per second (CGS)
              'pc/year' → parsecs per year (astronomical)
            DEFAULT: 'km/s'

        OUTPUTS
        ────────────────────────────────────────────────────────────────────────
        sigma : ndarray
            Velocity dispersion for each galaxy
            DIMENSION: (N_nucleated,)
            UNITS: km/s, cm/s, or pc/year (as specified)

        INTERNAL COMPUTATION
        ────────────────────────────────────────────────────────────────────────
        1. Retrieves Re from size-mass relation in cm
        2. Computes M_enclosed = 0.5 * Mstar (in grams)
        3. Applies G_cgs (cm^3 g^-1 s^-2) virial formula
        4. Returns result in CGS units (cm/s)
        5. Converts to requested unit

        EXAMPLE
        ──────────────────────────────────────────────────────────────────────
        >>> gal = Galaxy(lgMgal=np.array([10.0, 11.0]), z_gal=np.array([0.1, 0.1]))
        >>> sigma_km_s = gal.sigma(kvir=1.0, unit='km/s')
        >>> sigma_pc_yr = gal.sigma(kvir=1.0, unit='pc/year')
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

    def lgMBH(self, sigma_0=MBH_sigma0, unit='Msun', mbh_mass_method='use_sigma'):
        """
        Compute black hole mass from M-sigma scaling relation.

        PHYSICAL RELATION (M-SIGMA)
        ────────────────────────────────────────────────────────────────────────
        The relation between supermassive black hole (SMBH) mass and galaxy
        velocity dispersion follows:

            log10(MBH/Msun) = A + B*log10(sigma/sigma_0) + scatter

        where:
          - MBH is the black hole mass
          - sigma is the galaxy velocity dispersion
          - sigma_0 is a reference velocity dispersion
          - scatter is Gaussian intrinsic scatter in log-space

        REFERENCE: arXiv:2205.06277 (Greene et al. style M-sigma relation)

        INPUTS
        ────────────────────────────────────────────────────────────────────────
        A : float, optional
            M-sigma intercept (dimensionless)
            DEFAULT: 7.87 (from config.py)
            UNITS: dex

        B : float, optional
            M-sigma slope (dimensionless, dex/dex)
            DEFAULT: 4.55 (from config.py)
            UNITS: dex per dex

        sigma_0 : float, optional
            Reference velocity dispersion
            DEFAULT: 160.0 km/s (from config.py)
            UNITS: km/s
            Note: Some studies use 70 km/s instead, which significantly
                  affects both Gamma_E (peak rate) and t_EMRI (peak time)

        MBH_scatter : float, optional
            Intrinsic scatter (standard deviation in log-space)
            DEFAULT: 0.53 dex (from config.py)
            UNITS: dex

        unit : str, optional
            Output unit (currently only 'Msun' supported)
            DEFAULT: 'Msun'

        mbh_mass_method : str, required
            Method to compute MBH:
              'use_sigma' → compute MBH from sigma derived from lgMgal
              'use_lgMgal' → use provided lgMgal directly (no computation)

        OUTPUTS
        ────────────────────────────────────────────────────────────────────────
        lgMBH : ndarray
            log10(MBH mass / Msun) for each galaxy
            DIMENSION: (N_nucleated,)
            UNITS: dex (log10 scale)

        INTERNAL COMPUTATION
        ────────────────────────────────────────────────────────────────────────
        1. Retrieves sigma_km_s from virial relation
        2. Applies M-sigma power law: A + B*log10(sigma/sigma_0)
        3. Adds Gaussian scatter with std=MBH_scatter
        4. Returns log10(M_BH/Msun)

        EXAMPLE
        ──────────────────────────────────────────────────────────────────────
        >>> gal = Galaxy(lgMgal=np.array([10.0, 11.0]), z_gal=np.array([0.1, 0.1]))
        >>> lgMBH_mass = gal.lgMBH(A=7.87, B=4.55, sigma_0=160.0)
        >>> MBH_mass = 10**lgMBH_mass  # Convert to linear Msun

        DEPENDENCIES
        ────────────────────────────────────────────────────────────────────────
        Requires sigma in km/s, which is computed from:
          1. R_eff (size-mass relation)
          2. Virial relation with stellar mass
        """
        if mbh_mass_method == 'use_sigma':
            print("[INFO] : Computing lgMBH from sigma using the M-sigma relation.")
            sigma = self.sigma(unit='km/s')
            lgMBH = self.MBH_A + self.MBH_B * np.log10(sigma / MBH_sigma0)
            lgMBH += np.random.normal(0.0, self.MBH_scatter, size=sigma.shape)
            return lgMBH
        elif mbh_mass_method == 'use_lgMgal':
            print("[INFO] : Using provided lgMgal directly without computing from sigma.")
            lgMBH = self.MBH_A + self.MBH_B * np.log10(10**self.lgMgal / MGAL0)
            lgMBH += np.random.normal(0.0, self.MBH_scatter, size=self.lgMgal.shape)
            return lgMBH
        else:
            raise ValueError("Invalid method. Must be 'use_sigma' or 'use_lgMgal'.")

    
    @staticmethod
    def lgMBH_from_Mgal(lgMgal, A, B, sigma_0=MBH_sigma0):
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
        # Use lgMgal if available, otherwise use lgMBH_mass length
        size = len(self.lgMgal) if self.lgMgal is not None else len(self.lgMBH_mass)
        ra_deg = self.rng.uniform(0.0, 360.0, size=size)
        dec_deg = self.rng.uniform(-90.0, 90.0, size=size)
        return np.column_stack((ra_deg, dec_deg))