
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
        M_enclosed_grams = 0.5 * Mstar_linear * Msolarmass_to_grams

        sig_cgs = np.sqrt(G_cgs * M_enclosed_grams / (kvir * Re_cm))  # cm/s

        if unit == 'km/s':
            return sig_cgs / 1e5
        elif unit == 'pc/year':
            return sig_cgs / 1e3 / c_pc_per_year  # pc/year from m/s
        else:
            raise ValueError("Invalid unit. Must be 'km/s' or 'pc/year'")

    def lgMBH_mass(self, A=7.87, B=4.55, sigma_0=160.0, MBH_scatter=0.53, unit='solarmass'):
        """
        log10(MBH/Msun) from M-sigma; add Gaussian scatter.
        Defaults match your Greene+20-like parameters.
        """
        sigma = self.sigma(unit='km/s')
        lgMBH = A + B * np.log10(sigma / sigma_0)
        lgMBH += np.random.normal(0.0, MBH_scatter)
        return lgMBH



class NSC(Galaxy):
    """
    Nuclear Star Cluster / BH scales built on top of Galaxy.
    """

    def __init__(self, lgMgal, nucleation_occurs=True, lgMBH=None, MBH_params=None):
        """
        Parameters
        ----------
        lgMgal : float
            log10(Mstar/Msun) for host.
        nucleation_occurs : bool
            For completeness; typically True if created via check_nucleation.
        lgMBH : float or None
            If provided, fixes log10(MBH/Msun). Else sampled via M-sigma.
        MBH_params : dict or None
            Keys: A, B, sigma_0, MBH_scatter
        """
        super().__init__(lgMgal, nucleation_occurs=nucleation_occurs)

        if MBH_params is None:
            MBH_params = dict(A=7.87, B=4.55, sigma_0=160.0, MBH_scatter=0.53)

        if lgMBH is None:
            self._lgMBH = super().lgMBH_mass(
                A=MBH_params.get('A', 7.87),
                B=MBH_params.get('B', 4.55),
                sigma_0=MBH_params.get('sigma_0', 160.0),
                MBH_scatter=MBH_params.get('MBH_scatter', 0.53),
            )
        else:
            self._lgMBH = float(lgMBH)

    @property
    def lgMBH(self):
        """log10(MBH/Msun) fixed for this NSC instance."""
        return self._lgMBH

    @property
    def Mbh_grams(self):
        """BH mass in grams."""
        return (10.0**self._lgMBH) * Msun

    def sigma_cms(self, kvir=1.0):
        """Velocity dispersion of host in cm/s."""
        try:
            return super().sigma(kvir=kvir, unit='cm/s')
        except Exception:
            return super().sigma(kvir=kvir, unit='km/s') * 1e5

    # --- BH/NSC characteristic radii ---

    def influence_radius(self, kvir=1.0, unit='pc'):
        """
        r_h = G * M_bh / sigma^2  (Broggi Eq. 12 form; standard definition)
        """
        M = self.Mbh_grams
        sigma = self.sigma_cms(kvir=kvir)  # cm/s
        rh_cm = G_cgs * M / (sigma**2)         # cm
        if unit == 'pc':
            return rh_cm / pc
        elif unit == 'cm':
            return rh_cm
        else:
            raise ValueError("unit must be 'pc' or 'cm'")

    def scale_radius(self, kvir=1.0, factor=4.0, unit='pc'):
        """Dehnen scale radius r_a = factor * r_h (Broggi Sec. 3.1; default factor=4)."""
        rh = self.influence_radius(kvir=kvir, unit=unit)
        return factor * rh

    def capture_radius(self, unit='pc'):
        """
        Newtonian direct-capture proxy for compact objects:
        r_BH = 8 * G * M_bh / c^2
        """
        r_cm = 8.0 * G_cgs * self.Mbh_grams / (c_cgs**2)
        if unit == 'pc':
            return r_cm / pc_to_cm
        elif unit == 'cm':
            return r_cm
        else:
            raise ValueError("unit must be 'pc' or 'cm'")

    def tidal_radius_star(self, m_star=1.0*Msun, R_star=1.0*Rsun, unit='pc'):
        """
        Stellar tidal disruption radius:
        r_t = R_star * (M_bh / m_star)^(1/3)
        """
        rt_cm = R_star * (self.Mbh_grams / m_star)**(1.0/3.0)
        if unit == 'pc':
            return rt_cm / pc_to_cm
        elif unit == 'cm':
            return rt_cm
        else:
            raise ValueError("unit must be 'pc' or 'cm'")