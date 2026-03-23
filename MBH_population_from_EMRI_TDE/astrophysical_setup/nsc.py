import numpy as np 
from typing import Optional
from galaxy import Galaxy

from config import (pc_to_cm, G_pc3_per_Msun_yr2, c_pc_per_year, Msun, Rsun)


class NSC:
    """
    Nuclear Star Cluster
    """

    def __init__(self, galaxy: Galaxy, lgMBH: Optional[float] = None, MBH_params: Optional[dict] = None):

        self.gal = galaxy

        if MBH_params is None:
            MBH_params = dict(A=7.87, B=4.55, sigma_0=160.0, MBH_scatter=0.53)

        if lgMBH is None:
            self._lgMBH = self.gal.lgMBH_mass(
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
    def MBH_mass(self):
        return 10 ** self._lgMBH

    def r_influence(self, kvir=1.0, unit='pc'):
        """
        r_inf = G * M_bh / sigma^2
        """
        r_inf_pc = G_pc3_per_Msun_yr2 * self.MBH_mass / (self.gal.sigma(unit='pc/year')**2)
        if unit == 'pc':
            return r_inf_pc
        elif unit == 'cm':
            return r_inf_pc * pc_to_cm
        else:
            raise ValueError("unit must be 'pc' or 'cm'")

    def scale_radius(self, kvir=1.0, factor=4.0, unit='pc'):
        """Dehnen scale radius r_a = factor * r_h """
        return factor * self.r_influence(kvir=kvir, unit=unit)

    def r_capture(self, unit='pc'):
        """
        Newtonian direct-capture proxy for compact objects:
        r_sBH = 8 * G * M_bh / c^2
        """
        r_sBH_pc = 8.0 * G_pc3_per_Msun_yr2 * self.MBH_mass / (c_pc_per_year**2)
        if unit == 'pc':
            return r_sBH_pc
        elif unit == 'cm':
            return r_sBH_pc * pc_to_cm  # Convert back to cm
        else:
            raise ValueError("unit must be 'pc' or 'cm'")

    def r_tidal(self, m_star=1.0*Msun, R_star=1.0*Rsun, unit='pc'):
        """
        Stellar tidal disruption radius:
        r_t = R_star * (M_bh / m_star)^(1/3)
        """
        # THIS CALCULATION IS STILL WRONG SINCE THE R_STAR AND M_STAR IS INCORRECT
        # THIS SHOULD BE DISCUSSED IN THE NEXT MEETING
        rt_pc = R_star * (self.MBH_mass / m_star)**(1.0/3.0)
        if unit == 'pc':
            return rt_pc
        elif unit == 'cm':
            return rt_pc * pc_to_cm
        else:
            raise ValueError("unit must be 'pc' or 'cm'")

    def CO_mass_function(self):
        """
        Placeholder for compact object mass function in the NSC.
        Could be a power-law, log-normal, or based on stellar evolution models.
        """
        raise NotImplementedError("CO mass function not implemented yet.")