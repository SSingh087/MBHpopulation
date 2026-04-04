import numpy as np 
from typing import Optional
from galaxy import Galaxy

from config import (pc_to_cm, G_pc3_per_Msun_yr2, c_pc_per_year, Msun, Rsun)


class NSC:
    """
    Nuclear Star Cluster
    """

    def __init__(self, galaxy: Galaxy, lgMBH: Optional[np.ndarray] = None, MBH_params: Optional[dict] = None):

        self.gal = galaxy

        if MBH_params is None:
            MBH_params = dict(A=7.87, B=4.55, sigma_0=160.0, MBH_scatter=0.53)

        if lgMBH is None:
            # this block assumes that the galaxy instance has already computed sigma_pc_yr and lgMgal, which is true if it was initialized with lgMgal. If it was initialized with lgMBH, then this block will not be executed and we will use the provided lgMBH directly.
            self._lgMBH = self.gal.lgMBH_mass
        else:
            self._lgMBH = np.asarray(lgMBH)


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
        r_inf_pc = G_pc3_per_Msun_yr2 * self.MBH_mass / (self.gal.sigma_pc_yr**2)
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


class CompactObject:
    def __init__(self, nsc, masses, total_mass, types_CO, types_masses, type_CO_limits=None):

        self.nsc = nsc
        self.types_CO = types_CO  # 'sBH' or 'star'
        self.total_number = {}
        self.types_masses = types_masses  # 'same_mass' or 'random_mass'
        self.masses = masses  # dict of masses for each CO type, e.g. {'sBH': 10.0, 'star': 1.0}
        self.total_mass = total_mass  # dict of total mass for each CO type, e.g. {'sBH': 20.0, 'star': 100.0}

        # other CO types can be added in the future,
        # but for now we only have sBHs and stars,
        # so we can initialize the total_number dict with these two keys.
        if self.types_masses == 'same_mass':
            for type_CO in self.types_CO:
                self.total_number[type_CO] = self.total_mass[type_CO] * self.nsc.MBH_mass / self.masses[type_CO]
        
        elif self.types_masses == 'random_mass':
            raise NotImplementedError("Random mass sampling for COs is not implemented yet. Please use 'same_mass' for now.")
            # the issue here is that it will need number of COs for 
            # counting which is what we are trying to compute in the 
            # first place, so we will need to do some iterative sampling 
            # from the CO mass function until we reach the total/close
            # to the mass we want, and then count the number of COs.
        else:
            raise ValueError("types_masses must be 'same_mass' or 'random_mass'")

    @property
    def total_number_CO(self):
        return self.total_number

    # @property
    # def component_masses(self):
    #     if self.types_masses == 'same_mass':
    #         breakpoint()
    #         y = np.zeros_like(self.total_number['sBH'])
    #         return {type_CO: np.full_like(y, self.masses[type_CO]) for type_CO in self.types_CO}
    #     else:
    #         raise NotImplementedError("Random mass sampling for COs is not implemented yet. Please use 'same_mass' for now.")
