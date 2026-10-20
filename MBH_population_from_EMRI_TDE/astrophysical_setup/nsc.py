"""Nuclear star cluster, compact-object populations and MBH properties of a galaxy sample."""
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
            self._lgMBH_mass_ = self.gal.lgMBH_mass
        else:
            self._lgMBH_mass_ = np.asarray(lgMBH)

    @property
    def MBH_mass(self):
        return 10 ** self._lgMBH_mass_

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

    def r_tidal(self, m_star_solar, R_star_solar, unit='pc'):
        """
        Stellar tidal disruption radius:
        r_t = R_star * (M_bh / m_star)^(1/3)
        """
        R_star_pc = R_star_solar * Rsun / pc_to_cm  # Convert stellar radius from solar radii to parsecs, Rsun in cm, then to pc
        rt_pc = R_star_pc * (self.MBH_mass / m_star_solar)**(1.0/3.0)
        if unit == 'pc':
            return rt_pc
        elif unit == 'cm':
            return rt_pc * pc_to_cm
        else:
            raise ValueError("unit must be 'pc' or 'cm'")


class CompactObject:
    def __init__(self, nsc, masses, total_mass_in_MBH_mass_units, types_CO, types_masses, type_CO_limits=None):

        self.nsc = nsc
        self.types_CO = types_CO  # 'sBH' or 'star'
        self.total_number = {}
        self.types_masses = types_masses  # 'same_mass' or 'random_mass'
        self.masses = masses  # dict of masses for each CO type, e.g. {'sBH': 10.0, 'star': 1.0}
        self.total_mass_in_MBH_mass_units = total_mass_in_MBH_mass_units  # dict of total mass for each CO type, e.g. {'sBH': 0.2, 'star': 20.0}


        # other CO types can be added in the future,
        # but for now we only have sBHs and stars,
        # so we can initialize the total_number dict with these two keys.
        if self.types_masses == 'same_mass':
            for type_CO in self.types_CO:
                self.total_number[type_CO] = self.total_mass_in_MBH_mass_units[type_CO] * self.nsc.MBH_mass / self.masses[type_CO]

        elif self.types_masses == 'random_mass':
            raise NotImplementedError("Random mass sampling for COs is not implemented yet. Please use 'same_mass' for now.")

        self.CO_radius = {}
        self.CO_radius['sBH'] = 0
        self.CO_radius['star'] = 1  # this is in units of solar radii, and will be converted to pc in the density profile when needed.

        # self.orbit_sense = self.orbit_sense() 

    @property
    def total_number_CO(self):
        return self.total_number

    def orbit_sense(self):
        # ----------------------------------------------------------------
        # this is required for generating the MBHspin evolution.
        # this can be added in the nsc.py but it will take loads of memory to 
        # store the spin evolution for all the galaxies and all the events.
        # Hence we generate it on the fly for each EMRI event only when we need it, and we can also use it for TDEs if needed.
        # ----------------------------------------------------------------
        # this is only possible when the environemt is dry for prograde and retrograde.
        # for wet environments, p>0.5 since most of the COs will align with the disc.
        # we can also have a mixed case we will need to work on this.
        return np.random.binomial(n=1, p=0.5, size=self.total_number)

    # @property
    # def component_masses(self):
    #     if self.types_masses == 'same_mass':
    #         y = np.zeros_like(self.total_number['sBH'])
    #         return {type_CO: np.full_like(y, self.masses[type_CO]) for type_CO in self.types_CO}
    #     else:
    #         raise NotImplementedError("Random mass sampling for COs is not implemented yet. Please use 'same_mass' for now.")

class MBH_properties:
    def __init__(self, nsc, A=7.87, B=4.55, sigma_0=160.0, MBH_scatter=0.53):
        self.nsc = nsc
        self.lgMBH_mass = self.nsc._lgMBH_mass_
        self.initial_MBHspin= self.initial_MBHspin()
   
    def initial_MBHspin(self):

        alpha0_low_mass = 2.5
        lambda_alpha_low_mass = 0.1
        beta_low_mass = 5.0

        alpha0_high_mass = 7.0
        lambda_alpha_high_mass = 0.1
        beta_high_mass = 5.0

        spins = np.zeros(len(self.lgMBH_mass))
        spins[self.lgMBH_mass < 6] = np.random.beta(alpha0_low_mass + lambda_alpha_low_mass * (self.lgMBH_mass[self.lgMBH_mass < 6] - 6), beta_low_mass)
        spins[self.lgMBH_mass >= 6] = np.random.beta(alpha0_high_mass + lambda_alpha_high_mass * (self.lgMBH_mass[self.lgMBH_mass >= 6] - 6), beta_high_mass)

        return spins
    
    @staticmethod
    def MBHspin_evolution_at_time(initial_spin):
        # this needs a bit of discussion 
        # since a) the mass of e MBH can change 
        # and b) the spin evolution depends on the environment which can change with time, and also on the orbit sense which can change with time as well.
        # this should then feedback into the waveform which we dont have 
        # also I think for the span of observations we can assume that the mass of the MBH is constant, and also the environment is constant, and the orbit sense is constant as well, since we are only looking at a short time span compared to the evolution timescales of these properties.
        return initial_spin
