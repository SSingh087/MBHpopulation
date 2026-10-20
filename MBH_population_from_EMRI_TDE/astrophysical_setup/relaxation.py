"""Two-body relaxation time model for the NSC."""
from typing import Sequence, Optional
import numpy as np

from config import (G_pc3_per_Msun_yr2, lnLambda)

from nsc import NSC, CompactObject
from density import DehnenProfile

class RelaxationModel:

    """Two-body relaxation time at the influence radius."""
    def __init__(self, nsc: NSC, compact_object: CompactObject, profile: DehnenProfile, lnLambda: float = lnLambda):
        self.nsc = nsc
        self.compact_object = compact_object
        self.profile = profile
        self.lnLambda = float(lnLambda)
        
        self.sBH_mass = self.compact_object.masses['sBH']
        self.star_mass = self.compact_object.masses['star']
        self.star_radius = self.compact_object.CO_radius['star']

    def m_bar_at_radius(self, radius_pc, kvir=1.0):
        
        """See Spitzer 1987 for definition of m_bar at r_infl, which is the relevant mass scale for relaxation at r_infl."""
        
        if self.compact_object.types_masses == 'same_mass':
            numerator = self.star_mass**2 * self.profile.dehnen_number_density_at_radius(radius_pc, kind='TDE') + self.sBH_mass**2 * self.profile.dehnen_number_density_at_radius(radius_pc, kind='EMRI')

            denominator = self.star_mass * self.profile.dehnen_number_density_at_radius(radius_pc, kind='TDE') + self.sBH_mass * self.profile.dehnen_number_density_at_radius(radius_pc, kind='EMRI')

            mbar = numerator / denominator
        else:
            raise NotImplementedError("m_bar calculation currently only implemented for 'same_mass' CO mass distribution. Other distributions will require additional information about the mass function and number density of each component at r_infl.")
        
        return mbar
    
    def t_relax_at_radius(self, rho_at_radius, radius_pc, kvir=1.0, unit='Gyr'):
        """
        Two-body (non-resonant) relaxation time at r_infl.:
            t_rlx ≈ 0.34 * $\sigma$^3 / (G^2 * m_bar * $\rho(r_{infl})$ * lnΛ)
        """

        numerator = 0.34 * (self.nsc.gal.sigma_pc_yr**3)
        denominator = (G_pc3_per_Msun_yr2**2) * self.m_bar_at_radius(radius_pc, kvir=kvir) * rho_at_radius * self.lnLambda

        t_yr = numerator / denominator

        if unit == 'yr':
            return t_yr
        elif unit == 'Gyr':
            return t_yr / 1.0e9
        else:
            raise ValueError("unit must be one of: 'yr', 'Gyr'")

    def t_relax_at_rinfl(self, kvir=1.0, unit='Gyr'):
        
        """Relaxation time at the MBH influence radius."""
        r_inf_pc = self.nsc.r_influence(kvir=kvir, unit='pc')
        rho_at_rinfl = self.profile.mass_density_at_rinfl(kvir=kvir, unit='Msun/pc^3')
        return self.t_relax_at_radius(rho_at_radius=rho_at_rinfl, radius_pc=r_inf_pc, kvir=kvir, unit=unit)