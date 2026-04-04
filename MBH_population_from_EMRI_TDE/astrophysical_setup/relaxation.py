from typing import Sequence, Optional
import numpy as np

from config import (G_pc3_per_Msun_yr2, lnLambda)

from nsc import NSC, CompactObject
from density import DehnenProfile

class RelaxationModel:

    def __init__(self, nsc: NSC, compact_object: CompactObject, profile: DehnenProfile, lnLambda: float = lnLambda):
        self.nsc = nsc
        self.compact_object = compact_object
        self.profile = profile
        self.lnLambda = float(lnLambda)

    def m_bar(self):
        
        """See Spitzer 1987 for definition of m_bar at r_infl, which is the relevant mass scale for relaxation at r_infl."""

        r_infl = self.nsc.r_influence(kvir=1.0, unit='pc')
        
        if self.compact_object.types_masses == 'same_mass':
            numerator = self.compact_object.masses['star']**2 * self.profile.dehnen_at_radius(r_infl, kind='TDE') + self.compact_object.masses['sBH']**2 * self.profile.dehnen_at_radius(r_infl, kind='EMRI')

            denominator = self.compact_object.masses['star'] * self.profile.dehnen_at_radius(r_infl, kind='TDE') + self.compact_object.masses['sBH'] * self.profile.dehnen_at_radius(r_infl, kind='EMRI')

            mbar = numerator / denominator
        else:
            raise NotImplementedError("m_bar calculation currently only implemented for 'same_mass' CO mass distribution. Other distributions will require additional information about the mass function and number density of each component at r_infl.")
        
        return mbar
    
    def t_relax(self, rho_r, kvir=1.0, unit='Gyr'):
        """
        Two-body (non-resonant) relaxation time at r_infl.:
            t_rlx ≈ 0.34 * $\sigma$^3 / (G^2 * m_bar * $\rho(r_{infl})$ * lnΛ)
        """

        t_yr = 0.34 * (self.nsc.gal.sigma_pc_yr**3) / (G_pc3_per_Msun_yr2**2 * self.m_bar() * rho_r * lnLambda)

        if unit == 'yr':
            return t_yr
        elif unit == 'Gyr':
            return t_yr / 1.0e9
        else:
            raise ValueError("unit must be one of: 'yr', 'Gyr'")

    def t_relax_at_rinfl(self, kvir=1.0, unit='Gyr'):

        rho_at_rinfl = self.profile.mass_density_at_rinfl(kvir=kvir, unit='Msun/pc^3')
        return self.t_relax(rho_r=rho_at_rinfl, kvir=kvir, unit=unit)
