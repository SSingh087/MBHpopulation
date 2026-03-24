from typing import Sequence, Optional
import numpy as np

from config import (G_pc3_per_Msun_yr2, lnLambda)

from nsc import NSC
from density import DehnenProfile

class RelaxationModel:

    def __init__(self, nsc: NSC, profile: DehnenProfile, lnLambda: float = lnLambda):
        self.nsc = nsc
        self.profile = profile
        self.lnLambda = float(lnLambda)

    def t_relax(self, rho_r, Ntot, component_masses, kvir=1.0, kind='TDE', mbar=None, unit='Gyr'):
        """
        Two-body (non-resonant) relaxation time at r_infl.:
            t_rlx ≈ 0.34 * $\sigma$^3 / (G^2 * m_bar * $\rho(r_{infl})$ * lnΛ)

        Parameters
        ----------
        kvir       : float, virial coefficient in $\sigma$ definition
        mbar     : float [g], mean mass per scatterer; default = 1 Msun
        lnLambda   : float, Coulomb logarithm (10-15 typical)
        unit   :     'yr' or 'Gyr'

        Returns
        -------
        t_rlx : float, relaxation time in requested unit
        """

        t_yr = 0.34 * (self.nsc.gal.sigma(unit='pc/year')**3) / (G_pc3_per_Msun_yr2**2 * mbar * rho_r * lnLambda)

        if unit == 'yr':
            return t_yr
        elif unit == 'Gyr':
            return t_yr / 1.0e9
        else:
            raise ValueError("unit must be one of: 'yr', 'Gyr'")

    def t_relax_at_rinfl(self, Ntot, component_masses, kvir=1.0, kind='TDE', mbar=None, unit='Gyr'):

        rho_at_rinfl = self.profile.mass_density_at_rinfl(Ntot=Ntot, component_masses=component_masses, kvir=kvir, kind=kind, unit='Msun/pc^3')

        return self.t_relax(rho_r=rho_at_rinfl, Ntot=Ntot, component_masses=component_masses, kvir=kvir, kind=kind, mbar=mbar, unit=unit)
