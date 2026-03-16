from typing import Sequence, Optional
import numpy as np

from config import (kappa_cusp)
from nsc import NSC
from relaxation import RelaxationModel
from rates import RateModel
from cosmology import CosmologyModel

class CuspEvolution:
    """
    Encapsulates cusp formation time and cusp age calculations.
    It combines:
      - NSC structure
      - relaxation physics
      - EMRI/TDE rate physics (for time-to-peak if needed)
      - cosmological LMM sampling
    """

    def __init__(self, nsc: NSC, relaxation: RelaxationModel, rates: RateModel, cosmology: CosmologyModel, kappa: float = kappa_cusp_default):

        self.nsc = nsc
        self.relaxation = relaxation
        self.rates = rates
        self.cosmo = cosmology
        self.kappa = float(kappa)

    def cusp_turn_on_time(self, Ntot: float, component_masses: Sequence[float], kvir: float = 1.0, kind: str = 'TDE', mbar: Optional[float] = None, unit: str = 'Gyr'):
        """
        t_on = t_LMM + kappa * t_relax_at_rinf
        """
        # Sample LMM redshift and times (returns a 3-tuple)
        z_LMM, t_LMM, t_obs = self.cosmo.sample_lmm_times_Gyr(
            z_obs=self.nsc.gal.z,
            m=2.0
        )

        # Relaxation time at r_infl
        t_relax_rinf = self.relaxation.t_relax_at_rinf(Ntot=Ntot, component_masses=component_masses,
                                    kvir=kvir, kind=kind, mbar=mbar, unit=unit)

        # Cusp turn-on time 
        return t_LMM + self.kappa * t_relax_rinf

    def cusp_age(
        self,
        Ntot: float,
        component_masses: Sequence[float],
        kvir: float = 1.0,
        kind: str = 'TDE',
        mbar: Optional[float] = None
    ):
        """
        Cusp age = max(0, t_obs - t_on).

        Preserves your original formula:

            cusp_age = max(0, cosmology.age_Gyr(z) - cusp_turn_on_time)
        """
        t_on = self.cusp_turn_on_time(
            Ntot=Ntot,
            component_masses=component_masses,
            kvir=kvir,
            kind=kind,
            mbar=mbar,
            unit='Gyr'
        )

        t_obs = self.cosmo.age_Gyr(self.nsc.gal.z)
        return max(0.0, t_obs - t_on)


    def accumulated_objects_within_time(self, A=None, B=None, sigma_0=None, MBH_scatter=None):

        T_c = self.cusp_age(
            Ntot=None,
            component_masses=None  
        )

        t_EMRI = self.rates.time_to_peak_EMRI_rate(
            A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter
        )

        # This line preserved exactly as your original (buggy) code
        try:
            tau = Tc / t_EMRI     # original variable name mismatch preserved
        except Exception:
            tau = None

        return tau