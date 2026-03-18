from typing import Sequence, Optional
import numpy as np

from config import (kappa_cusp)
from nsc import NSC
from relaxation import RelaxationModel
from rate import RateModel
from cosmology import LastMajorMerger, CosmologyModel

from config import (MBH_A, MBH_B, MBH_sigma0, MBH_scatter)

class CuspEvolution:
    """
    Encapsulates cusp formation time and cusp age calculations.
    It combines:
      - NSC structure
      - relaxation physics
      - EMRI/TDE rate physics (for time-to-peak if needed)
      - cosmological LMM sampling
    """

    def __init__(self, nsc: NSC, relaxation: RelaxationModel, rate_model: RateModel, LastMajorMerger: LastMajorMerger, kappa: float = kappa_cusp):

        self.nsc = nsc
        self.relaxation = relaxation
        self.rate_model = rate_model
        self.LastMajorMerger = LastMajorMerger
        self.cosmo = self.LastMajorMerger.cosmo_model
        self.kappa = float(kappa)

    def cusp_turn_on_time(self, Ntot: float, component_masses: Sequence[float], kvir: float = 1.0, kind: str = 'TDE', mbar: Optional[float] = None, unit: str = 'Gyr'):
        """
        t_on = t_LMM + kappa * t_relax_at_rinf
        """
        # Sample LMM redshift and times (returns a 3-tuple)
        z_LMM, t_LMM, t_obs = self.LastMajorMerger.sample_LMM_times(z_obs=self.nsc.gal.z_gal, size=1)

        # Relaxation time at r_infl
        t_relax_rinf = self.relaxation.t_relax_at_rinfl(Ntot=Ntot, component_masses=component_masses, kvir=kvir, kind=kind, mbar=mbar, unit=unit)

        # Cusp turn-on time 
        return t_LMM + self.kappa * t_relax_rinf

    def cusp_age(self, Ntot: float, component_masses: Sequence[float], kvir: float = 1.0, kind: str = 'TDE', mbar: Optional[float] = None, unit: str = 'Gyr'):
        """
        cusp_age = max(0, cosmology.age_Gyr(z) - cusp_turn_on_time)
        """
        t_on = self.cusp_turn_on_time(Ntot=Ntot, component_masses=component_masses, kvir=kvir, kind=kind, mbar=mbar, unit='Gyr')

        t_obs = self.cosmo.age_Gyr(self.nsc.gal.z_gal)
        return max(0.0, t_obs - t_on)

    def evaluate_tau(self, Tc, t_EMRI):
        return Tc / t_EMRI

    def accumulated_objects_within_time(self, Ntot: float, component_masses: Sequence[float], kvir: float = 1.0, kind: str = 'TDE', mbar: Optional[float] = None, unit: str = 'Gyr', A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter):

        T_c = self.cusp_age(Ntot=Ntot, component_masses=component_masses, kvir=kvir, kind=kind, mbar=mbar, unit=unit)
        t_EMRI = self.rate_model.time_to_peak_EMRI_rate(A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter)
        Gamma_hat_EMRI = self.rate_model.peak_EMRI_rate(A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter)


        tau_final = self.evaluate_tau(T_c, t_EMRI)

        tau_grid = np.linspace(1E-6, tau_final, 4096)
        
        if kind == 'EMRI':
            _rate_ = self.rate_model.universal_EMRI_rate(tau_grid)
        
        if kind == 'TDE':
            _rate_ = self.rate_model.universal_TDE_rate(tau_grid)
        
        cummulative_distribution = np.trapezoid(_rate_, tau_grid)
        print(cummulative_distribution)
        N_EMRIs = Gamma_hat_EMRI * t_EMRI * cummulative_distribution

        return N_EMRIs
