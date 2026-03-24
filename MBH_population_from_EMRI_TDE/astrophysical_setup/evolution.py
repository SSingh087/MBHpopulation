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

    def cusp_turn_on_time(self, Ntot, component_masses: Sequence[float], mbar, kvir: float = 1.0, kind: str = 'TDE', unit: str = 'Gyr'):
        """
        t_on = t_LMM + kappa * t_relax_at_rinf
        """

        # Sample LMM redshift and times (returns a 3-tuple)
        z_LMM, t_LMM, t_obs = self.LastMajorMerger.sample_LMM_times(z_obs_array=self.nsc.gal.z_gal, size=1)

        # Relaxation time at r_infl
        t_relax_rinf = self.relaxation.t_relax_at_rinfl(Ntot=Ntot, component_masses=component_masses, kvir=kvir, kind=kind, mbar=mbar, unit=unit)

        # Cusp turn-on time 
        return t_LMM + self.kappa * t_relax_rinf

    def cusp_age(self, Ntot, component_masses: Sequence[float], mbar, kvir: float = 1.0, kind: str = 'TDE', unit: str = 'Gyr'):
        """
        cusp_age = max(0, cosmology.age_Gyr(z) - cusp_turn_on_time)
        """
        t_on = self.cusp_turn_on_time(Ntot=Ntot, component_masses=component_masses, mbar=mbar, kvir=kvir, kind=kind, unit='Gyr')
        t_obs = self.cosmo.age_Gyr(self.nsc.gal.z_gal)
        return np.maximum(0.0, t_obs - t_on)

    def evaluate_tau(self, Ntot, component_masses, kvir, kind, mbar, unit: str = 'Gyr', A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter):
        
        T_c_array = self.cusp_age(Ntot=Ntot, component_masses=component_masses, kvir=kvir, kind=kind, mbar=mbar, unit=unit)

        t_EMRI_array = self.t_EMRI(A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter)
        
        Gamma_hat_EMRI_array = self.Gamma_hat_EMRI(A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter)

        return T_c_array / t_EMRI_array, T_c_array, t_EMRI_array, Gamma_hat_EMRI_array
    
    def t_EMRI(self, A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter):
        return self.rate_model.time_to_peak_EMRI_rate(A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter)
    
    def Gamma_hat_EMRI(self, A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter):
        return self.rate_model.peak_EMRI_rate(A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter)

    def accumulated_objects_within_time(self, Ntot, component_masses, mbar,
        kvir: float = 1.0, kind: str = 'TDE', unit: str = 'Gyr',
        A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter, n_grid: int = 4096):
        """
        Compute the expected number of objects (TDEs or EMRIs) accumulated within time T_c for each galaxy, by integrating the universal rate over the normalized time grid and scaling by the peak rate and time to peak.
        """

        tau, T_c, t_EMRI, Gamma_hat_EMRI = self.evaluate_tau(
            Ntot=Ntot, component_masses=component_masses, kvir=kvir,
            kind=kind, mbar=mbar, unit=unit, A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter
        )

        tau = np.atleast_1d(tau)
        N = tau.size

        tau_norm_grid = np.linspace(1e-6, 1.0, n_grid)  # shape (n_grid,)
        tau_grid = tau[:, None] * tau_norm_grid[None, :]  # shape (N, n_grid)

        kind_upper = kind.upper()
        if kind_upper == 'EMRI':
            _rate_grid = self.rate_model.universal_EMRI_rate(tau_grid)  # shape (N, n_grid)
        elif kind_upper == 'TDE':
            _rate_grid = self.rate_model.universal_TDE_rate(tau_grid)   # shape (N, n_grid)
        else:
            raise ValueError(f"Unknown kind: {kind}")

        cumulative_distribution = np.trapezoid(_rate_grid, tau_grid, axis=1)  # shape (N,)
        N_objects = Gamma_hat_EMRI * t_EMRI * cumulative_distribution  # scalar * scalar * (N,) = (N,)

        return N_objects
