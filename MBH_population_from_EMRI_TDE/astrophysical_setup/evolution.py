from typing import Sequence, Optional
import numpy as np

from config import (kappa_cusp)
from nsc import NSC, CompactObject
from relaxation import RelaxationModel
from rate import RateModel, UniversalRate
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

    def __init__(self, nsc: NSC, compact_object: CompactObject, relaxation: RelaxationModel, rate_model: RateModel, LastMajorMerger: LastMajorMerger, kappa: float = kappa_cusp):

        self.nsc = nsc
        self.compact_object = compact_object
        self.relaxation = relaxation
        self.rate_model = rate_model
        self.LastMajorMerger = LastMajorMerger
        self.cosmo = self.LastMajorMerger.cosmo_model
        self.kappa = float(kappa)

    def cusp_turn_on_time(self, kvir: float = 1.0, unit: str = 'Gyr'):
        """
        t_on = t_LMM + kappa * t_relax_at_rinf
        """

        # Sample LMM redshift and times (returns a 3-tuple)
        z_LMM, t_LMM, t_obs = self.LastMajorMerger.sample_LMM_times(z_obs_array=self.nsc.gal.z_gal, size=1)

        # Relaxation time at r_infl
        t_relax_rinf = self.relaxation.t_relax_at_rinfl(kvir=kvir, unit=unit)

        # Cusp turn-on time 
        return t_LMM + self.kappa * t_relax_rinf

    def cusp_age(self, kvir: float = 1.0, unit: str = 'Gyr'):
        """
        cusp_age = max(0, cosmology.age_Gyr(z) - cusp_turn_on_time)
        """
        t_on = self.cusp_turn_on_time(kvir=kvir, unit=unit)
        t_obs = self.cosmo.age_Gyr(self.nsc.gal.z_gal)
        return np.maximum(0.0, t_obs - t_on)

    def t_EMRI(self, A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter):
        return self.rate_model.time_to_peak_EMRI_rate(A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter)
    
    def Gamma_hat(self, A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter, kind: str = 'EMRI'):
        if kind.upper() == 'EMRI':
            return self.rate_model.peak_EMRI_rate(A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter)
        elif kind.upper() == 'TDE':
            return self.rate_model.peak_TDE_rate(A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter)
        else:
            raise ValueError(f"Unknown kind: {kind}")

    def evaluate_tau(self, kvir, unit: str = 'Gyr', A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter):
        
        T_c_array = self.cusp_age(kvir=kvir, unit=unit)

        t_EMRI_array = self.t_EMRI(A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter)

        return T_c_array / t_EMRI_array
    
    def accumulated_objects_within_t_rlx_to_cusp_age(self, kvir: float = 1.0, unit: str = 'Gyr', A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter, n_grid: int = 4096, kind: str = 'EMRI'):
        """
        Compute the expected number of objects (TDEs or EMRIs) accumulated within time T_c for each galaxy, by integrating the universal rate over the normalized time grid and scaling by the peak rate and time to peak.
        """
        tau = self.evaluate_tau(kvir=kvir, unit=unit, A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter)
        
        t_EMRI = self.t_EMRI(A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter)
        
        Gamma_hat = self.Gamma_hat(A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter, kind=kind)

        tau = np.atleast_1d(tau)
        N = tau.size

        tau_norm_grid = np.linspace(1e-6, 1.0, n_grid)  # shape (n_grid,)
        tau_grid = tau[:, None] * tau_norm_grid[None, :]  # shape (N, n_grid)

        kind_upper = kind.upper()
        if kind_upper == 'EMRI':
            _rate_grid = UniversalRate.EMRI_rate(tau_grid)  # shape (N, n_grid)
        elif kind_upper == 'TDE':
            _rate_grid = UniversalRate.TDE_rate(tau_grid)   # shape (N, n_grid)
        else:
            raise ValueError(f"Unknown kind: {kind}")

        cumulative_distribution = np.trapezoid(_rate_grid, tau_grid, axis=1)  # shape (N,)
        N_objects = Gamma_hat * t_EMRI * cumulative_distribution  # scalar * scalar * (N,) = (N,)

        return N_objects

    def expected_objects_in_time(self, T_obs, kvir: float = 1.0, unit: str = 'Gyr', A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter, n_grid: int = 4096, kind: str = 'EMRI'):
        
        tau = self.evaluate_tau(kvir=kvir, unit=unit, A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter)
        Gamma_hat = self.Gamma_hat(A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter, kind=kind)

        if kind.upper() == 'EMRI':
            return Gamma_hat * UniversalRate.EMRI_rate(tau) * (T_obs / (1 + self.nsc.gal.z_gal))
        elif kind.upper() == 'TDE':
            return Gamma_hat * UniversalRate.TDE_rate(tau) * (T_obs / (1 + self.nsc.gal.z_gal))

    def number_of_objects_in_time(self, T_obs, kvir: float = 1.0, unit: str = 'Gyr', A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter, n_grid: int = 4096, kind: str = 'EMRI'):
        N_expected = self.expected_objects_in_time(T_obs=T_obs, kvir=kvir, unit=unit, A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter, n_grid=n_grid, kind=kind) * [1E7 if kind.upper() == 'EMRI' else 1E5]
        # scale up to get more realistic numbers for Poisson sampling
        return np.random.poisson(N_expected)