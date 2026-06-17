import numpy as np

from config import (kappa_cusp)
from nsc import NSC, CompactObject
from relaxation import RelaxationModel
from rate import RateModel, UniversalRate
from cosmology import LastMajorMerger

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
        # self.t_obs = self.cosmo_model.age_Gyr(z_obs_array=self.nsc.gal.z_gal)
        

    def cusp_turn_on_time(self, kvir: float = 1.0):
        """
        t_on = t_LMM + kappa * t_relax_at_rinf
            - t_LMM is the lookback time to the last major merger, sampled from cosmology (in Gyr)
            - t_relax_at_rinf is the relaxation time at the influence radius (in Gyr)
            - kappa is a dimensionless factor that sets how many relaxation times after the LMM the cusp is considered to be "turned on".
        
        Returns:
            t_on in Gyr
        """
        self.z_LMM, self.t_LMM, self.t_obs_at_z_obs = self.LastMajorMerger.sample_LMM_times(z_obs_array=self.nsc.gal.z_gal, size=1)

        t_relax_rinf = self.relaxation.t_relax_at_rinfl(kvir=kvir, unit='Gyr')

        return self.t_LMM + self.kappa * t_relax_rinf

    def cusp_age(self, kvir: float = 1.0):
        """
        cusp_age = max(0, t_obs - cusp_turn_on_time)
        
        Returns:
            cusp_age in Gyr
        """
        t_on = self.cusp_turn_on_time(kvir=kvir)
        
        return np.maximum(0.0, self.t_obs_at_z_obs - t_on)

    def t_EMRI(self, A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter):
        return self.rate_model.time_to_peak_EMRI_rate(A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter)
    
    def Gamma_hat(self, A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter, kind: str = 'EMRI'):
        if kind.upper() == 'EMRI':
            return self.rate_model.peak_EMRI_rate(A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter)
        elif kind.upper() == 'TDE':
            return self.rate_model.peak_TDE_rate(A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter)
        else:
            raise ValueError(f"Unknown kind: {kind}")

    def evaluate_tau(self, kvir, A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter):
        
        T_c_array = self.cusp_age(kvir=kvir)

        t_EMRI_array = self.t_EMRI(A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter)

        return T_c_array / t_EMRI_array
    
    # def expected_objects_in_time(self, T_obs, kvir: float = 1.0, A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter, n_grid: int = 4096, kind: str = 'EMRI'):
    #     """
    #     Integrates the rate over the observation period from t=0 to t=T_obs.
        
    #     N_expected = (1 / (1+z)) * integral from 0 to T_obs of:
    #         Gamma_hat * R(tau(t)) * dt
        
    #     where:
    #     - tau(t) = T_c(t) / t_EMRI evolves as the cusp ages
    #     - Gamma_hat is in units of [year^-1]
    #     - T_obs is in [Gyr] = 10^9 [years]
    #     - dt is in [Gyr] units, automatically converted to [years] for integration
        
    #     Uses numerical integration (trapezoidal rule) over a grid.
    #     The factor of (1+z) accounts for time dilation in an expanding universe.
    #     """
    #     # Create time grid from 0 to T_obs (in Gyr)
        
    #     t_grid = np.linspace(0, T_obs, n_grid)
        
    #     # For each time step, compute the cusp age at that time
    #     # We need to recalculate cusp_turn_on_time once (same LMM for all times)
    #     t_on = self.cusp_turn_on_time(kvir=kvir)
        
    #     # Cusp age at each time in the grid
    #     T_c_grid = np.maximum(0.0, t_grid[None, :] - t_on[:, None])  # Shape: (N_gal, n_grid)
        
    #     # Get t_EMRI (constant for each galaxy)
    #     t_EMRI_array = self.t_EMRI(A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter)  # Shape: (N_gal,)
        
    #     # Compute tau(t) at each grid point
    #     tau_grid = T_c_grid / t_EMRI_array[:, None]  # Shape: (N_gal, n_grid)
        
    #     # Get Gamma_hat (in units of [year^-1])
    #     Gamma_hat = self.Gamma_hat(A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter, kind=kind)  # Shape: (N_gal,)
        
    #     # Compute rate at each grid point (normalized, dimensionless)
    #     if kind.upper() == 'EMRI':
    #         rate_grid = UniversalRate.EMRI_rate(tau_grid)
    #     elif kind.upper() == 'TDE':
    #         rate_grid = UniversalRate.TDE_rate(tau_grid)
    #     else:
    #         raise ValueError(f"Unknown kind: {kind}")
        
    #     # Integrate using trapezoidal rule
    #     # integrand = Gamma_hat [yr^-1] * rate_grid [dimensionless]
    #     # dt = [Gyr] = 10^9 [years]
    #     # integral = [yr^-1] * [yr] = [dimensionless count]
    #     integrand = Gamma_hat[:, None] * rate_grid  # Shape: (N_gal, n_grid), units: [yr^-1]
        
    #     # Trapezoidal rule integration with dt in Gyr = 10^9 years
    #     dt_gyr = T_obs / (n_grid - 1)
    #     dt_yr = dt_gyr * 1e9  # Convert Gyr to years
    #     integral = np.trapezoid(integrand, dx=dt_yr, axis=1)  # Shape: (N_gal,), units: [dimensionless count]
        
    #     # Apply redshift factor (observation time is dilated by (1+z))
    #     return integral / (1 + self.nsc.gal.z_gal)

    # def number_of_objects_in_time(self, T_obs, kvir: float = 1.0, A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter, n_grid: int = 4096, kind: str = 'EMRI'):

    #     N_expected = self.expected_objects_in_time(T_obs=T_obs, kvir=kvir, A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter, n_grid=n_grid, kind=kind)
        

    #     return np.random.poisson(N_expected)


    def expected_objects_in_time_full_integration(self, T_obs, kvir: float = 1.0, A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter, n_grid: int = 4096, kind: str = 'EMRI'):
        """
        Calculate the expected number of objects in the observation time and do a full integration over the rate evolution, instead of assuming a constant rate.
         - T_obs is the observation time in Gyr (e.g., 4 Gyr for LISA)
         - n_grid is the number of grid points to use in the integration over tau
         - kind is either 'EMRI' or 'TDE' to specify which rate evolution to use
         - returns N_expected, the expected number of objects in the observation time (dimensionless)
         - Note: the final N_expected is dimensionless because the rate evolution is normalised to the peak rate, and we multiply by the time to peak (t_EMRI) to get a dimensionless number of events. We also include cosmological time dilation by dividing by (1 + z_gal).
         - The integration is done over tau = T_c / t_EMRI, where T_c is the cusp age at the time of observation. The rate evolution f(tau) is given by UniversalRate.EMRI_rate(tau) or UniversalRate.TDE_rate(tau) depending on the kind of event we are calculating for.
         - The expected number of events is then given by N_expected = t_EMRI * integral_over_tau, where integral_over_tau is the integral of Gamma_hat * f(tau) over the range of tau corresponding to the observation time.
         - Finally, we apply cosmological time dilation by dividing N_expected by (1 + z_gal).
        """
        t_on_Gyr      = np.atleast_1d(self.cusp_turn_on_time(kvir=kvir))   # (N_gal,) Gyr
        t_obs_Gyr     = np.atleast_1d(self.t_obs_at_z_obs)                  # (N_gal,) Gyr
        T_c_obs_Gyr   = np.maximum(0.0, t_obs_Gyr - t_on_Gyr)              # (N_gal,) Gyr
        T_c_obs_yr    = T_c_obs_Gyr * 1e9                                   # (N_gal,) yr

        N_gal = len(T_c_obs_yr)

        t_EMRI_Gyr = self.t_EMRI(A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter)
        t_EMRI_yr = t_EMRI_Gyr * 1e9                                        # (N_gal,) yr
        tau_obs   = T_c_obs_yr / t_EMRI_yr                                  # (N_gal,)
        tau_upper = tau_obs + T_obs / t_EMRI_yr                             # (N_gal,)
        s = np.linspace(0.0, 1.0, n_grid)                           # (n_grid,)
        tau_grid = (tau_obs[:, None] + s[None, :] * (tau_upper - tau_obs)[:, None])# (N_gal, n_grid)

        if kind.upper() == 'EMRI':
            rate_grid = UniversalRate.EMRI_rate(tau_grid)
        elif kind.upper() == 'TDE':
            rate_grid = UniversalRate.TDE_rate(tau_grid)
        else:
            raise ValueError(f"Unknown kind: {kind}")


        Gamma_hat = self.Gamma_hat(A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter, kind=kind)
        integrand = Gamma_hat[:, None] * rate_grid
        integral_over_tau = np.trapezoid(integrand, x=tau_grid, axis=1)    # (N_gal,) yr^-1
        integral = t_EMRI_yr * integral_over_tau                            # (N_gal,) dimensionless

        N_expected = integral / (1.0 + self.nsc.gal.z_gal)                 # (N_gal,)

        return N_expected

    def expected_objects_in_time(self, T_obs, kvir: float = 1.0, A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter, kind: str = 'EMRI'):
        """
        Calculate the expected number of objects in the observation time. See expected_objects_in_time_full_integration for details. Now here we assume that the rate is constant at the peak value (Gamma_hat) during the observation time, which is a good approximation if T_obs << t_EMRI, but can be inaccurate if T_obs ~ t_EMRI or larger. The full integration method should be used for more accurate results in those cases.
         - T_obs is the observation time in Gyr (e.g., 4 Gyr for LISA)
         - kind is either 'EMRI' or 'TDE' to specify which rate evolution to use
         - returns N_expected, the expected number of objects in the observation    
        """
        t_on_Gyr      = np.atleast_1d(self.cusp_turn_on_time(kvir=kvir))   # (N_gal,) Gyr
        t_obs_Gyr     = np.atleast_1d(self.t_obs_at_z_obs)                  # (N_gal,) Gyr

        T_c_obs_Gyr   = np.maximum(0.0, t_obs_Gyr - t_on_Gyr)              # (N_gal,) Gyr


        t_EMRI_Gyr = self.t_EMRI(A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter)
        Gamma_hat = self.Gamma_hat(A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter, kind=kind)

        tau_obs = T_c_obs_Gyr / t_EMRI_Gyr  # (N_gal,)

        if kind.upper() == 'EMRI':
            rate_grid = UniversalRate.EMRI_rate(tau_obs)
        elif kind.upper() == 'TDE':
            rate_grid = UniversalRate.TDE_rate(tau_obs)
        else:
            raise ValueError(f"Unknown kind: {kind}")
        
        # print(f"Rate grid (first 10): {rate_grid[:10]}")
        # print(f"Tau obs (first 10): {tau_obs[:10]}")
        # print(f"Gamma_hat (first 10): {self.Gamma_hat(A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter, kind=kind)[:10]}")

        N_expected = np.sum(rate_grid * T_obs * Gamma_hat / (1.0 + self.nsc.gal.z_gal))

        print(f"Total expected events {kind.upper()}: {N_expected}")

        return rate_grid * T_obs * Gamma_hat / (1.0 + self.nsc.gal.z_gal)                 # (N_gal,)

    def number_of_objects_in_time(self, T_obs, kvir: float = 1.0, A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter, n_grid: int = 4096, kind: str = 'EMRI'):

        # N_expected = self.expected_objects_in_time_full_integration(T_obs=T_obs, kvir=kvir, A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter, n_grid=n_grid, kind=kind)

        N_expected = self.expected_objects_in_time(T_obs=T_obs, kvir=kvir, A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter, kind=kind) * 1E8 if kind.upper() == 'EMRI' else self.expected_objects_in_time(T_obs=T_obs, kvir=kvir, A=A, B=B, sigma_0=sigma_0, MBH_scatter=MBH_scatter, kind=kind) * 1E8

        return np.random.poisson(N_expected)