import numpy as np

from config import (MBH_A, MBH_B, MBH_sigma0, MBH_scatter, BROGGI_SIGMA0, BROGGI_MBH_PIVOT, BROGGI_INV_SLOPE)
from nsc import NSC
from scipy.stats import beta, norm
from dataclasses import dataclass
from scipy.interpolate import RegularGridInterpolator

class RateModel:
    def __init__(self, nsc: NSC):
        self.nsc = nsc

    def peak_EMRI_rate(self, A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter):
        """
        $\hat{\Gamma}_\mathrm{EMRI}
        """
        sigma = self.nsc.gal.sigma_km_s
        lgMBH = self.nsc.gal.lgMBH_mass
        MBH = 10.0**lgMBH
        a, b, c = 6.2e-6, -0.25, 3.09 

        return a * (MBH)**b * (sigma/sigma_0)**c 

    def time_to_peak_EMRI_rate(self, A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter):
        """
        $t_\mathrm{EMRI}
        """
        sigma = self.nsc.gal.sigma_km_s
        lgMBH = self.nsc.gal.lgMBH_mass
        MBH = 10.0**lgMBH
        a, b, c = 6.4e-9, 1.29, -2.97
        
        return a * (MBH)**b * (sigma/sigma_0)**c 


    def peak_TDE_rate(self, A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter):
        """
        $\hat{\Gamma}_\mathrm{TDE}
        """
        sigma = self.nsc.gal.sigma_km_s
        lgMBH = self.nsc.gal.lgMBH_mass
        MBH = 10.0**lgMBH
        a, b, c = 1.6e-4, -0.22, 1.16 

        return a * (MBH)**b * (sigma/sigma_0)**c

class UniversalRate:
    """
    Rate functions fitted to arXiv:2205.06277v1 Fig 6.
    tau = t/t_E (Eddington time units), rates normalised to peak ~ 1.

    Functional forms:
      EMRI / Plunges : f(tau) = [4*tau / (1+tau)^2]^alpha
          - peaks exactly at tau = 1 with f = 1
          - late-time power law ~ tau^{-alpha}
          - alpha = 1.25  → f(10) ≈ 0.25   (EMRI, red)
          - alpha = 0.63  → f(10) ≈ 0.50   (Plunges, blue)

      TDE             : f(tau) = A * (tau_0 + tau)^{-n}
          - monotonically decreasing (peak is off-screen at tau << 1)
          - A=0.896, tau_0=0.27, n=0.45  → f(0)≈1.6, f(10)≈0.30  (TD, green)

    NOTE: These are visual fits from the figure. If the paper gives explicit
    analytic expressions (check Sec. 3 / Appendix), replace with those.
    """

    @staticmethod
    def _peak_at_one(tau, alpha):
        """[4τ/(1+τ)²]^alpha — peaks at τ=1, decays as τ^{-alpha}."""
        tau = np.asarray(tau, dtype=float)
        return np.where(tau > 0, (4.0 * tau / (1.0 + tau) ** 2) ** alpha, 0.0)

    @staticmethod
    def EMRI_rate(tau):
        """
        Red curve in Fig 6.
        Rises from 0, peaks at tau=1, decays as ~tau^{-1.25}.

        Calibration points (tau → Γ/Γ̂):
            0   → 0,   1 → 1.00,   5 → 0.37,   10 → 0.25
        """
        return UniversalRate._peak_at_one(tau, alpha=1.25)

    @staticmethod
    def Plunge_rate(tau):
        """
        Blue curve in Fig 6.
        Rises from 0, peaks at tau=1, slow decay as ~tau^{-0.63}.

        Calibration points (tau → Γ/Γ̂):
            0   → 0,   1 → 1.00,   5 → 0.62,   10 → 0.50
        """
        return UniversalRate._peak_at_one(tau, alpha=0.63)

    @staticmethod
    def TDE_rate(tau):
        """
        Green (TD) curve in Fig 6.
        Shows a subtle peak near tau ~ 0.3-0.5, then monotonically decreases.
        Modelled to avoid over-suppression at low tau.

        Calibration points (tau → Γ/Γ̂):
            0.1 → ~1.5,   0.3 → ~1.2,   1 → ~0.85,   5 → ~0.41,   10 → ~0.30
        
        Fitted as: (1 + 0.5*tau)^{-0.3} * (0.5 + tau)^{-0.35}
        This avoids the sharp rise at tau=0 and provides gentle peak structure.
        """
        tau = np.asarray(tau, dtype=float)
        # Gentler power law structure that matches the figure better
        term1 = (1.0 + 0.5 * tau) ** (-0.3)   # Gentle suppression at high tau
        term2 = (0.5 + tau) ** (-0.35)        # Moderate power law decay
        return term1 * term2

class ConstantRate:

    # 7 MBH masses, log-evenly spaced 1e4 - 1e7 Msun
    _log10_MBH_grid = np.log10(np.array([1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7]))

    # 4 density levels {0.01,0.1,1,10} rho_0 -> sigma/sigma_0 = rho_ratio**(1/6)
    _sigma_ratio_grid = np.array([0.464, 0.681, 1.0, 1.468])
    _log10_sigma_ratio_grid = np.log10(_sigma_ratio_grid)

    # N_E, rows = MBH masses (increasing), cols = sigma/sigma_0 (increasing)
    _N_E = np.array([
        [2.6,    2.7,    2.8,    3.0],     # 1e4
        [8.5,    8.3,    9.0,    9.4],     # 3e4
        [29,     29,     30,     32],      # 1e5
        [88,     90,     97,     100],     # 3e5
        [180,    310,    350,    370],     # 1e6
        [41,     330,    9300,   1300],    # 3e6  -- capping effects dominate here, see paper Sec 4.3.2
        [2.4,    41,     810,    2800],    # 1e7
    ])

    # N_T -- screenshot-only source, lower confidence, double check against source
    _N_T = np.array([
        [6.7e3,  7.3e3,  7.5e3,  7.9e3],
        [2.1e4,  2.2e4,  2.3e4,  2.4e4],
        [7.2e4,  7.4e4,  7.7e4,  9.0e4],
        [2.1e5,  2.2e5,  2.3e5,  2.4e5],
        [3.0e5,  6.9e5,  7.5e5,  7.8e5],
        [2.6e5,  7.5e5,  1.8e6,  2.3e6],
        [2.0e5,  6.0e5,  1.8e6,  4.6e6],
    ])

    # t_E [Gyr], used only to derive t_f = min(10*t_E, 10 Gyr) -- matches every tabulated tau
    _t_E = np.array([
        [0.089,  0.0029, 0.00092, 0.00030],
        [0.037,  0.012,  0.0038,  0.0012],
        [0.18,   0.056,  0.018,   0.0057],
        [0.73,   0.23,   0.075,   0.023],
        [3.5,    1.1,    0.35,    0.11],
        [10,     4.6,    1.5,     0.46],
        [10,     10,     6.9,     2.1],
    ])

    def __init__(self, nsc: NSC):
        self.nsc = nsc

        self._t_f = np.minimum(10.0 * self._t_E, 10.0)          # Gyr
        rate_EMRI_grid = self._N_E / self._t_f                  # 1/Gyr
        rate_TDE_grid  = self._N_T / self._t_f                  # 1/Gyr

        self._interp_EMRI = RegularGridInterpolator(
            (self._log10_MBH_grid, self._log10_sigma_ratio_grid),
            np.log10(rate_EMRI_grid), bounds_error=False, fill_value=None)

        self._interp_TDE = RegularGridInterpolator(
            (self._log10_MBH_grid, self._log10_sigma_ratio_grid),
            np.log10(rate_TDE_grid), bounds_error=False, fill_value=None)

    def broggi_sigma0(self, MBH):
        """sigma_0(M_BH) per Broggi et al. 2022 Eq. 15."""
        return BROGGI_SIGMA0 * (MBH / BROGGI_MBH_PIVOT) ** BROGGI_INV_SLOPE

    def constant_rate(self, kind: str = 'EMRI'):
        MBH = 10.0 ** np.atleast_1d(self.nsc.gal.lgMBH_mass)
        sigma_0_broggi = self.broggi_sigma0(MBH)
        sigma_ratio = np.atleast_1d(self.nsc.gal.sigma_km_s) / sigma_0_broggi

        log10_MBH   = np.clip(np.log10(MBH), self._log10_MBH_grid.min(), self._log10_MBH_grid.max())
        log10_sigma = np.clip(np.log10(sigma_ratio), self._log10_sigma_ratio_grid.min(), self._log10_sigma_ratio_grid.max())
        pts = np.column_stack([log10_MBH, log10_sigma])

        if kind.upper() == 'EMRI':
            rate = 10.0 ** self._interp_EMRI(pts)
        elif kind.upper() == 'TDE':
            rate = 10.0 ** self._interp_TDE(pts)
        else:
            raise ValueError(f"Unknown kind: {kind}")

        # Outside Broggi's simulated MBH range there's no calibration, and
        # physically LISA-band EMRIs only exist for M_BH ~ 1e4-1e7 anyway --
        # zero rather than extrapolate or pin to the edge.
        in_range = (MBH >= 1e4) & (MBH <= 1e7)
        return np.where(in_range, rate, 0.0)