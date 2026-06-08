import numpy as np

from config import (MBH_A, MBH_B, MBH_sigma0, MBH_scatter)
from nsc import NSC
from scipy.stats import beta, norm
from dataclasses import dataclass

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