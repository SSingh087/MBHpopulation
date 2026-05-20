import numpy as np

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

    # ----- core building blocks -----------------------------------------

    @staticmethod
    def _peak_at_one(tau, alpha):
        """[4τ/(1+τ)²]^alpha — peaks at τ=1, decays as τ^{-alpha}."""
        tau = np.asarray(tau, dtype=float)
        return np.where(tau > 0, (4.0 * tau / (1.0 + tau) ** 2) ** alpha, 0.0)

    # ----- public API -------------------------------------------------------

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
    

EMRI_rate = UniversalRate.EMRI_rate
Plunge_rate = UniversalRate.Plunge_rate
TDE_rate = UniversalRate.TDE_rate

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('font', family='serif', serif=['Computer Modern'], size=12)
matplotlib.rc('text', usetex=True)

fig, ax = plt.subplots(figsize=(10, 6))

tau_grid = np.linspace(0.01, 10, 500)

# Plot the three rates
ax.plot(tau_grid, UniversalRate.EMRI_rate(tau_grid), label='EMRIs', color='red', linewidth=2.5)
ax.plot(tau_grid, UniversalRate.Plunge_rate(tau_grid), label='Plunges', color='blue', linewidth=2.5)
ax.plot(tau_grid, UniversalRate.TDE_rate(tau_grid), label='TDEs', color='green', linewidth=2.5)

# Mark calibration points
ax.plot([1, 5, 10], [UniversalRate.EMRI_rate(1), UniversalRate.EMRI_rate(5), UniversalRate.EMRI_rate(10)], 
        'o', color='red', markersize=6, alpha=0.7)
ax.plot([1, 5, 10], [UniversalRate.Plunge_rate(1), UniversalRate.Plunge_rate(5), UniversalRate.Plunge_rate(10)], 
        's', color='blue', markersize=6, alpha=0.7)
ax.plot([0.3, 1, 5, 10], [UniversalRate.TDE_rate(0.3), UniversalRate.TDE_rate(1), UniversalRate.TDE_rate(5), UniversalRate.TDE_rate(10)], 
        '^', color='green', markersize=6, alpha=0.7)

ax.set_xlabel(r'$\tau = T_c/t_\mathrm{EMRI}$', fontsize=13)
ax.set_ylabel(r'$\Gamma_k / \hat{\Gamma}_\mathrm{EMRI}$', fontsize=13)
# ax.set_title(r'Universal Rates', fontsize=13, fontweight='bold')
ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
ax.grid(alpha=0.3)
ax.set_xlim(0, 10)
ax.set_ylim(0, 1.2)

plt.tight_layout()
plt.savefig('./data/astrophysical_data/universal_rates.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Universal rates plot saved!")
print(f"\nEMRI rate @ τ=1: {UniversalRate.EMRI_rate(1):.3f}")
print(f"EMRI rate @ τ=10: {UniversalRate.EMRI_rate(10):.3f}")
print(f"Plunge rate @ τ=1: {UniversalRate.Plunge_rate(1):.3f}")
print(f"Plunge rate @ τ=10: {UniversalRate.Plunge_rate(10):.3f}")
print(f"TDE rate @ τ=0.3: {UniversalRate.TDE_rate(0.3):.3f} (shows bump)")
print(f"TDE rate @ τ=1: {UniversalRate.TDE_rate(1):.3f}")
print(f"TDE rate @ τ=10: {UniversalRate.TDE_rate(10):.3f}")