from rate import UniversalRate
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='serif', serif=['Computer Modern'], size=12)
matplotlib.rc('text', usetex=True)


EMRI_rate = UniversalRate.EMRI_rate
Plunge_rate = UniversalRate.Plunge_rate
TDE_rate = UniversalRate.TDE_rate

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('font', family='serif', serif=['Computer Modern'], size=12)
matplotlib.rc('text', usetex=True)

fig, ax = plt.subplots(figsize=(10, 6))

import numpy as np

tau_grid = np.linspace(0.01, 10, 500)

# Plot the three rates
ax.plot(tau_grid, UniversalRate.EMRI_rate(tau_grid), label='EMRI', color='red', linewidth=2.5)
ax.plot(tau_grid, UniversalRate.Plunge_rate(tau_grid), label='Plunges', color='blue', linewidth=2.5)
ax.plot(tau_grid, UniversalRate.TDE_rate(tau_grid), label='TD', color='green', linewidth=2.5)

# Mark calibration points
ax.plot([1, 5, 10], [UniversalRate.EMRI_rate(1), UniversalRate.EMRI_rate(5), UniversalRate.EMRI_rate(10)], 
        'o', color='red', markersize=6, alpha=0.7)
ax.plot([1, 5, 10], [UniversalRate.Plunge_rate(1), UniversalRate.Plunge_rate(5), UniversalRate.Plunge_rate(10)], 
        's', color='blue', markersize=6, alpha=0.7)
ax.plot([0.3, 1, 5, 10], [UniversalRate.TDE_rate(0.3), UniversalRate.TDE_rate(1), UniversalRate.TDE_rate(5), UniversalRate.TDE_rate(10)], 
        '^', color='green', markersize=6, alpha=0.7)

ax.set_xlabel(r'$\tau = t/t_E$', fontsize=13)
ax.set_ylabel(r'$\Gamma_i / \hat{\Gamma}_i$', fontsize=13)
ax.set_title(r'Universal Rate Functions (arXiv:2205.06277v1 Fig 6)', fontsize=13, fontweight='bold')
ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
ax.grid(alpha=0.3)
ax.set_xlim(0, 10)
ax.set_ylim(0, 1.2)

plt.tight_layout()
plt.savefig('universal_rates.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Universal rates plot saved!")
print(f"\nEMRI rate @ τ=1: {UniversalRate.EMRI_rate(1):.3f}")
print(f"EMRI rate @ τ=10: {UniversalRate.EMRI_rate(10):.3f}")
print(f"Plunge rate @ τ=1: {UniversalRate.Plunge_rate(1):.3f}")
print(f"Plunge rate @ τ=10: {UniversalRate.Plunge_rate(10):.3f}")
print(f"TDE rate @ τ=0.3: {UniversalRate.TDE_rate(0.3):.3f} (shows bump)")
print(f"TDE rate @ τ=1: {UniversalRate.TDE_rate(1):.3f}")
print(f"TDE rate @ τ=10: {UniversalRate.TDE_rate(10):.3f}")