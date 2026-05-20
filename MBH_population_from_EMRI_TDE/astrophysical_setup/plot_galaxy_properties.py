import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('font', family='serif', serif=['Computer Modern'], size=12)
matplotlib.rc('text', usetex=True)

def initial_MBHspin(lgMBH_mass):

    alpha0_low_mass = 2.5
    lambda_alpha_low_mass = 0.1
    beta_low_mass = 5.0

    alpha0_high_mass = 9.5
    lambda_alpha_high_mass = 0.1
    beta_high_mass = 5.0

    spins = np.zeros(len(lgMBH_mass))
    spins[lgMBH_mass < 6] = np.random.beta(alpha0_low_mass + lambda_alpha_low_mass * (lgMBH_mass[lgMBH_mass < 6] - 6), beta_low_mass)
    spins[lgMBH_mass >= 6] = np.random.beta(alpha0_high_mass + lambda_alpha_high_mass * (lgMBH_mass[lgMBH_mass >= 6] - 6), beta_high_mass)

    return spins

lgMBH_mass_grid = np.linspace(4.0, 10.0, 100000)
spin_samples = np.array(initial_MBHspin(lgMBH_mass_grid))

# Separate into low and high mass components
low_mass_mask = lgMBH_mass_grid < 6
high_mass_mask = lgMBH_mass_grid >= 6

spin_low_mass = spin_samples[low_mass_mask]
spin_high_mass = spin_samples[high_mass_mask]


plt.hist(spin_samples, bins=100, histtype='step', linewidth=2.5, density=True, 
        color='black', label=f'Average spin={spin_samples.mean():.3f}')
plt.hist(spin_low_mass, bins=50, histtype='step', linewidth=2, density=True, 
        color='steelblue', alpha=0.6, label=r'$M_{{\rm MBH}} < 10^6 M_\odot$ contribution')
plt.hist(spin_high_mass, bins=50, histtype='step', linewidth=2, density=True, 
        color='coral', alpha=0.6, label=r'$M_{{\rm MBH}} > 10^6 M_\odot$ contribution')
plt.vlines(x=0.3, ymin=0, ymax=plt.ylim()[1], color='gray', linestyle='--', linewidth=2, alpha=0.7)
plt.vlines(x=0.7, ymin=0, ymax=plt.ylim()[1], color='gray', linestyle=':', linewidth=2, alpha=0.7)
plt.xlabel(r'$a_{\rm BH}$ (spin)', fontsize=13)
plt.ylabel('', fontsize=13)
plt.yticks([])
plt.legend(fontsize=11, loc='upper right')
plt.grid(alpha=0.3)
plt.xlim(0, 1)

plt.tight_layout()
plt.savefig('./data/astrophysical_data/MBHSpin_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
