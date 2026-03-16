from utils import *

from galaxy import *
from nsc import NSC
from density import DehnenProfile
from relaxation import RelaxationModel
from rate import RateModel


lgMgal = 10.0
z_obs = 0.5
m = 2.0

obj = Galaxy.check_nucleation(lgMgal, z_obs)
if obj:
    print(obj.z_gal)
    print(obj.lgMBH_mass())

    NSC_obj = NSC(obj, obj.lgMBH_mass())

    # r_inf = NSC_obj.r_influence(unit='pc')
    # r_cap = NSC_obj.r_capture(unit='pc')
    # r_tid = NSC_obj.r_tidal(unit='pc')

    # print(r_cap, r_tid, r_inf, )

    dehnen_obj = DehnenProfile(NSC_obj)

    r = np.logspace(-5, 3, 100)  # pc   
    component_masses=np.random.uniform(1., 100, 100000)
    Ntot = 1E5
    # n_star = dehnen_obj.dehnen_number_density(r, Ntot=Ntot, kind='EMRI')
    # nr_star = dehnen_obj.radial_number_distribution(r, Ntot=Ntot, kind='EMRI')
    # Ncum_star = dehnen_obj.cumulative_number_within_radius(r, Ntot=Ntot, kind='EMRI')
    # rho_star = dehnen_obj.mass_density(r, Ntot=Ntot, component_masses=component_masses, kind='EMRI', unit='Msun/pc^3')

    # Plotting.plot_NSCprofile(NSC_obj, dehnen_obj, r, component_masses=component_masses, kind='EMRI', Ntot=Ntot)

    relax_obj = RelaxationModel(NSC_obj, dehnen_obj)
    rho_at_rinfl = relax_obj.rho_at_rinfl(Ntot=Ntot, component_masses=component_masses, kvir=1.0, kind='EMRI', unit='Msun/pc^3', renormalize=False)
    print(rho_at_rinfl)

    print(relax_obj.t_relax(rho_at_rinfl, Ntot=Ntot, component_masses=component_masses, kvir=1.0, kind='EMRI', mbar=10, unit='Gyr',))

    print(relax_obj.t_relax_at_rinfl(Ntot=Ntot, component_masses=component_masses, kvir=1.0, kind='EMRI', mbar=10., unit='Gyr'))

    tau = np.linspace(0, 1, 1000)
    rate_obj = RateModel(NSC_obj, tau)

    print(f"t_EMRI: {rate_obj.time_to_peak_EMRI_rate()}, Gamma_hat_EMRI: {rate_obj.peak_EMRI_rate()}")

    Plotting.plot_rate_evolution(tau, rate_obj.universal_EMRI_rate(), rate_obj.universal_TDE_rate())
    
    pdf = rate_obj.universal_EMRI_rate()
    samples = Distributions(tau, pdf).get_samples(size=1000)

    import matplotlib.pyplot as plt
    plt.hist(samples, bins=50, density=True)
    plt.xlabel(r'$\tau$')
    plt.ylabel('samples')
    plt.show()
    plt.savefig('samples_Rate_EMRI.pdf', dpi=200)


else:
    print("No nucleation in this draw.")






# # TESTING GALAXY CLASS
# Mstar = np.logspace(6, 12, 1000)     # Msun, linear
# lgMgal = np.log10(Mstar)
# lgMBH = []
# sigma = []

# z_obs = 0.5
# m = 2.0

# for i in range(len(lgMgal)):
#     gal = Galaxy.check_nucleation(lgMgal[i], z_obs)  # returns None if no nucleation
#     if gal is None:
#         lgMBH.append(np.nan)
#         sigma.append(np.nan)
#     else:
#         lgMBH.append(gal.lgMBH_mass())
#         sigma.append(gal.sigma(unit='km/s'))

# Plotting.plot_lgMgal_vs_lgMBH(lgMgal, lgMBH)
# Plotting.plot_lgMgal_vs_lgsigma(lgMgal, sigma)
# Plotting.plot_lgsigma_vs_lgMBH(sigma, lgMBH)
