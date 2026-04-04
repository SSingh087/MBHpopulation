from utils import *

from galaxy import *
from nsc import NSC
from density import DehnenProfile
from relaxation import RelaxationModel
from rate import RateModel
from evolution import CuspEvolution
from cosmology import LastMajorMerger, CosmologyModel

cosmo_model = CosmologyModel()
lgMgal = 10.0
z_obs = 0.5
m = 2.0


if Galaxy.check_nucleation(lgMgal, z_obs):

    obj = Galaxy(lgMgal=lgMgal, z_gal=z_obs, nucleation_occurs=True)
    print(obj.z_gal)
    print(obj.lgMBH_mass())

    NSC_obj = NSC(obj, obj.lgMBH_mass())

    r_inf = NSC_obj.r_influence(unit='pc')
    r_cap = NSC_obj.r_capture(unit='pc')
    r_tid = NSC_obj.r_tidal(unit='pc')

    print(r_cap, r_tid, r_inf, )

    dehnen_obj = DehnenProfile(NSC_obj)

    r = np.logspace(-5, 3, 100)  # pc   
    component_masses=np.random.uniform(1., 100, 10)
    Ntot = 1E5
    n_star = dehnen_obj.dehnen_number_density(r, Ntot=Ntot, kind='EMRI')
    nr_star = dehnen_obj.radial_number_distribution(r, Ntot=Ntot, kind='EMRI')
    Ncum_star = dehnen_obj.cumulative_number(r, Ntot=Ntot, kind='EMRI')
    total_N_star = dehnen_obj.number_of_CO_within_shell(r[0], r[-1], Ntot=Ntot, kind='EMRI')
    rho_star = dehnen_obj.mass_density(r, Ntot=Ntot, component_masses=component_masses, kind='EMRI', unit='Msun/pc^3')

    print(Ncum_star[-1], np.log10(Ncum_star[-1]))
    print(total_N_star, np.log10(total_N_star))
    Plotting.plot_NSCprofile(NSC_obj, dehnen_obj, r, component_masses=component_masses, kind='EMRI', Ntot=Ntot)

    plt.loglog(r[:-1], Ncum_star, label='cumulative number')
    plt.xlabel('Radius (pc)')
    plt.ylabel('Cumulative Number')
    plt.legend()
    plt.savefig('cumulative_number.pdf', dpi=200)
    plt.show()

    relax_obj = RelaxationModel(NSC_obj, dehnen_obj)
    rho_at_rinfl = relax_obj.rho_at_rinfl(Ntot=Ntot, component_masses=component_masses, kvir=1.0, kind='EMRI', unit='Msun/pc^3', renormalize=False)
    print(rho_at_rinfl)

    print(relax_obj.t_relax(rho_at_rinfl, Ntot=Ntot, component_masses=component_masses, kvir=1.0, kind='EMRI', mbar=10, unit='Gyr',))

    print(relax_obj.t_relax_at_rinfl(Ntot=Ntot, component_masses=component_masses, kvir=1.0, kind='EMRI', mbar=10., unit='Gyr'))

    tau_grid = np.linspace(0, 1, 1000)
    rate_obj = RateModel(NSC_obj)

    print(f"t_EMRI: {rate_obj.time_to_peak_EMRI_rate()}, Gamma_hat_EMRI: {rate_obj.peak_EMRI_rate()}")

    Plotting.plot_rate_evolution(tau_grid, rate_obj.universal_EMRI_rate(tau_grid), rate_obj.universal_TDE_rate(tau_grid))
    
    pdf = rate_obj.universal_EMRI_rate(tau_grid)
    samples = Distributions(tau_grid, pdf).get_samples(size=1000)

    import matplotlib.pyplot as plt
    plt.hist(samples, bins=50, density=True)
    plt.xlabel(r'$\tau$')
    plt.ylabel('samples')
    plt.tight_layout()
    plt.savefig('samples_Rate_EMRI.pdf', dpi=200)
    plt.show()

    cusp_evolution_object = CuspEvolution(NSC_obj, relax_obj, rate_obj, LastMajorMerger(CosmologyModel()))
    t_on = cusp_evolution_object.cusp_turn_on_time(Ntot=Ntot, component_masses=component_masses, kvir=1.0, kind='EMRI', mbar=10., unit='Gyr')
    cusp_age = cusp_evolution_object.cusp_age(Ntot=Ntot, component_masses=component_masses, kvir=1.0, kind='EMRI', mbar=10., unit='Gyr')
    print(f"t_ON : {t_on}, T_c : {cusp_age}")
    
    accumulated_EMRIs = cusp_evolution_object.accumulated_objects_within_time(Ntot=Ntot, component_masses=component_masses, kvir=1.0, kind='EMRI', mbar=10., unit='Gyr')

    print(f"Total number of EMRIs accumulated for a cusp age of {cusp_age} Gyr is {accumulated_EMRIs}.")

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
