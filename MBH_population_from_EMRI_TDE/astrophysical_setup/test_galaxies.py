from utils import *

from galaxy import *
from nsc import NSC
from density import DehnenProfile
from relaxation import RelaxationModel
from rate import RateModel
from evolution import CuspEvolution
from cosmology import LastMajorMerger, CosmologyModel, GalaxyStellarMassFunction, MBHMassFunction
# from distributions import Distributions

cosmo_model = CosmologyModel()
N_objs = 5

# right now this is just a random draw of redshifts for testing.
# In the future, we can use a more physically motivated distribution
# of redshifts for the galaxy population.
z_grid = np.random.uniform(0.01, 10.0, N_objs) 

GSMF = GalaxyStellarMassFunction()
lgMgal_samples = GSMF.sample_gsmf(z_gal=z_grid, size=N_objs)

m = 2.0 # this is the slope of merger rate


nucleation_indices = Galaxy.check_nucleation(lgMgal_samples, z_grid)
 
obj = Galaxy(lgMgal=lgMgal_samples, z_gal=z_grid, nucleation_occurs=nucleation_indices)

print("obj.z_gal:", obj.z_gal)
print("obj.lgMBH_mass:", obj.lgMBH_mass)

NSC_obj = NSC(obj)

r_inf = NSC_obj.r_influence(unit='pc')
r_cap = NSC_obj.r_capture(unit='pc')
r_tid = NSC_obj.r_tidal(unit='pc')

# print(r_cap, r_tid, r_inf)

dehnen_obj = DehnenProfile(NSC_obj)

# these properties of the COs in the NSC, can be scaled with galaxy properties in the future

N = z_grid[nucleation_indices].shape[0]
r_grid = np.logspace(-5, 3, 100).reshape(1, -1).repeat(N, axis=0)# pc  this needs to be 2D for the DehnenProfile methods, so we make it (N_nucleated_galaxies, Nr)

N_COs = 10 # number of different CO species (e.g. MS, WD, NS, sBH, etc.)

# N_objs x N_COs array of component masses for each galaxy and each CO species
component_masses = np.full_like(z_grid[nucleation_indices], N_COs)
Ntot = 1E5 * np.ones_like(z_grid[nucleation_indices])


n_star = dehnen_obj.dehnen_number_density(r_grid, Ntot=Ntot, kind='EMRI')
nr_star = dehnen_obj.radial_number_distribution(r_grid, Ntot=Ntot, kind='EMRI')
Ncum_star = dehnen_obj.cumulative_number(r_grid, Ntot=Ntot, kind='EMRI')

# Plotting.plot_NSCprofile(NSC_obj, dehnen_obj, r_grid, component_masses=component_masses, kind='EMRI', Ntot=Ntot)



r_min = r_grid[:, 0]      # smallest radius
r_max = r_grid[:, -1]

breakpoint()
total_N_star = dehnen_obj.number_of_CO_within_shell(r_min, r_max, Ntot=Ntot, kind='EMRI')


rho_star = dehnen_obj.mass_density(r_grid, Ntot=Ntot, component_masses=component_masses, kind='EMRI', unit='Msun/pc^3')


# print(Ncum_star[-1], np.log10(Ncum_star[-1]))
print(total_N_star, np.log10(total_N_star))

# plt.loglog(r_grid[:-1], Ncum_star.T, label='cumulative number')
# plt.xlabel('Radius (pc)')
# plt.ylabel('Cumulative Number')
# plt.legend()
# plt.savefig('cumulative_number.pdf', dpi=200)
# plt.show()



relax_obj = RelaxationModel(NSC_obj, dehnen_obj)

rho_at_rinfl = relax_obj.profile.mass_density_at_rinfl(Ntot=Ntot, component_masses=component_masses, kvir=1.0, kind='EMRI', unit='Msun/pc^3')

print("rho_at_rinfl:", rho_at_rinfl)

print("t_relax:", relax_obj.t_relax(rho_at_rinfl, Ntot=Ntot, component_masses=component_masses, kvir=1.0, kind='EMRI', mbar=10, unit='Gyr',))

print("t_relax_at_rinfl:", relax_obj.t_relax_at_rinfl(Ntot=Ntot, component_masses=component_masses, kvir=1.0, kind='EMRI', mbar=10., unit='Gyr'))

tau_grid = np.linspace(0, 1, 1000)
rate_obj = RateModel(NSC_obj)

print(f"t_EMRI: {rate_obj.time_to_peak_EMRI_rate()}, Gamma_hat_EMRI: {rate_obj.peak_EMRI_rate()}")

# Plotting.plot_rate_evolution(tau_grid, rate_obj.universal_EMRI_rate(tau_grid), rate_obj.universal_TDE_rate(tau_grid))

# pdf = rate_obj.universal_EMRI_rate(tau_grid)
# samples = Distributions(tau_grid, pdf).get_samples(size=1000)

# import matplotlib.pyplot as plt
# plt.hist(samples, bins=50, density=True)
# plt.xlabel(r'$\tau$')
# plt.ylabel('samples')
# plt.tight_layout()
# plt.savefig('samples_Rate_EMRI.pdf', dpi=200)
# plt.show()


cusp_evolution_object = CuspEvolution(NSC_obj, relax_obj, rate_obj, LastMajorMerger(CosmologyModel()))
t_on = cusp_evolution_object.cusp_turn_on_time(Ntot=Ntot, component_masses=component_masses, kvir=1.0, kind='EMRI', mbar=10., unit='Gyr')

cusp_age = cusp_evolution_object.cusp_age(Ntot=Ntot, component_masses=component_masses, kvir=1.0, kind='EMRI', mbar=10., unit='Gyr')
print(f"t_ON : {t_on}, T_c : {cusp_age}")

print(cusp_evolution_object.evaluate_tau(Ntot=Ntot, component_masses=component_masses, kvir=1.0, kind='EMRI', mbar=10., unit='Gyr', A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter))

accumulated_EMRIs = cusp_evolution_object.accumulated_objects_within_time(Ntot=Ntot, component_masses=component_masses, kvir=1.0, kind='EMRI', mbar=10., unit='Gyr')


print(f"Total number of EMRIs accumulated for a cusp age of {cusp_age} Gyr is {accumulated_EMRIs}.")





