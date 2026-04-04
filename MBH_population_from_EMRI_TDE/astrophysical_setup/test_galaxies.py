from utils import *

from galaxy import *
from nsc import NSC, CompactObject
from density import DehnenProfile
from relaxation import RelaxationModel
from rate import RateModel, UniversalRate
from evolution import CuspEvolution
from cosmology import LastMajorMerger, CosmologyModel, GalaxyStellarMassFunction, MBHMassFunction
# from distributions import Distributions

cosmo_model = CosmologyModel()
N_objs = 10

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

CO_objs = CompactObject(nsc=NSC_obj, masses={'sBH': 10.0, 'star': 1.0}, total_mass={'sBH': 20.0, 'star': 100.0}, types_CO=['sBH', 'star'], types_masses='same_mass', type_CO_limits=None)

print("Total number of sBHs:", CO_objs.total_number['sBH'])
print("Total number of stars:", CO_objs.total_number['star'])
# print("Mass of each sBH:", CO_objs.component_masses['sBH'])
# print("Mass of each star:", CO_objs.component_masses['star'])

r_inf = NSC_obj.r_influence(unit='pc')
r_cap = NSC_obj.r_capture(unit='pc')
r_tid = NSC_obj.r_tidal(unit='pc')

print("Capture radius:", r_cap, "Tidal radius:", r_tid, "Influence radius:", r_inf)

dehnen_obj = DehnenProfile(NSC_obj, CO_objs)

# these properties of the COs in the NSC, can be scaled with galaxy properties in the future

N = z_grid[nucleation_indices].shape[0]
r_grid = np.logspace(-5, 3, 100).reshape(1, -1).repeat(N, axis=0)# pc  this needs to be 2D for the DehnenProfile methods, so we make it (N_nucleated_galaxies, Nr)

n_star = dehnen_obj.dehnen_number_density(r_grid, kind='EMRI')
nr_star = dehnen_obj.radial_number_distribution(r_grid, kind='EMRI')
Ncum_star = dehnen_obj.cumulative_number(r_grid, kind='EMRI')


# Plotting.plot_NSCprofile(NSC_obj, CO_objs, dehnen_obj, r_grid, kind='EMRI')

print("mass density at influence radius:", mass_density_at_rinfl := dehnen_obj.mass_density_at_rinfl(kvir=1.0, unit='Msun/pc^3'))

r_min = r_grid[:, 0]      # smallest radius
r_max = r_grid[:, -1]

total_CO_in_shell = dehnen_obj.number_of_CO_within_shell(r_min, r_max, kind='EMRI')

# print(Ncum_star, np.log10(Ncum_star))
# print(total_CO_in_shell, np.log10(total_CO_in_shell))

relax_obj = RelaxationModel(nsc=NSC_obj, compact_object=CO_objs, profile=dehnen_obj)

print("t_relax:", relax_obj.t_relax(rho_r=mass_density_at_rinfl, kvir=1.0, unit='Gyr'))

print("t_relax_at_rinfl:", relax_obj.t_relax_at_rinfl(kvir=1.0, unit='Gyr'))

tau_grid = np.linspace(0, 1, 1000)
rate_obj = RateModel(NSC_obj)

print(f"t_EMRI: {rate_obj.time_to_peak_EMRI_rate()}, Gamma_hat_EMRI: {rate_obj.peak_EMRI_rate()}")

# Plotting.plot_rate_evolution(tau_grid, UniversalRate.EMRI_rate(tau_grid), UniversalRate.TDE_rate(tau_grid))

cusp_evolution_object = CuspEvolution(nsc=NSC_obj, compact_object=CO_objs, relaxation=relax_obj, rate_model=rate_obj, LastMajorMerger=LastMajorMerger(CosmologyModel()))

t_on = cusp_evolution_object.cusp_turn_on_time(kvir=1.0, unit='Gyr')

cusp_age = cusp_evolution_object.cusp_age(kvir=1.0, unit='Gyr')
print(f"t_ON : {t_on}, T_c : {cusp_age}")

print("tau = ", tau := cusp_evolution_object.evaluate_tau(kvir=1.0, unit='Gyr', A=MBH_A, B=MBH_B, sigma_0=MBH_sigma0, MBH_scatter=MBH_scatter))

accumulated_EMRIs = cusp_evolution_object.accumulated_objects_within_time(kvir=1.0, kind='EMRI', unit='Gyr')

accumulated_TDEs = cusp_evolution_object.accumulated_objects_within_time(kvir=1.0, kind='TDE', unit='Gyr')

print(f"Total number of EMRIs accumulated for a cusp age of {cusp_age} Gyr is {accumulated_EMRIs}.")
print(f"Total number of TDEs accumulated for a cusp age of {cusp_age} Gyr is {accumulated_TDEs}.")




