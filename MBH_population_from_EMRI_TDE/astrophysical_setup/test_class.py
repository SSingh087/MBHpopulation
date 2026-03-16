from galaxy import *
from utils import *

Mstar = np.logspace(6, 12, 10)     # Msun, linear
lgMgal = np.log10(Mstar)
lgMBH = []
sigma = []

z_obs = 0.5
m = 2.0


# TESTING GALAXY CLASS
# for i in range(len(lgMgal)):
#     gal = Galaxy.check_nucleation(lgMgal[i])  # returns None if no nucleation
#     if gal is None:
#         lgMBH.append(np.nan)
#         sigma.append(np.nan)
#     else:
#         lgMBH.append(gal.lgMBH_mass())
#         sigma.append(gal.sigma(unit='km/s'))

# Plotting.plot_lgMgal_vs_lgMBH(lgMgal, lgMBH)
# Plotting.plot_lgMgal_vs_lgsigma(lgMgal, sigma)

# SINGLE VALUE CHECK
lgMgal = 10.0
obj = NSCProfile(lgMgal, z_obs, gamma_initial=1.5).check_nucleation(lgMgal, z_obs)
if obj:
    print(obj.z_gal)
    # obj.cusp_turn_on_time(Ntot=1e5, component_masses=np.random.uniform(1., 100, 100000), kvir=1.0, kind='EMRI', mbar=10, unit='Gyr')

    # obj.cusp_age(Ntot=1e5, component_masses=np.random.uniform(1., 100, 100000), kvir=1.0, kind='EMRI', mbar=10, unit='Gyr')

    obj.accumulated_objects_within_time(Ntot=1e5, component_masses=np.random.uniform(1., 100, 100000), kvir=1.0, kind='EMRI', mbar=10, unit='Gyr', A=7.87, B=4.55, sigma_0=160.0, MBH_scatter=0.53)

#     # obj.accumulated_objects_within_time(A=7.87, B=4.55, sigma_0=160.0, MBH_scatter=0.53)
#     # r = np.logspace(-5, 3, 100)  # pc
#     # Plotting.plot_NSCprofile(obj, r, component_masses=np.random.uniform(1., 100, 100000), kind='EMRI', Ntot=1e5)

#     # r_inf = obj.influence_radius(unit='pc')
#     # r_cap = obj.capture_radius(unit='pc')
#     # r_tid = obj.tidal_radius_star(unit='pc')
#     # n_star = obj.dehnen_number_density(r, Ntot=1e5, kind='EMRI')
#     # nr_star = obj.radial_number_distribution(r, Ntot=1e5, kind='EMRI')
#     # # Ncum_star = obj.cumulative_number_within_radius(r, Ntot=1e5, kind='EMRI')
#     # rho_star = obj.mass_density(r, Ntot=1e5, component_masses=np.random.uniform(1., 100, 100000), kind='EMRI', unit='Msun/pc^3')
#     # print(f"r_inf={r_inf:.3e} pc, r_cap={r_cap:.3e} pc, r_tid={r_tid:.3e} pc,")
#     # # N(<r_max)={Ncum_star.max():.5e}")
#     # print(np.max(n_star), r_tid, r_inf, r_cap)
#     # # breakpoint()
#     # plt.loglog(r, n_star, label='$n_i^\mathrm{EMRI}(r)$')
#     # plt.loglog(r, nr_star, label='$n_r^\mathrm{EMRI}(r)$')
#     # # plt.loglog(r, Ncum_star, label='$N_\mathrm{EMRI}(r)$')
#     # plt.loglog(r, rho_star, label='$\\rho^\mathrm{EMRI}(r)$')
#     # # plt.vlines(r_tid, np.min(n_star), np.max(n_star), label='$r_\mathrm{star}$', color='red')
#     # plt.vlines(r_inf, np.min(n_star), np.max(n_star), label='$r_\mathrm{infl.}$', linestyle='--', color='black')
#     # # plt.vlines(r_cap, np.min(n_star), np.max(n_star), label='$r_\mathrm{sBH}$', linestyle=':', color='pink')
#     # plt.xlabel('pc')
#     # plt.xlim(1e-5, 1e3)
#     # plt.ylim(1e-5, 1e8)
#     # plt.legend()    
#     # plt.savefig(f"{lgMgal}_properties.pdf", dpi=200)
#     # plt.show()

#     # print(obj.rho_at_rinfl(Ntot=1e5, component_masses=np.random.uniform(1., 100, 100000), kvir=1.0, kind='EMRI', unit='Msun/pc^3', renormalize=True))
#     # t_rlx = obj.t_relax_at_rinfl(Ntot=1e7, component_masses=np.random.uniform(1., 100, 100000), kvir=1.0, kind='EMRI', mbar=10, unit='Gyr')
#     print(t_rlx)
# else:
#     print("No nucleation in this draw.")



# from cosmology import *
# cosmo = CosmologyModel()
# z_obs = 0.5
# m = 2.0
# print(cosmo.sample_lmm_times_Gyr(z_obs, m))
# z_grid, pdf, cdf = cosmo.lmm_pdf_cdf(z_obs, m)
# plt.plot(z_grid, pdf, label='PDF')
# plt.plot(z_grid, cdf, label='CDF')
# plt.xlabel('Redshift z')
# plt.legend()
# plt.savefig('lmm_pdf_cdf.pdf', dpi=200)
# plt.show()
# for i in range(1000):
#     z_LMM, t_LMM, t_obs = cosmo.sample_lmm_times_Gyr(z_obs, m)
#     print(z_obs < z_LMM, t_obs > t_LMM)