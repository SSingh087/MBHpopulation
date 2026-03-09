from galaxy import *

Mstar = np.logspace(6, 12, 10)     # Msun, linear
lgMgal = np.log10(Mstar)
lgMBH = []
sigma = []


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
obj = NSC.check_nucleation(lgMgal)
if obj:
    r = np.logspace(-5, 3, 100)  # pc
    r_inf = obj.influence_radius(unit='pc')
    r_cap = obj.capture_radius(unit='pc')
    r_tid = obj.tidal_radius_star(unit='pc')
    n_star = obj.dehnen_number_density(r, Ntot=1e5, gamma=1.5, kind='EMRI')
    nr_star = obj.radial_number_distribution(r, Ntot=1e5, gamma=1.5, kind='EMRI')
    Ncum_star = obj.cumulative_number(r, Ntot=1e5, gamma=1.5, kind='EMRI')
    rho_star = obj.mass_density(r, Ntot=1e5, component_masses=np.random.uniform(1., 100, 100000), gamma=1.5, kind='EMRI', unit='Msun/pc^3')
    # print(f"r_inf={r_inf:.3e} pc, r_cap={r_cap:.3e} pc, r_tid={r_tid:.3e} pc,")
    # N(<r_max)={Ncum_star.max():.5e}")
    print(np.max(n_star), r_tid, r_inf, r_cap)
    # breakpoint()
    plt.loglog(r, n_star, label='$n_i^\mathrm{EMRI}(r)$')
    plt.loglog(r, nr_star, label='$n_r^\mathrm{EMRI}(r)$')
    plt.loglog(r, Ncum_star, label='$N_\mathrm{EMRI}(r)$')
    plt.loglog(r, rho_star, label='$\\rho^\mathrm{EMRI}(r)$')
    # plt.vlines(r_tid, np.min(n_star), np.max(n_star), label='$r_\mathrm{star}$', color='red')
    plt.vlines(r_inf, np.min(n_star), np.max(n_star), label='$r_\mathrm{infl.}$', linestyle='--', color='black')
    # plt.vlines(r_cap, np.min(n_star), np.max(n_star), label='$r_\mathrm{sBH}$', linestyle=':', color='pink')
    plt.xlabel('pc')
    plt.legend()
    plt.savefig(f"{lgMgal}_properties.pdf", dpi=200)
    plt.show()

    print(obj.rho_at_rinf(Ntot=1e5, component_masses=np.random.uniform(1., 100, 100000), gamma=1.5, kvir=1.0, kind='EMRI', unit='Msun/pc^3', renormalize=True))
else:
    print("No nucleation in this draw.")
