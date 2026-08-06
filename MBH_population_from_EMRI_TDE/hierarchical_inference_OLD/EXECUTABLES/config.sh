#!/bin/sh


# ===========================
# these priors need to be set from some sript 

# def compute_MBH_bounds_from_galaxies(
#         N_samples=10000,
#         galaxy_mass_function="Schechter",
#     ):
#     # 1. Draw galaxies from the mass function
#     Mgal = draw_galaxies(N_samples, galaxy_mass_function)

#     # 2. Compute r_eff(Mgal) from arXiv:2106.07663v2
#     r_eff = r_eff_relation(Mgal)

#     # 3. Compute velocity dispersion sigma
#     sigma = np.sqrt(k * Mgal * G_sol.value / r_eff)

#     # 4. Compute MBH mass from MBH–sigma or MBH–Mgal
#     MBH = MBH_from_Mgal(Mgal, sigma)

#     # 5. Return bounds
#     return np.log10(MBH.min()), np.log10(MBH.max())

# ===========================

B_PRIOR_LAMBDA_M_LOW=-3
B_PRIOR_LAMBDA_M_HIGH=-1.01

# mass dist of the CO
B_PRIOR_LAMBDA_MU_LOW=-4
B_PRIOR_LAMBDA_MU_HIGH=-1.5

# spin 
B_PRIOR_MU_A_LOW=0.1
B_PRIOR_MU_A_HIGH=0.7

B_PRIOR_SIGMA_A_LOW=0.001
B_PRIOR_SIGMA_A_HIGH=0.05