#!/bin/sh
. ./EXECUTABLES/config.sh

# ============================================================
# PRIORS for population A hyper-parameters
# Cross-ref: FixedLimitSchechterFunction, DoubleBetaDistribution, DoubleGaussianDistribution
# ============================================================

# MBH mass  (FixedLimitSchechterFunction)
A_PRIOR_XC_M_LOW=4.0
A_PRIOR_XC_M_HIGH=8.0
A_PRIOR_LAM_SCHECHTER_LOW=6.5
A_PRIOR_LAM_SCHECHTER_HIGH=15.0

# MBH spin  (DoubleBetaDistribution → alpha_1, alpha_2, beta_1, beta_2, weight)
A_PRIOR_ALPHA_1_MBHSpin_LOW=0.5
A_PRIOR_ALPHA_1_MBHSpin_HIGH=10.0
A_PRIOR_ALPHA_2_MBHSpin_LOW=0.5
A_PRIOR_ALPHA_2_MBHSpin_HIGH=15.0
A_PRIOR_BETA_1_MBHSpin_LOW=0.5
A_PRIOR_BETA_1_MBHSpin_HIGH=15.0
A_PRIOR_BETA_2_MBHSpin_LOW=0.5
A_PRIOR_BETA_2_MBHSpin_HIGH=15.0
A_PRIOR_WEIGHT_MBHSpin_LOW=0.001
A_PRIOR_WEIGHT_MBHSpin_HIGH=0.999

# CO mass  (DoubleGaussianDistribution → mu_sBH, sigma_sBH, mu_star, sigma_star, weight)
A_PRIOR_MU_sBH_LOW=5.0
A_PRIOR_MU_sBH_HIGH=15.0
A_PRIOR_SIGMA_sBH_LOW=0.1
A_PRIOR_SIGMA_sBH_HIGH=5.0
A_PRIOR_MU_STAR_LOW=0.1
A_PRIOR_MU_STAR_HIGH=3.0
A_PRIOR_SIGMA_STAR_LOW=0.1
A_PRIOR_SIGMA_STAR_HIGH=3.0
A_PRIOR_WEIGHT_COmass_LOW=0.001
A_PRIOR_WEIGHT_COmass_HIGH=0.999

# ============================================================
# PRIORS for population B hyper-parameters
# Cross-ref: FixedLimitPowerLaw, DoubleGaussianDistribution, DoubleGaussianDistribution
# ============================================================

# MBH mass  (FixedLimitsPowerLaw → lam)
B_PRIOR_LAMBDA_M_LOW=-3.0
B_PRIOR_LAMBDA_M_HIGH=-1.01

# MBH spin  (DoubleGaussianDistribution → mu_1, sigma_1, mu_2, sigma_2, weight)
B_PRIOR_MU_1_MBHSpin_LOW=0.05
B_PRIOR_MU_1_MBHSpin_HIGH=0.5
B_PRIOR_SIGMA_1_MBHSpin_LOW=0.001
B_PRIOR_SIGMA_1_MBHSpin_HIGH=0.3
B_PRIOR_MU_2_MBHSpin_LOW=0.5
B_PRIOR_MU_2_MBHSpin_HIGH=0.99
B_PRIOR_SIGMA_2_MBHSpin_LOW=0.001
B_PRIOR_SIGMA_2_MBHSpin_HIGH=0.3
B_PRIOR_WEIGHT_MBHSpin_LOW=0.001
B_PRIOR_WEIGHT_MBHSpin_HIGH=0.999

# CO mass  (DoubleGaussianDistribution → mu_sBH, sigma_sBH, mu_star, sigma_star, weight)
B_PRIOR_MU_sBH_LOW=5.0
B_PRIOR_MU_sBH_HIGH=15.0
B_PRIOR_SIGMA_sBH_LOW=0.1
B_PRIOR_SIGMA_sBH_HIGH=5.0
B_PRIOR_MU_STAR_LOW=0.1
B_PRIOR_MU_STAR_HIGH=3.0
B_PRIOR_SIGMA_STAR_LOW=0.1
B_PRIOR_SIGMA_STAR_HIGH=3.0
B_PRIOR_WEIGHT_COmass_LOW=0.001
B_PRIOR_WEIGHT_COmass_HIGH=0.999

# ============================================================
# PRIORS for population MIX hyper-parameters
# (placeholder — fill once MIX distributions are defined
#  in make_dist / get_true_x / get_posterior_samples)
# ============================================================

# MIX_PRIOR_...

# ============================================================
# SHARED mixture weight
# ============================================================
PRIOR_WEIGHT_LOW=0.001
PRIOR_WEIGHT_HIGH=0.999