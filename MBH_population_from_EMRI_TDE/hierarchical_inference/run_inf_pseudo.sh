#!/bin/sh

export GALAXIES=100
export UNIVERSE_MODEL=0
export Z_MAX=10.0
export SEED=42
export IS_CLUSTERING=0


OBSERVING_WINDOW_EMRI=4 # this is in years

OBSERVING_WINDOW_TDE_ZTF=400 # this is in days
OBSERVING_WINDOW_TDE_LSST=1000 # this is in days

BANDS_ZTF="ztfg ztfr ztfi" # default for ZTF
BANDS_LSST="lsstu lsstg lsstr lssti lsstz lssty"

SOURCE=$1
POPULATION=$2
SURVEY=${3:-'None'}


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
A_PRIOR_ALPHA_1_MBHSpin_HIGH=5.0
A_PRIOR_ALPHA_2_MBHSpin_LOW=5.1
A_PRIOR_ALPHA_2_MBHSpin_HIGH=15.0
A_PRIOR_BETA_1_MBHSpin_LOW=0.5
A_PRIOR_BETA_1_MBHSpin_HIGH=15.0
A_PRIOR_BETA_2_MBHSpin_LOW=0.5
A_PRIOR_BETA_2_MBHSpin_HIGH=15.0
A_PRIOR_WEIGHT_MBHSpin_LOW=0.001
A_PRIOR_WEIGHT_MBHSpin_HIGH=0.999

# e_pl  (DoubleBetaDistribution → alpha_1, alpha_2, beta_1, beta_2, weight)
A_PRIOR_ALPHA_1_e_pl_LOW=0.5
A_PRIOR_ALPHA_1_e_pl_HIGH=5.0
A_PRIOR_ALPHA_2_e_pl_LOW=5.1
A_PRIOR_ALPHA_2_e_pl_HIGH=15.0
A_PRIOR_BETA_1_e_pl_LOW=0.5
A_PRIOR_BETA_1_e_pl_HIGH=15.0
A_PRIOR_BETA_2_e_pl_LOW=0.5
A_PRIOR_BETA_2_e_pl_HIGH=15.0
A_PRIOR_WEIGHT_e_pl_LOW=0.001
A_PRIOR_WEIGHT_e_pl_HIGH=0.999

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

# Z_GAL DIST (BETA DIST)
A_PRIOR_Z_GAL_ALPHA_LOW=0.5
A_PRIOR_Z_GAL_ALPHA_HIGH=5.0
A_PRIOR_Z_GAL_BETA_LOW=5
A_PRIOR_Z_GAL_BETA_HIGH=10


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

# e_pl  (DoubleGaussianDistribution → mu_1, sigma_1, mu_2, sigma_2, weight)
B_PRIOR_MU_1_e_pl_LOW=0.05
B_PRIOR_MU_1_e_pl_HIGH=0.5
B_PRIOR_SIGMA_1_e_pl_LOW=0.001
B_PRIOR_SIGMA_1_e_pl_HIGH=0.3
B_PRIOR_MU_2_e_pl_LOW=0.5
B_PRIOR_MU_2_e_pl_HIGH=0.99
B_PRIOR_SIGMA_2_e_pl_LOW=0.001
B_PRIOR_SIGMA_2_e_pl_HIGH=0.3
B_PRIOR_WEIGHT_e_pl_LOW=0.001
B_PRIOR_WEIGHT_e_pl_HIGH=0.999

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

# Z_GAL DIST (BETA DIST)
B_PRIOR_Z_GAL_ALPHA_LOW=0.5
B_PRIOR_Z_GAL_ALPHA_HIGH=5.0
B_PRIOR_Z_GAL_BETA_LOW=5
B_PRIOR_Z_GAL_BETA_HIGH=10

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

echo "Doing inference for source $SOURCE and survey $SURVEY assuming POPULATION $POPULATION"

if [ "$POPULATION" = "A" ]; then
    python hierarchical_inference_A.py \
        --GALAXIES $GALAXIES \
        --SOURCE $SOURCE \
        --SURVEY $SURVEY \
        --xc              $A_PRIOR_XC_M_LOW            $A_PRIOR_XC_M_HIGH            \
        --lam_schechter   $A_PRIOR_LAM_SCHECHTER_LOW   $A_PRIOR_LAM_SCHECHTER_HIGH   \
        --alpha_1_MBHSpin $A_PRIOR_ALPHA_1_MBHSpin_LOW $A_PRIOR_ALPHA_1_MBHSpin_HIGH \
        --alpha_2_MBHSpin $A_PRIOR_ALPHA_2_MBHSpin_LOW $A_PRIOR_ALPHA_2_MBHSpin_HIGH \
        --beta_1_MBHSpin  $A_PRIOR_BETA_1_MBHSpin_LOW  $A_PRIOR_BETA_1_MBHSpin_HIGH  \
        --beta_2_MBHSpin  $A_PRIOR_BETA_2_MBHSpin_LOW  $A_PRIOR_BETA_2_MBHSpin_HIGH  \
        --weight_MBHSpin  $A_PRIOR_WEIGHT_MBHSpin_LOW  $A_PRIOR_WEIGHT_MBHSpin_HIGH  \
        --alpha_1_e_pl    $A_PRIOR_ALPHA_1_e_pl_LOW     $A_PRIOR_ALPHA_1_e_pl_HIGH     \
        --alpha_2_e_pl    $A_PRIOR_ALPHA_2_e_pl_LOW     $A_PRIOR_ALPHA_2_e_pl_HIGH     \
        --beta_1_e_pl     $A_PRIOR_BETA_1_e_pl_LOW      $A_PRIOR_BETA_1_e_pl_HIGH      \
        --beta_2_e_pl     $A_PRIOR_BETA_2_e_pl_LOW      $A_PRIOR_BETA_2_e_pl_HIGH      \
        --weight_e_pl      $A_PRIOR_WEIGHT_e_pl_LOW     $A_PRIOR_WEIGHT_e_pl_HIGH     \
        --mu_sBH          $A_PRIOR_MU_sBH_LOW           $A_PRIOR_MU_sBH_HIGH          \
        --sigma_sBH       $A_PRIOR_SIGMA_sBH_LOW        $A_PRIOR_SIGMA_sBH_HIGH       \
        --mu_star         $A_PRIOR_MU_STAR_LOW          $A_PRIOR_MU_STAR_HIGH         \
        --sigma_star      $A_PRIOR_SIGMA_STAR_LOW       $A_PRIOR_SIGMA_STAR_HIGH      \
        --weight_COmass   $A_PRIOR_WEIGHT_COmass_LOW    $A_PRIOR_WEIGHT_COmass_HIGH   \
        --z_gal_alpha     $A_PRIOR_Z_GAL_ALPHA_LOW      $A_PRIOR_Z_GAL_ALPHA_HIGH      \
        --z_gal_beta      $A_PRIOR_Z_GAL_BETA_LOW       $A_PRIOR_Z_GAL_BETA_HIGH       \

elif [ "$POPULATION" = "B" ]; then
    python hierarchical_inference_B.py \
        --GALAXIES $GALAXIES \
        --SOURCE $SOURCE \
        --SURVEY $SURVEY \
        --lambda_M        $B_PRIOR_LAMBDA_M_LOW         $B_PRIOR_LAMBDA_M_HIGH        \
        --mu_1_MBHSpin    $B_PRIOR_MU_1_MBHSpin_LOW     $B_PRIOR_MU_1_MBHSpin_HIGH    \
        --sigma_1_MBHSpin $B_PRIOR_SIGMA_1_MBHSpin_LOW  $B_PRIOR_SIGMA_1_MBHSpin_HIGH \
        --mu_2_MBHSpin    $B_PRIOR_MU_2_MBHSpin_LOW     $B_PRIOR_MU_2_MBHSpin_HIGH    \
        --sigma_2_MBHSpin $B_PRIOR_SIGMA_2_MBHSpin_LOW  $B_PRIOR_SIGMA_2_MBHSpin_HIGH \
        --weight_MBHSpin  $B_PRIOR_WEIGHT_MBHSpin_LOW   $B_PRIOR_WEIGHT_MBHSpin_HIGH  \
        --mu_1_e_pl       $B_PRIOR_MU_1_e_pl_LOW         $B_PRIOR_MU_1_e_pl_HIGH       \
        --sigma_1_e_pl    $B_PRIOR_SIGMA_1_e_pl_LOW      $B_PRIOR_SIGMA_1_e_pl_HIGH    \
        --mu_2_e_pl       $B_PRIOR_MU_2_e_pl_LOW         $B_PRIOR_MU_2_e_pl_HIGH       \
        --sigma_2_e_pl    $B_PRIOR_SIGMA_2_e_pl_LOW      $B_PRIOR_SIGMA_2_e_pl_HIGH    \
        --weight_e_pl     $B_PRIOR_WEIGHT_e_pl_LOW       $B_PRIOR_WEIGHT_e_pl_HIGH      \
        --mu_sBH          $B_PRIOR_MU_sBH_LOW            $B_PRIOR_MU_sBH_HIGH          \
        --sigma_sBH       $B_PRIOR_SIGMA_sBH_LOW         $B_PRIOR_SIGMA_sBH_HIGH       \
        --mu_star         $B_PRIOR_MU_STAR_LOW           $B_PRIOR_MU_STAR_HIGH         \
        --sigma_star      $B_PRIOR_SIGMA_STAR_LOW        $B_PRIOR_SIGMA_STAR_HIGH      \
        --weight_COmass   $B_PRIOR_WEIGHT_COmass_LOW     $B_PRIOR_WEIGHT_COmass_HIGH \
        --z_gal_alpha     $B_PRIOR_Z_GAL_ALPHA_LOW      $B_PRIOR_Z_GAL_ALPHA_HIGH      \
        --z_gal_beta      $B_PRIOR_Z_GAL_BETA_LOW       $B_PRIOR_Z_GAL_BETA_HIGH

else
    echo "Unknown POPULATION: $POPULATION"
    exit 1
fi