#!/bin/sh

export GALAXIES=100000
export UNIVERSE_MODEL=0
export Z_MAX=8.0
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
MBH_MASS_METHODS=$4 # "use_sigma" "use_lgMgal"
NUCLEATION_METHODS=$5 # "hannah" "neumayer"
OCCUPATION_METHODS=$6 # "uniform" "varying"


# ============================================================
# PRIORS for population A hyper-parameters
# Cross-ref: FixedLimitSchechterFunction, DoubleBetaDistribution, DoubleGaussianDistribution
# ============================================================

# MBH mass  (FixedLimitSchechterFunction)
A_PRIOR_XC_M_LOW=4.0
A_PRIOR_XC_M_HIGH=7.0
A_PRIOR_LAM_SCHECHTER_LOW=6.5
A_PRIOR_LAM_SCHECHTER_HIGH=22.0   # was 15.0 -- true values go up to 20 (H_sigma/N_sigma)

# MBH spin  (DoubleBetaDistribution → alpha_1, alpha_2, beta_1, beta_2, weight)
A_PRIOR_ALPHA_1_MBHSpin_LOW=0.5
A_PRIOR_ALPHA_1_MBHSpin_HIGH=5.0
A_PRIOR_ALPHA_2_MBHSpin_LOW=1.0   # was 5.1 -- true values are 2.8-3.4, entirely below the old floor
A_PRIOR_ALPHA_2_MBHSpin_HIGH=15.0
A_PRIOR_BETA_1_MBHSpin_LOW=0.5
A_PRIOR_BETA_1_MBHSpin_HIGH=15.0
A_PRIOR_BETA_2_MBHSpin_LOW=0.5
A_PRIOR_BETA_2_MBHSpin_HIGH=15.0
A_PRIOR_WEIGHT_MBHSpin_LOW=0.001
A_PRIOR_WEIGHT_MBHSpin_HIGH=0.999

# e_pl  (DoubleBetaDistribution → alpha_1, alpha_2, beta_1, beta_2, weight)
A_PRIOR_ALPHA_1_e_pl_LOW=1.001
A_PRIOR_ALPHA_1_e_pl_HIGH=5.0    # was 2.999 -- true values are 3.4-4.0, entirely above the old ceiling
A_PRIOR_ALPHA_2_e_pl_LOW=2.5     # was 3.001 -- true min is 3.0, right at/below the old floor
A_PRIOR_ALPHA_2_e_pl_HIGH=7.0    # was 5.0 -- true max is 6.0
A_PRIOR_BETA_1_e_pl_LOW=6.5
A_PRIOR_BETA_1_e_pl_HIGH=16.0    # was 8.0 -- true values are 12.0-14.0, entirely above the old ceiling
A_PRIOR_BETA_2_e_pl_LOW=4.5
A_PRIOR_BETA_2_e_pl_HIGH=9.0     # was 6.5 -- true values are 7.0-7.5, entirely above the old ceiling
A_PRIOR_WEIGHT_e_pl_LOW=0.001
A_PRIOR_WEIGHT_e_pl_HIGH=0.999

# Z_GAL DIST (BETA DIST)
A_PRIOR_Z_GAL_LAM_LOW=-2.0
A_PRIOR_Z_GAL_LAM_HIGH=-1.01


# ============================================================
# PRIORS for population B hyper-parameters
# Cross-ref: FixedLimitPowerLaw, DoubleGaussianDistribution, DoubleGaussianDistribution
# ============================================================

# MBH mass  (FixedLimitsPowerLaw → lam)
B_PRIOR_LAMBDA_M_LOW=-3.0
B_PRIOR_LAMBDA_M_HIGH=3.0        # was -1.01 -- true values include +2.0/+2.5 (H_sigma/N_sigma),
                                  # entirely outside the old all-negative range.
                                  # NOTE: this range now spans lam=-1.0, which is a genuine
                                  # singularity for FixedLimitsPowerLaw (division by lam+1 in
                                  # its normalization -- see the earlier z_gal NaN bug). A
                                  # continuous uniform prior landing exactly on -1.0 is
                                  # measure-zero so this won't crash in practice, but if nessai
                                  # ever proposes a point numerically very close to -1.0 you may
                                  # see isolated NaN likelihoods -- worth a nan-guard in
                                  # dist_lgMBH.pdf/draw_samples if that shows up in practice.

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
B_PRIOR_MU_1_e_pl_LOW=0.1
B_PRIOR_MU_1_e_pl_HIGH=0.25      # was 0.2 -- true max is 0.22, right at/above the old ceiling
B_PRIOR_SIGMA_1_e_pl_LOW=0.01
B_PRIOR_SIGMA_1_e_pl_HIGH=0.09
B_PRIOR_MU_2_e_pl_LOW=0.2
B_PRIOR_MU_2_e_pl_HIGH=0.4
B_PRIOR_SIGMA_2_e_pl_LOW=0.02     # was 0.1 -- true values are 0.04-0.05, entirely below the old floor
B_PRIOR_SIGMA_2_e_pl_HIGH=0.2
B_PRIOR_WEIGHT_e_pl_LOW=0.001
B_PRIOR_WEIGHT_e_pl_HIGH=0.999

# Z_GAL DIST (BETA DIST)
B_PRIOR_Z_GAL_LAM_LOW=-2.0
B_PRIOR_Z_GAL_LAM_HIGH=-1.01

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

echo "Doing inference for source $SOURCE and survey $SURVEY assuming POPULATION $POPULATION ($MBH_MASS_METHODS/$NUCLEATION_METHODS/$OCCUPATION_METHODS) with $GALAXIES galaxies"

if [ "$POPULATION" = "A" ]; then
    python hierarchical_inference_A.py \
        --GALAXIES $GALAXIES \
        --SOURCE $SOURCE \
        --SURVEY $SURVEY \
        --mbh_mass_method $MBH_MASS_METHODS \
        --nucleation_method $NUCLEATION_METHODS \
        --occupation_method $OCCUPATION_METHODS \
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
        --lam_z_gal       $A_PRIOR_Z_GAL_LAM_LOW        $A_PRIOR_Z_GAL_LAM_HIGH

elif [ "$POPULATION" = "B" ]; then
    python hierarchical_inference_B.py \
        --GALAXIES $GALAXIES \
        --SOURCE $SOURCE \
        --SURVEY $SURVEY \
        --mbh_mass_method $MBH_MASS_METHODS \
        --nucleation_method $NUCLEATION_METHODS \
        --occupation_method $OCCUPATION_METHODS \
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
        --lam_z_gal       $B_PRIOR_Z_GAL_LAM_LOW         $B_PRIOR_Z_GAL_LAM_HIGH

else
    echo "Unknown POPULATION: $POPULATION"
    exit 1
fi