#!/bin/sh
. ./EXECUTABLES/config.sh
. ./EXECUTABLES/priors.sh

SOURCE=$1
POPULATION=$2
SURVEY=${3:-'None'}

echo "Doing inference for source $SOURCE and survey $SURVEY assuming POPULATION $POPULATION"

if [ "$POPULATION" = "A" ]; then
    python ./hierarchical_inference/hierarchical_inference_A.py \
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
        --mu_sBH          $A_PRIOR_MU_sBH_LOW           $A_PRIOR_MU_sBH_HIGH          \
        --sigma_sBH       $A_PRIOR_SIGMA_sBH_LOW        $A_PRIOR_SIGMA_sBH_HIGH       \
        --mu_star         $A_PRIOR_MU_STAR_LOW          $A_PRIOR_MU_STAR_HIGH         \
        --sigma_star      $A_PRIOR_SIGMA_STAR_LOW       $A_PRIOR_SIGMA_STAR_HIGH      \
        --weight_COmass   $A_PRIOR_WEIGHT_COmass_LOW    $A_PRIOR_WEIGHT_COmass_HIGH

elif [ "$POPULATION" = "B" ]; then
    python ./hierarchical_inference/hierarchical_inference_B.py \
        --GALAXIES $GALAXIES \
        --SOURCE $SOURCE \
        --SURVEY $SURVEY \
        --lambda_M        $B_PRIOR_LAMBDA_M_LOW         $B_PRIOR_LAMBDA_M_HIGH        \
        --mu_1_MBHSpin    $B_PRIOR_MU_1_MBHSpin_LOW     $B_PRIOR_MU_1_MBHSpin_HIGH    \
        --sigma_1_MBHSpin $B_PRIOR_SIGMA_1_MBHSpin_LOW  $B_PRIOR_SIGMA_1_MBHSpin_HIGH \
        --mu_2_MBHSpin    $B_PRIOR_MU_2_MBHSpin_LOW     $B_PRIOR_MU_2_MBHSpin_HIGH    \
        --sigma_2_MBHSpin $B_PRIOR_SIGMA_2_MBHSpin_LOW  $B_PRIOR_SIGMA_2_MBHSpin_HIGH \
        --weight_MBHSpin  $B_PRIOR_WEIGHT_MBHSpin_LOW   $B_PRIOR_WEIGHT_MBHSpin_HIGH  \
        --mu_sBH          $B_PRIOR_MU_sBH_LOW            $B_PRIOR_MU_sBH_HIGH          \
        --sigma_sBH       $B_PRIOR_SIGMA_sBH_LOW         $B_PRIOR_SIGMA_sBH_HIGH       \
        --mu_star         $B_PRIOR_MU_STAR_LOW           $B_PRIOR_MU_STAR_HIGH         \
        --sigma_star      $B_PRIOR_SIGMA_STAR_LOW        $B_PRIOR_SIGMA_STAR_HIGH      \
        --weight_COmass   $B_PRIOR_WEIGHT_COmass_LOW     $B_PRIOR_WEIGHT_COmass_HIGH

else
    echo "Unknown POPULATION: $POPULATION"
    exit 1
fi