#!/bin/sh

. ./EXECUTABLES/config.sh

SOURCE=$1

EVENTS_SCI=$(printf "%.0E" $GALAXIES | sed 's/+0//')_galaxies_${SOURCE}

# python prepare_data.py 

echo "Doing inference"
python hierarchical_inference.py\
        --lambda_M $B_PRIOR_LAMBDA_M_LOW $B_PRIOR_LAMBDA_M_HIGH \
        --mu_a $B_PRIOR_MU_A_LOW $B_PRIOR_MU_A_HIGH --sigma_a $B_PRIOR_SIGMA_A_LOW $B_PRIOR_SIGMA_A_HIGH \
        --source $SOURCE \
        # --work_dir /$GALAXIES/

# python compare_posterior_1D_EMRI_TDE.py
