#!/bin/sh

. ./EXECUTABLES/config.sh

EVENTS=$1
SOURCE=$2

EVENTS_SCI=$(printf "%.0E" $EVENTS | sed 's/+0//')_events

echo "Doing inference"
python hierarchical_inference.py\
        --lambda_M $B_PRIOR_LAMBDA_M_LOW $B_PRIOR_LAMBDA_M_HIGH \
        --lambda_mu $B_PRIOR_LAMBDA_MU_LOW $B_PRIOR_LAMBDA_MU_HIGH \
        --mu_a $B_PRIOR_MU_A_LOW $B_PRIOR_MU_A_HIGH --sigma_a $B_PRIOR_SIGMA_A_LOW $B_PRIOR_SIGMA_A_HIGH \
        --source $SOURCE \
        --work_dir ./data/INFERENCE_DATA/_pop_test/$EVENTS_SCI/

# python compare_posterior_1D_EMRI_TDE.py
