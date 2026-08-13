#!/bin/sh

. ./EXECUTABLES/config.sh

MBH_MASS_METHODS=$1 # "use_sigma" "use_lgMgal"
NUCLEATION_METHODS=$2 # "hannah" "neumayer"
OCCUPATION_METHODS=$3 # "uniform" "non_uniform"

echo "Preparing data for inference for $GALAXIES galaxies for MBH mass method: $MBH_MASS_METHODS, nucleation method: $NUCLEATION_METHODS, occupation method: $OCCUPATION_METHODS"
python ./hierarchical_inference/prepare_data_MEDIAN.py\
        --GALAXIES $GALAXIES \
        --mbh_mass_method "$MBH_MASS_METHODS" \
        --nucleation_method "$NUCLEATION_METHODS" \
        --occupation_method "$OCCUPATION_METHODS"