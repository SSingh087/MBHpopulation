#!/bin/bash

GALAXIES=100000
POPULATION=$1
INCLUDE_LSST=$2
MBH_MASS_METHODS=$3 # "use_sigma" "use_lgMgal"
NUCLEATION_METHODS=$4 # "hannah" "neumayer"
OCCUPATION_METHODS=$5 # "uniform" "varying"

python prep_pseudo_data.py \
    --GALAXIES $GALAXIES \
    --POPULATION $POPULATION \
    --INCLUDE_LSST $INCLUDE_LSST \
    --mbh_mass_method $MBH_MASS_METHODS \
    --nucleation_method $NUCLEATION_METHODS \
    --occupation_method $OCCUPATION_METHODS \
    --SIGMA_MAX_MASS 1.0

# python plots/plotting_pseudo_data.py \
#     --GALAXIES $GALAXIES \
#     --POPULATION $POPULATION \
#     --mbh_mass_method $MBH_MASS_METHODS \
#     --nucleation_method $NUCLEATION_METHODS
        
