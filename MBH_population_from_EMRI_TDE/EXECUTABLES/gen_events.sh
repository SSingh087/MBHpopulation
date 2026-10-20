#!/bin/bash

. ./EXECUTABLES/config.sh

T_SIGNAL_MIN=0.5
T_SIGNAL_MAX=$OBSERVING_WINDOW_EMRI

UNIVERSE_MODEL=$1
SEED=$2
MBH_MASS_METHODS=$3 # "use_sigma" "use_lgMgal"
NUCLEATION_METHODS=$4 # "hannah" "neumayer"
OCCUPATION_METHODS=$5 # "uniform" "varying"
MERGER_MODELS=$6 # "mundy17" "oleary21" "duan24" "baseline"


echo "==========================================================================================="
echo "Generating EMRI and TDE events for $GALAXIES galaxies and universe model $UNIVERSE_MODEL for MBH mass method: $MBH_MASS_METHODS, nucleation method: $NUCLEATION_METHODS, occupation method: $OCCUPATION_METHODS, merger model: $MERGER_MODELS"
echo "==========================================================================================="

python ./data_generation/sources_num.py \
    --UNIVERSE_MODEL "$UNIVERSE_MODEL" \
    --GALAXIES "$GALAXIES" \
    --Z_MAX "$Z_MAX" \
    --T_OBS_EMRI "$T_OBS_EMRI" \
    --T_OBS_TDE "$T_OBS_TDE" \
    --mbh_mass_method "$MBH_MASS_METHODS" \
    --nucleation_method "$NUCLEATION_METHODS" \
    --occupation_method "$OCCUPATION_METHODS" \
    --merger_model "$MERGER_MODELS" \
    --SEED "$SEED" \
