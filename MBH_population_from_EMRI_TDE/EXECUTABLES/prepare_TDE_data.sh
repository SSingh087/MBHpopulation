#!/bin/sh

. EXECUTABLES/config.sh

UNIVERSE_MODEL=$1
SEED=$2
MBH_MASS_METHODS=$3 # "use_sigma" "use_lgMgal"
NUCLEATION_METHODS=$4 # "hannah" "neumayer"

echo "==========================================================================================="
echo "Preparing TDE events for $GALAXIES galaxies and universe model $UNIVERSE_MODEL for MBH mass method: $MBH_MASS_METHODS, nucleation method: ${NUCLEATION_METHODS}"
echo "==========================================================================================="
python ./data_generation/gen_TDE_params.py \
    --UNIVERSE_MODEL "$UNIVERSE_MODEL" \
    --SEED "$SEED" \
    --GALAXIES $GALAXIES \
    --eta 0.001 0.1 \
    --alpha 0.01 0.1 \
    --beta 1.001 30 \
    --mbh_mass_method "$MBH_MASS_METHODS" \
    --nucleation_method "$NUCLEATION_METHODS" 