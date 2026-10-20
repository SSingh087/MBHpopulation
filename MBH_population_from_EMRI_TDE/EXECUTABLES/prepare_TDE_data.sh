#!/bin/sh

. EXECUTABLES/config.sh

UNIVERSE_MODEL=$1
SEED=$2
MBH_MASS_METHODS=$3 # "use_sigma" "use_lgMgal"
NUCLEATION_METHODS=$4 # "hannah" "neumayer"
OCCUPATION_METHODS=$5 # "uniform" "non_uniform"
TDE_MODEL=$6 # "cooling_envelope" "stream_stream_tde" "tde_analytical" "tde_fallback"
MERGER_MODELS=$7 # "mundy17" "oleary21" "duan24" "baseline"

echo "==========================================================================================="
echo "Preparing TDE events for $GALAXIES galaxies and universe model $UNIVERSE_MODEL for MBH mass method: $MBH_MASS_METHODS, nucleation method: ${NUCLEATION_METHODS}, occupation method: $OCCUPATION_METHODS, merger model: $MERGER_MODELS, TDE model: $TDE_MODEL"
echo "==========================================================================================="
python ./data_generation/gen_TDE_params.py \
    --UNIVERSE_MODEL "$UNIVERSE_MODEL" \
    --SEED "$SEED" \
    --GALAXIES $GALAXIES \
    --TDE_model "$TDE_MODEL" \
    --eta 0.001 0.1 \
    --alpha 0.01 0.1 \
    --beta 1.001 30 \
    --mbh_mass_method "$MBH_MASS_METHODS" \
    --nucleation_method "$NUCLEATION_METHODS" \
    --occupation_method "$OCCUPATION_METHODS" \
    --merger_model "$MERGER_MODELS"

# NOTE: every event in this universe's source TDE_catalog gets modeled
# by the ONE redback model passed in via $TDE_MODEL -- comparing models
# means the DAG runs this script once per model (see make_data_gen_dag.py),
# not splitting events across models.