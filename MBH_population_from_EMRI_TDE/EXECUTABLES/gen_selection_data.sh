#!/bin/bash
# One chunk of selection-function training data:
#   gen_selection_data.sh <A|B> <EMRI|TDE_ZTF|TDE_LSST> <chunk> <tde_model>
. ./EXECUTABLES/config.sh

POPULATION=$1
CHANNEL=$2
CHUNK=$3
TDE_MODEL=$4

SURVEY=${CHANNEL#TDE_}
NEURONS_VAR="TDE_ARCH_NEURONS_${TDE_MODEL}_${SURVEY}"
LAYERS_VAR="TDE_ARCH_LAYERS_${TDE_MODEL}_${SURVEY}"
ZMAX_VAR="Z_GAL_MAX_${SURVEY}"

python ./selection_function/gen_selection_data.py \
    --POPULATION "$POPULATION" \
    --CHANNEL "$CHANNEL" \
    --CHUNK "$CHUNK" \
    --T_OBS_EMRI "$T_OBS_EMRI" \
    --MODEL_PATH "$EMRI_SNR_MODEL_PATH" \
    --SCALING_PATH "$EMRI_SNR_SCALING_PATH" \
    --SNR_THRESHOLD "$EMRI_SNR_THRESHOLD" \
    --TDE_MODEL "$TDE_MODEL" \
    --num_neurons "$(eval echo \$$NEURONS_VAR)" \
    --layers "$(eval echo \$$LAYERS_VAR)" \
    --Z_GAL_MAX "$(eval echo \$$ZMAX_VAR)"
