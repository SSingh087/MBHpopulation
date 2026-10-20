#!/bin/bash
# One chunk of selection-function training data: gen_selection_data.sh <A|B> <EMRI|TDE_ZTF|TDE_LSST> <chunk>
. ./EXECUTABLES/config.sh

POPULATION=$1
CHANNEL=$2
CHUNK=$3

python ./selection_function/gen_selection_data.py \
    --POPULATION "$POPULATION" \
    --CHANNEL "$CHANNEL" \
    --CHUNK "$CHUNK" \
    --T_OBS_EMRI "$T_OBS_EMRI" \
    --MODEL_PATH "$EMRI_SNR_MODEL_PATH" \
    --SCALING_PATH "$EMRI_SNR_SCALING_PATH" \
    --SNR_THRESHOLD "$EMRI_SNR_THRESHOLD"
