#!/usr/bin/env bash

. EXECUTABLES/config.sh

N_UNIVERSES=$1
MERGER_MODELS=$2 # "mundy17" "oleary21" "duan24" "baseline"
echo "=== compare_TDE_detection: comparing ${N_UNIVERSES} universes per (scenario, TDE model), merger model ${MERGER_MODELS} ==="

python data_generation/compare_TDE_detection.py \
    --GALAXIES "$GALAXIES" \
    --N_UNIVERSES "$N_UNIVERSES" \
    --merger_model "$MERGER_MODELS"
