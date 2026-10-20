#!/usr/bin/env bash

. EXECUTABLES/config.sh

N_UNIVERSES=$1
MERGER_MODELS=$2 # "mundy17" "oleary21" "duan24" "baseline"
echo "=== report_universes: scanning ${N_UNIVERSES} universes per scenario, merger model ${MERGER_MODELS} ==="

python report_universes.py \
    --GALAXIES "$GALAXIES" \
    --N_UNIVERSES "$N_UNIVERSES" \
    --merger_model "$MERGER_MODELS"