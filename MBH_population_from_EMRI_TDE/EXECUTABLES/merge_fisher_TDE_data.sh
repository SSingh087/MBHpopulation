#!/bin/bash
. EXECUTABLES/config.sh

SURVEY=$1

echo "[run_merge] GALAXIES=${GALAXIES} survey=${SURVEY}"

python fisher_analysis/merge_fisher_batch_TDE.py \
    --GALAXIES "${GALAXIES}" \
    --SURVEY "${SURVEY}"