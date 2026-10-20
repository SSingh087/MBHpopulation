#!/bin/bash

. EXECUTABLES/config.sh


# args must come first -- SURVEY is needed to build the *_VAR lookups below
SURVEY=$1
START_IDX=$2
END_IDX=$3
BATCH_ID=$4

OBSERVING_WINDOW_VAR="OBSERVING_WINDOW_TDE_${SURVEY}"
OBSERVING_WINDOW=$(eval echo \$$OBSERVING_WINDOW_VAR)
BANDS_VAR="BANDS_${SURVEY}"
BANDS=$(eval echo \$$BANDS_VAR)

echo "[run_fisher_worker] batch=${BATCH_ID} rows=[${START_IDX}:${END_IDX}) survey=${SURVEY}"

echo ${OBSERVING_WINDOW_VAR} = ${OBSERVING_WINDOW}
echo ${BANDS_VAR} = ${BANDS}

python ./fisher_analysis/eval_TDE_fisher.py \
    --GALAXIES "${GALAXIES}" \
    --SURVEY "${SURVEY}" \
    --BANDS ${BANDS} \
    --OBSERVING_WINDOW "${OBSERVING_WINDOW}" \
    --START_IDX "${START_IDX}" \
    --END_IDX "${END_IDX}"\
    --BATCH_ID "${BATCH_ID}"\
    --PLOT_CORNER 1\
    --PLOT_FISHER 1 \
    --PLOT_COVARIANCE 1