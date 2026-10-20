#!/bin/bash
. EXECUTABLES/config.sh

SURVEY=$1
CHUNK=$2
N_CHUNKS=$3
N_EVENTS=${4:-200}

OBSERVING_WINDOW_VAR="OBSERVING_WINDOW_TDE_${SURVEY}"
OBSERVING_WINDOW=$(eval echo \$$OBSERVING_WINDOW_VAR)
BANDS_VAR="BANDS_${SURVEY}"
BANDS=$(eval echo \$$BANDS_VAR)

echo "[calibrate_sigma_TDE] survey=${SURVEY} chunk=${CHUNK}/${N_CHUNKS} N_EVENTS=${N_EVENTS} window=${OBSERVING_WINDOW} bands=${BANDS}"

python ./fisher_analysis/calibrate_sigma_TDE_fallback.py \
    --GALAXIES "${GALAXIES}" \
    --SURVEY "${SURVEY}" \
    --BANDS ${BANDS} \
    --OBSERVING_WINDOW "${OBSERVING_WINDOW}" \
    --N_EVENTS "${N_EVENTS}" \
    --N_CHUNKS "${N_CHUNKS}" \
    --CHUNK "${CHUNK}"
