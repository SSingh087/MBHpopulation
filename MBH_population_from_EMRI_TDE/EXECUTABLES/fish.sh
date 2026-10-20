#!/bin/bash
# One Fisher batch: fish.sh <EMRI|TDE_ZTF|TDE_LSST> <scenario> <start_idx> <end_idx> <batch_id>
. EXECUTABLES/config.sh

SOURCE=$1
SCENARIO=$2
START_IDX=$3
END_IDX=$4
BATCH_ID=$5

if [ "$SOURCE" = "EMRI" ]; then
    python ./fisher_analysis/eval_EMRI_fisher.py \
        --GALAXIES "${GALAXIES}" \
        --SCENARIO "${SCENARIO}" \
        --OBSERVING_WINDOW "${T_OBS_EMRI}" \
        --PSD_PATH "${EMRI_PSD_PATH}" \
        --START_IDX "${START_IDX}" --END_IDX "${END_IDX}" --BATCH_ID "${BATCH_ID}"
else
    SURVEY=${SOURCE#TDE_}
    OBSERVING_WINDOW_VAR="OBSERVING_WINDOW_TDE_${SURVEY}"
    BANDS_VAR="BANDS_${SURVEY}"
    python ./fisher_analysis/eval_TDE_fisher.py \
        --GALAXIES "${GALAXIES}" \
        --SURVEY "${SURVEY}" \
        --SCENARIO "${SCENARIO}" \
        --BANDS $(eval echo \$$BANDS_VAR) \
        --OBSERVING_WINDOW "$(eval echo \$$OBSERVING_WINDOW_VAR)" \
        --START_IDX "${START_IDX}" --END_IDX "${END_IDX}" --BATCH_ID "${BATCH_ID}"
fi
