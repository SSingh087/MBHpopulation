#!/bin/sh
. EXECUTABLES/config.sh

MIN_DETECTIONS=2
SURVEY=$1
CHUNK_ID=$2
N_CHUNKS=$3

if [ "$SURVEY" = "LSST" ]; then
    Z_MIN=$Z_GAL_MIN_LSST
    Z_MAX=$Z_GAL_MAX_LSST
else
    Z_MIN=$Z_GAL_MIN_ZTF
    Z_MAX=$Z_GAL_MAX_ZTF
fi

OBSERVING_WINDOW=$(eval echo \$OBSERVING_WINDOW_TDE_${SURVEY})
BANDS=$(eval echo \$BANDS_${SURVEY})

python calculate_SNR_TDE_PARALLEL.py \
    --events            $EVENTS \
    --OBSERVING_WINDOW  $OBSERVING_WINDOW \
    --MIN_DETECTIONS    $MIN_DETECTIONS \
    --SURVEY            "$SURVEY" \
    --BANDS             $BANDS \
    --Z_MIN             $Z_MIN \
    --Z_MAX             $Z_MAX \
    --chunk_id          $CHUNK_ID \
    --n_chunks          $N_CHUNKS