#!/bin/sh
. EXECUTABLES/config.sh

MIN_DETECTIONS=2
SURVEY=$1
MODEL=$2
CHUNK_ID=$3
N_CHUNKS=$4

if [ "$SURVEY" = "LSST" ]; then
    Z_MIN=$Z_GAL_MIN_LSST
    Z_MAX=$Z_GAL_MAX_LSST
else
    Z_MIN=$Z_GAL_MIN_ZTF
    Z_MAX=$Z_GAL_MAX_ZTF
fi

LENGTH_OF_TDE=$(eval echo \$LENGTH_OF_TDE_${SURVEY})
BANDS=$(eval echo \$BANDS_${SURVEY})

python calculate_SNR_TDE_PARALLEL.py \
    --events            $EVENTS \
    --LENGTH_OF_TDE     $LENGTH_OF_TDE \
    --MIN_DETECTIONS    $MIN_DETECTIONS \
    --SURVEY            "$SURVEY" \
    --BANDS             $BANDS \
    --Z_MIN             $Z_MIN \
    --Z_MAX             $Z_MAX \
    --model             $MODEL \
    --chunk_id          $CHUNK_ID \
    --n_chunks          $N_CHUNKS
