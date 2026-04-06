#!/bin/sh

. EXECUTABLES/config.sh

OBSERVING_WINDOW=$OBSERVING_WINDOW_TDE # this is in days

python gen_TDE_params.py \
    --eta 0.0001 0.01 \
    --alpha 0.0001 0.01 \
    --beta 1 100 \
    --OBSERVING_WINDOW $OBSERVING_WINDOW \
    --file_name TDE_params.npy

