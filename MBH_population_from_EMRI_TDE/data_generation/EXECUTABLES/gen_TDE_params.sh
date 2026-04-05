#!/bin/sh

. EXECUTABLES/config.sh

OBSERVING_WINDOW=$OBSERVING_WINDOW_TDE # this is in days

python gen_TDE_params.py \
    --eta 0.1 0.7 \
    --alpha 0.1 0.7 \
    --beta 0.1 0.7 \
    --OBSERVING_WINDOW $OBSERVING_WINDOW \
    --file_name TDE_params.npy

