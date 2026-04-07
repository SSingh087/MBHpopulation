#!/bin/sh

. EXECUTABLES/config.sh

python gen_TDE_params.py \
    --eta 0.0001 0.01 \
    --alpha 0.0001 0.01 \
    --beta 1 100 \
    --file_name TDE_params.npy

