#!/bin/sh

. EXECUTABLES/config.sh

SURVEY=$1

if [ "$SURVEY" = "LSST" ]; then
    Z_MIN=$Z_GAL_MIN_LSST
    Z_MAX=$Z_GAL_MAX_LSST
else
    Z_MIN=$Z_GAL_MIN_ZTF
    Z_MAX=$Z_GAL_MAX_ZTF
fi

echo "Generating TDE parameters for $EVENTS events with $SURVEY survey settings..."
echo "Redshift range: $Z_MIN to $Z_MAX"

python gen_TDE_params.py \
    --events $EVENTS \
    --lgMBH 4.0 7.0 \
    --z_gal $Z_MIN $Z_MAX \
    --eta 0.001 0.1 \
    --alpha 0.01 0.1 \
    --beta 1.001 30 \
    --SURVEY $SURVEY
