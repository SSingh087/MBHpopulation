#!/bin/sh

. EXECUTABLES/config.sh

python prep_TDE_params_detectability.py --events $EVENTS

for MODEL in $MODELS; do
    python check_training_data_structure.py --events $EVENTS --TELESCOPE ZTF  --model $MODEL
    python check_training_data_structure.py --events $EVENTS --TELESCOPE LSST --model $MODEL
done
