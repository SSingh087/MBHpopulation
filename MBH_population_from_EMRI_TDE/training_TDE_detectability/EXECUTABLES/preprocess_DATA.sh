#!/bin/sh

. EXECUTABLES/config.sh

python prep_TDE_params_detectability.py --events $EVENTS
python check_training_data_structure.py --events $EVENTS --TELESCOPE ZTF
python check_training_data_structure.py --events $EVENTS --TELESCOPE LSST