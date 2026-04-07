#!/bin/sh

# the sequence can be changed but needs to 
# followed in the consequent files.
# Read about the params from here 
# https://bhptoolkit.org/FastEMRIWaveforms/html/user/main.html#generic-kerr-aak-with-5pn-trajectory
# for Gpc to z https://journals.aps.org/prd/abstract/10.1103/PhysRevD.105.123531
# Time, T here is in years so 1 month is 1/12 = 0.083 years.

. EXECUTABLES/config.sh

T_SIGNAL_MIN=0.5
T_SIGNAL_MAX=$OBSERVING_WINDOW_EMRI

python gen_EMRI_params_ex_p0.py \
    --e0 0.1 0.7 \
    --Y0 0.1 0.7 \
    --T_SIGNAL $T_SIGNAL_MIN $T_SIGNAL_MAX \
    --file_name EMRI_params_ex_p0.npy

# echo "=============="
# echo "Calculating p0"
# echo "=============="
# python eval_p0.py --MAX_SIGNAL_DURATION $T_SIGNAL_MAX --OBSERVING_WINDOW $OBSERVING_WINDOW_EMRI\ 
