#!/bin/sh

. EXECUTABLES/config.sh

echo "============================"
echo "Generating SNR for T > T_OBS"
echo "============================"
python get_SNR_T_gt_T_obs.py --events $EVENTS --OBSERVING_WINDOW $OBSERVING_WINDOW \
