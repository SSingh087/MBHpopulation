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

echo "==============================================================="
echo "Generating EMRI and TDE events for $GALAXIES galaxies"
echo "==============================================================="

python gen_events.py --GALAXIES $GALAXIES --OBSERVING_WINDOW $OBSERVING_WINDOW_EMRI