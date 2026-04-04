#!/bin/sh

. EXECUTABLES/config.sh

echo "Calculating PSD for $OBSERVING_WINDOW years"
python calculate_PSD.py --events $EVENTS --OBSERVING_WINDOW $OBSERVING_WINDOW