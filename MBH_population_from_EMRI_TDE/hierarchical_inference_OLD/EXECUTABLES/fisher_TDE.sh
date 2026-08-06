#!/bin/sh


. ./EXECUTABLES/config.sh

EVENTS=$1

EVENTS_SCI=$(printf "%.0E" $EVENTS | sed 's/+0//')_events

echo "================================"
echo "Evaluating TDE Fisher matrices"
echo "================================"
python calculate_fisher_TDE.py --events $EVENTS --work_dir ./data/INFERENCE_DATA/_pop_test/$EVENTS_SCI/

# echo "================================"
# echo "Evaluating Fisher matrices AGAIN"
# echo "================================"
# python calculate_fisher_noise.py --work_dir ./INFERENCE_DATA/_pop_A/$EVENTS_SCI/