#!/bin/sh

. EXECUTABLES/config.sh

# echo "Preparing Training Data"
# python prep_SNR_training_data.py --events $EVENTS


# # # this has to run after the SNR DATA is ready
# # # because it used data prepared by SNR PREP.
echo "Preparing p0 Data"
python prep_p0_training_data.py --events $EVENTS
