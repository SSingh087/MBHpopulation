#!/bin/sh

. EXECUTABLES/config.sh
EVENTS_SCI=$(printf "%.0E" $EVENTS | sed 's/+0//')_events

EVENTS=$(printf "%.0E" "$EVENTS" | sed 's/E+0/E/')

NUM_NEURONS=$1
LAYERS=$2
LOSS_TYPE=$3
RESCALER_FLAG=$4
SURVEY=$5

if [ "$SURVEY" = "LSST" ]; then
    Z_MIN=$Z_GAL_MIN_LSST
    Z_MAX=$Z_GAL_MAX_LSST
else
    Z_MIN=$Z_GAL_MIN_ZTF
    Z_MAX=$Z_GAL_MAX_ZTF
fi

echo "Training TDE detectability model"
echo "  Neurons: $NUM_NEURONS"
echo "  Layers: $LAYERS"
echo "  Loss type: $LOSS_TYPE"
echo "  Rescaler: $RESCALER_FLAG"
echo "  Survey: $SURVEY"
echo "  Events: $EVENTS"

python ../train_model.py \
   --events $EVENTS \
   --x_data_loc ./data/PRE_TRAIN_DATA_TDE/$EVENTS_SCI/TDE_params_scaled_${EVENTS}_${SURVEY}.npy \
   --y_data_loc ./data/PRE_TRAIN_DATA_TDE/$EVENTS_SCI/TDE_detectability_${EVENTS}_${SURVEY}.npy \
   --num_neurons $NUM_NEURONS \
   --layers $LAYERS \
   --train_test_frac 0.8 \
   --learning_rate 0.0001 \
   --n_epochs 500 \
   --n_batches 10 \
   --update_every 10 \
   --verbose True \
   --outdir TDE_DETECT_${LOSS_TYPE}_${RESCALER_FLAG} \
   --device cuda \
   --loss_type $LOSS_TYPE \
   --rescaler_flag $RESCALER_FLAG
