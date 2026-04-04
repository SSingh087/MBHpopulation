#!/bin/sh

. EXECUTABLES/config.sh
EVENTS_SCI=$(printf "%.0E" $EVENTS | sed 's/+0//')_events

NUM_NEURONS=$1
LAYERS=$2
LOSS_TYPE=$3
RESCALER_FLAG=$4

echo "Training prams-> p0"
python train_model.py \
   --events $EVENTS \
   --x_data_loc /data/wiay/postgrads/shashwat/EMRI_data/PRE_TRAIN_DATA/$EVENTS_SCI/params_for_p0_ALL_COMBINED.npy \
   --y_data_loc /data/wiay/postgrads/shashwat/EMRI_data/PRE_TRAIN_DATA/$EVENTS_SCI/p0_minmax_rescaled_ALL_COMBINED.npy \
   --train_cat p0 \
   --num_neurons $NUM_NEURONS \
   --layers $LAYERS \
   --train_test_frac 0.8 \
   --learning_rate .0001 \
   --n_epochs 10000 \
   --n_batches 500 \
   --update_every 500 \
   --verbose True \
   --outdir MODEL_traj_p0_config_${LOSS_TYPE}_${RESCALER_FLAG} \
   --device cuda \
  --loss_type $LOSS_TYPE \
  --rescaler_flag $RESCALER_FLAG
