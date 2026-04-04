#!/bin/sh

. EXECUTABLES/config.sh
EVENTS_SCI=$(printf "%.0E" $EVENTS | sed 's/+0//')_events

NUM_NEURONS=$1
LAYERS=$2
LOSS_TYPE=$3
RESCALER_FLAG=$4
YFILE="snrs_dist_log_minmax_rescaled_COMBINED.npy"

python train_model.py \
  --events $EVENTS \
  --x_data_loc /data/wiay/postgrads/shashwat/EMRI_data/PRE_TRAIN_DATA/$EVENTS_SCI/injection_params_dist_REMOVED_COMBINED.npy \
  --y_data_loc /data/wiay/postgrads/shashwat/EMRI_data/PRE_TRAIN_DATA/$EVENTS_SCI/$YFILE \
  --train_cat SNR \
  --num_neurons $NUM_NEURONS \
  --layers $LAYERS \
  --train_test_frac 0.9 \
  --learning_rate 0.001 \
  --n_epochs 1000 \
  --n_batches 500 \
  --update_every 500 \
  --verbose True \
  --outdir MODEL_theta_SNR_config_${LOSS_TYPE}_${RESCALER_FLAG} \
  --device cuda \
  --loss_type $LOSS_TYPE \
  --rescaler_flag $RESCALER_FLAG
