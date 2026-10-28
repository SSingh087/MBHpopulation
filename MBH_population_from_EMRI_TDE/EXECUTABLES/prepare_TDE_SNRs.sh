#!/bin/sh

. EXECUTABLES/config.sh

MIN_DETECTIONS=2

UNIVERSE_MODEL=$1
SURVEY=$2
MBH_MASS_METHODS=$3 # "use_sigma" "use_lgMgal"
NUCLEATION_METHODS=$4 # "hannah" "neumayer"
OCCUPATION_METHODS=$5 # "uniform" "non_uniform"
TDE_MODEL=$6 # "cooling_envelope" "stream_stream_tde" "tde_analytical" "tde_fallback"
MERGER_MODELS=$7 # "mundy17" "oleary21" "duan24" "baseline"

# == CALCULATING THE SNRs ==

OBSERVING_WINDOW_VAR="OBSERVING_WINDOW_TDE_${SURVEY}"
OBSERVING_WINDOW=$(eval echo \$$OBSERVING_WINDOW_VAR)

BANDS_VAR="BANDS_${SURVEY}"
BANDS=$(eval echo \$$BANDS_VAR)

# Best-performing architecture for THIS (TDE_MODEL, SURVEY) combo -- see
# the TDE_ARCH_NEURONS_<model>_<survey>/TDE_ARCH_LAYERS_<model>_<survey>
# table in config.sh (picked by ROC AUC across all trained architectures).
NEURONS=$(eval echo \$TDE_ARCH_NEURONS_${TDE_MODEL}_${SURVEY})
LAYERS=$(eval echo \$TDE_ARCH_LAYERS_${TDE_MODEL}_${SURVEY})

echo "Calculating SNRs for $GALAXIES galaxies and universe model $UNIVERSE_MODEL for survey $SURVEY, MBH mass method: $MBH_MASS_METHODS, nucleation method: $NUCLEATION_METHODS, occupation method: $OCCUPATION_METHODS, merger model: $MERGER_MODELS, TDE model: $TDE_MODEL (arch: ${NEURONS}x${LAYERS}), observing window: $OBSERVING_WINDOW, bands: $BANDS"

python ./data_generation/calculate_SNR_TDE.py \
  --UNIVERSE_MODEL "$UNIVERSE_MODEL" \
  --GALAXIES $GALAXIES \
  --SURVEY "$SURVEY" \
  --TDE_model "$TDE_MODEL" \
  --Z_GAL_MIN $(eval echo \$Z_GAL_MIN_${SURVEY}) \
  --Z_GAL_MAX $(eval echo \$Z_GAL_MAX_${SURVEY}) \
  --DETECTION_THRESHOLD 0.5 \
  --num_neurons $NEURONS \
  --layers $LAYERS \
  --mbh_mass_method "$MBH_MASS_METHODS" \
  --nucleation_method "$NUCLEATION_METHODS" \
  --occupation_method "$OCCUPATION_METHODS" \
  --merger_model "$MERGER_MODELS"

# Fisher analysis of the detected events runs after the best universe is chosen (see make_fisher_dag.py).
