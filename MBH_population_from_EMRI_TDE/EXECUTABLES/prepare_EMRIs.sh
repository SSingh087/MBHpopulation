#!/bin/sh

# the sequence can be changed but needs to 
# followed in the consequent files.
# Read about the params from here 
# https://bhptoolkit.org/FastEMRIWaveforms/html/user/main.html#generic-kerr-aak-with-5pn-trajectory
# for Gpc to z https://journals.aps.org/prd/abstract/10.1103/PhysRevD.105.123531
# Time, T here is in years so 1 month is 1/12 = 0.083 years.

. EXECUTABLES/config.sh

T_SIGNAL_MIN=1.0
T_SIGNAL_MAX=$T_OBS_EMRI
UNIVERSE_MODEL=$1
SEED=$2
MBH_MASS_METHODS=$3 # "use_sigma" "use_lgMgal"
NUCLEATION_METHODS=$4 # "hannah" "neumayer"

echo "==========================================================================================="
echo "Preparing EMRI events for $GALAXIES galaxies and universe model $UNIVERSE_MODEL for MBH mass method: $MBH_MASS_METHODS, nucleation method: $NUCLEATION_METHODS"
echo "==========================================================================================="

python ./data_generation/gen_EMRI_params.py \
    --UNIVERSE_MODEL "$UNIVERSE_MODEL" \
    --GALAXIES $GALAXIES\
    --Y0 0.1 0.7 \
    --T_SIGNAL $T_SIGNAL_MIN $T_SIGNAL_MAX \
    --mbh_mass_method "$MBH_MASS_METHODS" \
    --nucleation_method "$NUCLEATION_METHODS" \
    --SEED "$SEED" \


# == CALCULATING THE SNRs ==

echo "Calculating SNRs for EMRI events"

python ./data_generation/calculate_SNR_EMRI_pseudo.py \
    --UNIVERSE_MODEL "$UNIVERSE_MODEL" \
    --GALAXIES $GALAXIES \
    --mbh_mass_method "$MBH_MASS_METHODS" \
    --nucleation_method "$NUCLEATION_METHODS" \

# # == Fisher matrix calculation ==

# echo "Calculating Fisher matrices for EMRI events"

# python ./fisher_analysis/eval_EMRI_fisher.py \
#   --GALAXIES $GALAXIES \
#   --OBSERVING_WINDOW $T_OBS_EMRI \
#   # required only when plotting. Cannot be set to false
#   # just comment because of the action="store_true" in the argument parser.
#   # --PLOT_CORNER True \
#   # --PLOT_FISHER True \
#   # --PLOT_COVARIANCE True \
#   # --PLOT_HISTOGRAMS True 