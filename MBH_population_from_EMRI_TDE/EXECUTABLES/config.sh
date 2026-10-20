#!/bin/sh

export GALAXIES=100000
export UNIVERSE_MODEL=0
export Z_MAX=8.0
export SEED=42
export IS_CLUSTERING=0

# All this is in years
T_OBS_EMRI=6       
T_OBS_TDE=17.25
T_OBS_TDE_ZTF=8.75  # 1 Mar 2018 (MJD 58178) → Dec 2026 (MJD 61396)
T_OBS_TDE_LSST=9.84 # Jun 2025 (MJD 60827) → Jun 2035 (MJD 64423)

BANDS_ZTF="ztfg ztfr ztfi" # default for ZTF
BANDS_LSST="lsstu lsstg lsstr lssti lsstz lssty"

Z_GAL_MIN_ZTF=0.0
Z_GAL_MAX_ZTF=0.5

Z_GAL_MIN_LSST=0.0
Z_GAL_MAX_LSST=4.0

OBSERVING_WINDOW_TDE_ZTF=500 # this is in days
OBSERVING_WINDOW_TDE_LSST=700 # this is in days

# Best-performing (neurons, layers) architecture per (redback TDE model,
# survey), picked by ROC AUC over all 9 architectures actually trained by
# training_TDE_detectability/train_TDE_detectability.sub (checked against
# each combo's predictions.npy/ground_truth.npy under
# PRE_TRAIN_DATA_TDE/1E6_events/TDE_DETECT_BCE_no_rescaler_<model>_<n>_<l>_10_<survey>/).
# AUCs were tightly clustered (~0.977-0.991) across architectures per combo,
# so this is a modest-margin pick, not a huge outlier -- update this table if
# the models get retrained.
TDE_ARCH_NEURONS_cooling_envelope_ZTF=512
TDE_ARCH_LAYERS_cooling_envelope_ZTF=4
TDE_ARCH_NEURONS_cooling_envelope_LSST=128
TDE_ARCH_LAYERS_cooling_envelope_LSST=4

TDE_ARCH_NEURONS_stream_stream_tde_ZTF=256
TDE_ARCH_LAYERS_stream_stream_tde_ZTF=3
TDE_ARCH_NEURONS_stream_stream_tde_LSST=512
TDE_ARCH_LAYERS_stream_stream_tde_LSST=4

TDE_ARCH_NEURONS_tde_analytical_ZTF=512
TDE_ARCH_LAYERS_tde_analytical_ZTF=4
TDE_ARCH_NEURONS_tde_analytical_LSST=512
TDE_ARCH_LAYERS_tde_analytical_LSST=4

TDE_ARCH_NEURONS_tde_fallback_ZTF=512
TDE_ARCH_LAYERS_tde_fallback_ZTF=4
TDE_ARCH_NEURONS_tde_fallback_LSST=128
TDE_ARCH_LAYERS_tde_fallback_LSST=4


# Trained EMRI SNR network (training_EMRI_SNR/) and the scaling saved with its training set
EMRI_SNR_MODEL_PATH=./data/EMRI/PRE_TRAIN_DATA/1E6_events/MODEL_theta_SNR_best/best_final_model/model/model.pth
EMRI_SNR_SCALING_PATH=./data/EMRI/PRE_TRAIN_DATA/1E6_events/snr_scaling.json
EMRI_SNR_THRESHOLD=20
