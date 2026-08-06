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
