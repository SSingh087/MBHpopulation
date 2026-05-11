#!/bin/sh

EVENTS=1000000

OBSERVING_WINDOW_TDE_ZTF=400 # this is in days
OBSERVING_WINDOW_TDE_LSST=1000 # this is in days

BANDS_ZTF="ztfg ztfr ztfi" # default for ZTF
BANDS_LSST="lsstu lsstg lsstr lssti lsstz lssty"

Z_GAL_MIN_ZTF=0.0
Z_GAL_MAX_ZTF=1.0

Z_GAL_MIN_LSST=0.0
Z_GAL_MAX_LSST=3.0