#!/bin/sh

. EXECUTABLES/config.sh

# SURVEY can be from ZTF LSST

SURVEY=$1

OBSERVING_WINDOW_VAR="OBSERVING_WINDOW_TDE_${SURVEY}"
OBSERVING_WINDOW=$(eval echo \$$OBSERVING_WINDOW_VAR)

BANDS_VAR="BANDS_${SURVEY}"
BANDS=$(eval echo \$$BANDS_VAR)

# echo "Running TDE Fisher analysis once for survey: $SURVEY"

# python eval_TDE_fisher.py \
#   --OBSERVING_WINDOW "$OBSERVING_WINDOW" \
#   --SURVEY "$SURVEY" \
#   --BANDS $BANDS

echo "Running TDE Fisher analysis AGAIN for survey: $SURVEY"

python eval_TDE_fisher_noise.py \
  --OBSERVING_WINDOW "$OBSERVING_WINDOW" \
  --SURVEY "$SURVEY" \
  --BANDS $BANDS