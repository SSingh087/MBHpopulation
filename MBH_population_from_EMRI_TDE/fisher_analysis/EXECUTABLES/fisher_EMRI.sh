#!/bin/sh

. EXECUTABLES/config.sh

python eval_EMRI_fisher.py \
  --OBSERVING_WINDOW "$OBSERVING_WINDOW_EMRI" \
  --PLOT_CORNER False \
  --PLOT_FISHER False \
  --PLOT_COVARIANCE False \
  --PLOT_HISTOGRAMS False