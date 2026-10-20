#!/bin/bash
# Run from anywhere. Uses repo cwd because sources_num.py-style imports are relative to it.
T="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(dirname "$T")"
PY=python
N=${1:-20000}; SEED=${2:-42}
cd "$REPO"
for M in baseline mundy17 mundy17_n5e4 oleary_9.7 oleary_10.3 oleary_11.0; do
  $PY $T/scripts/run_impact.py --UNIVERSE_MODEL 0 --GALAXIES $N --Z_MAX 8.0 --T_OBS_EMRI 6 --T_OBS_TDE 17.25 \
     --mbh_mass_method use_sigma --nucleation_method hannah --occupation_method uniform --SEED $SEED \
     --MERGER_MODEL $M --OUT $T/outputs/impact_${M}_N${N}_s${SEED}.json > $T/outputs/log_${M}_N${N}_s${SEED}.txt 2>&1
  echo "$M done (exit $?)"
done
