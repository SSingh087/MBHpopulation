#!/bin/bash
# One replicate of the paired merger-model test. Submit from the REPO ROOT (see paired.sub).
# Uses whatever `python` is active via getenv=true (same convention as EXECUTABLES/gen_events.sh),
# so activate the python environment before condor_submit.
. ./EXECUTABLES/config.sh      # GALAXIES, Z_MAX, T_OBS_EMRI, T_OBS_TDE, ...

SEED=$1
N=${2:-$GALAXIES}
OUT=./tests/TEST_merger_rates/outputs
mkdir -p $OUT

echo "==========================================================================================="
echo "Paired merger-model test: seed $SEED, $N galaxies"
echo "==========================================================================================="

python ./tests/TEST_merger_rates/scripts/run_paired.py \
    --UNIVERSE_MODEL 0 \
    --GALAXIES "$N" \
    --Z_MAX "$Z_MAX" \
    --T_OBS_EMRI "$T_OBS_EMRI" \
    --T_OBS_TDE "$T_OBS_TDE" \
    --mbh_mass_method use_sigma \
    --nucleation_method hannah \
    --occupation_method uniform \
    --SEED "$SEED" \
    --DUMP $OUT/paired_N${N}_s${SEED}.npz \
    --OUT  $OUT/paired_N${N}_s${SEED}.json
