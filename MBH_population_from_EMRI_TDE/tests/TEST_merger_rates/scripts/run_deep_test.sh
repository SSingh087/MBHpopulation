#!/bin/bash
# Deep paired test: for each replicate seed, build ONE galaxy population and evaluate every merger model /
# sensitivity variant on it (common random numbers). Runs replicates one after another (low memory).
#
# Usage (from anywhere, ideally in a normal terminal, not the VS Code integrated one):
#   cd <repo_root>
#   setsid nohup tests/TEST_merger_rates/scripts/run_deep_test.sh 20000 "1 2 3 4 5 6" > tests/TEST_merger_rates/outputs/deep_run.log 2>&1 < /dev/null &
# Args: $1 = GALAXIES (default 20000; 100000 is the repo default but slower/more memory)
#       $2 = quoted list of seeds (default "1 2 3")
# Progress:  tail -f tests/TEST_merger_rates/outputs/deep_run.log     (or the per-seed logs paired_N*_s*.log)
# Results:   tests/TEST_merger_rates/outputs/paired_N<N>_s<seed>.json  (written after every variant)
#            tests/TEST_merger_rates/outputs/paired_N<N>_s<seed>.npz   (per-galaxy arrays, written at the end)

T="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(dirname "$T")"
PY=python
N=${1:-20000}
SEEDS=${2:-"1 2 3"}

cd "$REPO" || exit 1
mkdir -p "$T/outputs"
export PYTHONWARNINGS=ignore
for S in $SEEDS; do
  echo "=== seed $S, N=$N, start $(date) ==="
  $PY "$T/scripts/run_paired.py" --UNIVERSE_MODEL 0 --GALAXIES "$N" --Z_MAX 8.0 \
      --T_OBS_EMRI 6 --T_OBS_TDE 17.25 \
      --mbh_mass_method use_sigma --nucleation_method hannah --occupation_method uniform \
      --SEED "$S" \
      --DUMP "$T/outputs/paired_N${N}_s${S}.npz" --OUT "$T/outputs/paired_N${N}_s${S}.json" \
      > "$T/outputs/paired_N${N}_s${S}.log" 2>&1
  echo "=== seed $S finished with exit code $? at $(date) ==="
done
