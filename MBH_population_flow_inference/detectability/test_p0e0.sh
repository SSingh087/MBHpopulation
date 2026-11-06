#!/bin/bash
# Head-node test of the p0/e0 emulator pipeline (small, ~10-15 min). Usage: bash detectability/test_p0e0.sh
set -e
cd "$(dirname "$0")"
PY=/home/2673888s/.conda/envs/redback_few/bin/python
export PYTHONPATH=/home/2673888s/poplar MPLBACKEND=Agg
T=/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference/simulations/p0e0_test
rm -rf $T; mkdir -p $T
echo "=== 1. timing / acceptance of single trajectories (16 workers, 400 trajectories) ==="
$PY time_traj.py --n 400 --workers 16 2>&1 | grep -v Warn
echo "=== 2. eight small data chunks (<=150 rows or 4 min each, in parallel) ==="
for c in 0 1 2 3 4 5 6 7; do $PY gen_traj_chunk.py --chunk $c --n_keep 150 --max_wall_s 240 --out_dir $T 2>&1 | grep -v Warn | tail -2 & done; wait
echo "=== 3. quick training (CPU, 40 epochs) + plots in plots/detectability/_test_p0e0 ==="
$PY train_p0e0_emulator.py --data_dir $T --out_name _test_p0e0 --epochs 40 --device cpu 2>&1 | grep -v "Warn\|torch.load\|return torch" | tail -25
