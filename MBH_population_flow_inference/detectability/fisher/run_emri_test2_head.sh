#!/bin/bash
# Head-node version of the emri_fisher_test2 DAG: 6 chunks x 3 events, fixed Fisher SNR 100, on the two head-node GPUs in parallel
# (chunks 0-2 on GPU 0, chunks 3-5 on GPU 1), then the summary.  Detached with nohup so it survives closing the terminal.
# Usage: bash run_emri_test2_head.sh        Monitor: tail -f $LOGS/head_gpu0.log   Done when $LOGS/DONE_head exists.
DATA=/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference
CODE=/home/2673888s/EMRI_TDE_flow_inference/MBHpopulation/MBH_population_flow_inference/detectability/fisher
PY=/home/2673888s/.conda/envs/redback_few/bin/python
OUT=$DATA/simulations/emri_fisher_test2; LOGS=$DATA/logs/emri_fisher_test2; mkdir -p $OUT $LOGS; rm -f $LOGS/DONE_head
export PYTHONPATH=/home/2673888s/poplar MPLBACKEND=Agg
run_gpu() {  # $1 = gpu index, rest = chunks
  g=$1; shift
  for c in "$@"; do
    CUDA_VISIBLE_DEVICES=$g $PY $CODE/emri_fisher_bank.py --chunk $c --n_events 3 --target_snr 100 --max_wall_s 3000 \
      --traj_dir $DATA/simulations/p0e0_traj --out_dir $OUT >> $LOGS/head_gpu$g.log 2>&1
  done
}
( run_gpu 0 0 1 2 & run_gpu 1 3 4 5 & wait
  $PY $CODE/summarize_emri_fisher.py --bank_dir $OUT --tag emri_fisher_test2 > $LOGS/summary_head.out 2>&1
  touch $LOGS/DONE_head ) > /dev/null 2>&1 &
echo "started; logs: $LOGS/head_gpu0.log $LOGS/head_gpu1.log ; summary: $LOGS/summary_head.out"
