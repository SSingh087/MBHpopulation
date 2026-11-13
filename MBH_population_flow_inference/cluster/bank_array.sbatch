#!/bin/bash
#SBATCH --job-name=emri_bank
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --array=0-799%24
#SBATCH --output=logs/bank_%A_%a.out
# one chunk per GPU; same seed 32 and chunk numbering as the HTCondor DAG so chunks are interchangeable. Submit only a range NOT yet done on wiay:
#   sbatch --array=400-799%24 bank_array.sbatch
source ${EMRI_ROOT:-$HOME}/emri_run/env.sh; C="bash ${EMRI_ROOT:-$HOME}/emri_run/ctr.sh"
$C python $CODE/detectability/fisher/emri_fisher_bank.py --chunk $SLURM_ARRAY_TASK_ID --n_events 20 --target_snr 100.0 --proposal mixed --catalogue_fraction 0.7 \
  --master_seed 32 --max_wall_s 20000 --traj_dir $EMRI_DATA/simulations/p0e0_traj --out_dir ${EMRI_ROOT:-$HOME}/emri_run/emri_fisher_bank2
