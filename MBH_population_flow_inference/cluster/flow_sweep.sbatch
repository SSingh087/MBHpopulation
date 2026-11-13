#!/bin/bash
#SBATCH --job-name=emri_flow
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/flow_%A_%a.out
# EMRI Fisher-error flow hyper-parameter sweep on bank1+bank2 (full data, one config per GPU). Results: ~/emri_run/flow_out/{models,plots}
source ${EMRI_ROOT:-$HOME}/emri_run/env.sh; C="bash ${EMRI_ROOT:-$HOME}/emri_run/ctr.sh"
CFG=("4 64 8" "4 128 8" "4 256 8" "4 512 8" "6 128 8" "6 256 8" "6 512 8" "8 128 8" "8 256 8" "8 512 8" "6 256 16" "6 512 16")
set -- ${CFG[$SLURM_ARRAY_TASK_ID]}; L=$1; H=$2; K=$3
export EMRI_FLOW_DATA=$HOME/emri_run/flow_out; mkdir -p $EMRI_FLOW_DATA
$C python $CODE/detectability/fisher/train_emri_fisher_flow.py --bank_dir $HOME/emri_run/bank1 $HOME/emri_run/emri_fisher_bank2 \
  --device cuda --fracs 1.0 --seeds 0 --n_layers $L --hidden $H --bins $K --ctx_hidden $H --out_name sweep_L${L}_H${H}_K${K}
