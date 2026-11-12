#!/bin/bash
#SBATCH --job-name=emri_test
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=ALL
source ${EMRI_ROOT:-$HOME}/emri_run/env.sh; C="bash ${EMRI_ROOT:-$HOME}/emri_run/ctr.sh"
nvidia-smi -L; $C python -c "import torch,cupy;print('torch gpu',torch.cuda.is_available(),'cupy',cupy.cuda.runtime.getDeviceCount())"
$C python $CODE/detectability/fisher/emri_fisher_bank.py --chunk 0 --n_events 8 --target_snr 100.0 --proposal mixed --catalogue_fraction 0.7 \
  --master_seed 31 --max_wall_s 3000 --traj_dir $EMRI_DATA/simulations/p0e0_traj --out_dir ${EMRI_ROOT:-$HOME}/emri_run/test_out
echo "DONE: compare ${EMRI_ROOT:-$HOME}/emri_run/test_out/chunk_0000.npz with the same chunk on wiay (simulations/emri_fisher_conc_test was deleted; use bank chunk_0000 with --n_events 8: first events must match)"
