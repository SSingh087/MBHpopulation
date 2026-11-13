# Handoff: moving the EMRI Fisher bank run from wiay to cognition (2026-11-12)

## Done
- Packed on wiay with pack_here.sh, rsynced to cognition (2673888s@cognition.cose.gla.ac.uk:~/), sizes verified.
- Ran setup_on_cluster.sh on cognition: env at ~/envs/redback_few, code/data at ~/emri_run. torch 2.4.0 (cuda 11.8) imports; GPU is an NVIDIA L40S and torch/cupy see it.
- Tarballs deleted on cognition (originals still in wiay:/data/wiay/postgrads/shashwat/cluster_bundle/).
- LISA SNR model package is at ~/emri_run/LISA_selection_function/lisa_snr_package (model/model.pth + snr_model.py).

## Blocking problem
Test job 1918921 (test_chunk.sbatch) reported COMPLETED/0:0 but python crashed:
`ImportError: /lib64/libm.so.6: version 'GLIBC_2.35' not found` from
few/.../pyinterp_cpu.cpython-312 (fastemriwaveforms 1.5.5, built on wiay).
Cognition's glibc is older. COMPLETED is misleading: the sbatch ends with an echo, so always read the .out file.
Next: run `ldd --version | head -1` on cognition, then rebuild FEW from source in ~/envs/redback_few (or use a container).

## Then
1. Rerun `sbatch ~/emri_run/test_chunk.sbatch`; confirm ~/emri_run/test_out/chunk_0000.npz exists and its first events match wiay's bank chunk_0000 (--n_events 8).
2. Find which chunks wiay already finished; submit only the rest, e.g. `sbatch --array=400-799%24 bank_array.sbatch` (800 chunks, 20 events each, 1 L40S per chunk, master_seed 32, same numbering as the HTCondor DAG).
3. rsync results back (~/emri_run/emri_fisher_bank2), verify chunk counts, only then free wiay. Keep cluster_bundle/ until then.

## Rules
- Do not submit jobs/DAGs unless the user explicitly asks.
- `squeue -u $USER` shows only pending/running jobs; use `sacct -u $USER -X --starttime today` for finished ones.
