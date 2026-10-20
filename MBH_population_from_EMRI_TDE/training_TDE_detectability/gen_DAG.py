#!/usr/bin/env python3
"""
Usage:  python gen_DAG.py --survey ZTF --n_chunks 100

Per survey, emits:
  for each active redback TDE model (parallel branches):
    - a param-gen JOB (gen_TDE_params.sub, SURVEY+model)
      -> PARENT of that model's N SNR-chunk worker JOBs (gen_TDE_SNRs_PARALLEL.sub)
  all SNR-chunk worker JOBs (across all active models) -> PARENT of ONE final
  merge JOB (merge_TDE_SNRs.sub) that merges each model's results (each model
  is trained as its own independent classifier -- see merge_TDE_SNRs.py).
"""
import argparse, os

# bpl_cooling_envelope and gaussianrise_cooling_envelope are excluded: both
# crash deterministically (KeyError: 'bands') under redback's
# SimulateOpticalTransient (confirmed still broken in redback 1.18.0).
MODELS = [
    "cooling_envelope",
    "stream_stream_tde",
    "tde_analytical",
    "tde_fallback",
]

parser = argparse.ArgumentParser()
parser.add_argument("--survey",   required=True)
parser.add_argument("--n_chunks", type=int, default=100, help="SNR chunks PER MODEL")
args = parser.parse_args()

S = args.survey
N = args.n_chunks
lines = []

all_chunk_jobs = []

for model in MODELS:
    # ── Param-gen job for this model ────────────────────────────────────────
    param_job = f"PARAMS_{S}_{model}"
    lines += [
        f"JOB      {param_job}  SUBMIT_FILES/gen_TDE_params.sub",
        f"VARS     {param_job}  SURVEY=\"{S}\" MODEL=\"{model}\"",
        "",
    ]

    # ── SNR-chunk worker jobs for this model ────────────────────────────────
    model_chunk_jobs = []
    for i in range(N):
        job = f"SNR_{S}_{model}_chunk{i:04d}"
        model_chunk_jobs.append(job)
        lines += [
            f"JOB      {job}  SUBMIT_FILES/gen_TDE_SNRs_PARALLEL.sub",
            f"VARS     {job}  SURVEY=\"{S}\" MODEL=\"{model}\" CHUNK_ID=\"{i}\" N_CHUNKS=\"{N}\"",
            f"CATEGORY {job}  snr_workers",   # ← assign to named category
            "",
        ]

    # Param-gen job is parent of this model's chunk jobs
    lines.append(f"PARENT {param_job} CHILD {' '.join(model_chunk_jobs)}")
    lines.append("")

    all_chunk_jobs.extend(model_chunk_jobs)

# ── Cap concurrent jobs in that category (across all models) ───────────────
max_concurrent = 50                        # tune to cluster policy
lines.append(f"MAXJOBS snr_workers {max_concurrent}")

# ── Merge job (combines all 6 models into one injection set) ───────────────
merge_job = f"MERGE_{S}"
lines += [
    f"JOB  {merge_job}  SUBMIT_FILES/merge_TDE_SNRs.sub",
    f"VARS {merge_job}  SURVEY=\"{S}\" N_CHUNKS=\"{N}\"",
    "",
]

lines.append(f"PARENT {' '.join(all_chunk_jobs)} CHILD {merge_job}")

dag_path = f"dag_{S}_{N}chunks.dag"
with open(dag_path, "w") as f:
    f.write("\n".join(lines))

n_param_jobs = len(MODELS)
n_worker_jobs = len(all_chunk_jobs)
print(f"Written: {dag_path}  "
      f"({n_param_jobs} param-gen jobs across {len(MODELS)} models + "
      f"{n_worker_jobs} SNR worker jobs ({N} chunks x {len(MODELS)} models) + "
      f"1 merge job = {n_param_jobs + n_worker_jobs + 1} total jobs)")
