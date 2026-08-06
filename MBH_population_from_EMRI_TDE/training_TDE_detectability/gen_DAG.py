#!/usr/bin/env python3
"""
Usage:  python gen_DAG.py --survey ZTF --n_chunks 100
"""
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("--survey",   required=True)
parser.add_argument("--n_chunks", type=int, default=100)
args = parser.parse_args()

S = args.survey
N = args.n_chunks
lines = []

# ── Worker jobs ────────────────────────────────────────────────────────────
for i in range(N):
    job = f"SNR_{S}_chunk{i:04d}"
    lines += [
        f"JOB      {job}  SUBMIT_FILES/gen_TDE_SNRs_PARALLEL.sub",
        f"VARS     {job}  SURVEY=\"{S}\" CHUNK_ID=\"{i}\" N_CHUNKS=\"{N}\"",
        f"CATEGORY {job}  snr_workers",   # ← assign to named category
        "",
    ]

# ── Cap concurrent jobs in that category ──────────────────────────────────
max_concurrent = 50                        # tune to cluster policy
lines.append(f"MAXJOBS snr_workers {max_concurrent}")

# ── Merge job (no category needed, only 1) ────────────────────────────────
merge_job = f"MERGE_{S}"
lines += [
    f"JOB  {merge_job}  SUBMIT_FILES/merge_TDE_SNRs.sub",
    f"VARS {merge_job}  SURVEY=\"{S}\" N_CHUNKS=\"{N}\"",
    "",
]

chunk_jobs = " ".join(f"SNR_{S}_chunk{i:04d}" for i in range(N))
lines.append(f"PARENT {chunk_jobs} CHILD {merge_job}")

dag_path = f"dag_{S}_{N}chunks.dag"
with open(dag_path, "w") as f:
    f.write("\n".join(lines))

print(f"Written: {dag_path}  ({N} worker jobs + 1 merge job)")