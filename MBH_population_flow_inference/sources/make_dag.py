#!/usr/bin/env python
"""
Write the HTCondor DAG for the full source-generation run (does NOT submit anything).

    python make_dag.py --n_universes 10000 --per_job 50
    cd <dag dir> && condor_submit_dag sources_full.dag

Writes (all under /data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference/):
    dag/sources_full/sources_full.dag, sources_chunk.sub, run_chunk.sh
    logs/sources_full/            (condor .log/.out/.err per chunk)
    simulations/full_run/chunk_XXXX/univ_XXXXXX.npz   (created by the jobs)
Resumable: re-submitting (or DAGMan RETRY) only simulates universes whose file is missing.
"""
import argparse, os, stat

CODE = os.path.dirname(os.path.abspath(__file__))
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"
PY = "/home/2673888s/.conda/envs/redback_few/bin/python"

ap = argparse.ArgumentParser()
ap.add_argument("--n_universes", type=int, default=10000)
ap.add_argument("--per_job", type=int, default=50, help="universes per condor job (~5.5 s each => 50 -> ~5 min)")
ap.add_argument("--master_seed", type=int, default=2026)
ap.add_argument("--n_pool", type=int, default=100_000)
ap.add_argument("--name", default="sources_full")
ap.add_argument("--request_memory", default="3GB", help="measured peak ~0.7 GB in the highest-count corner")
ap.add_argument("--retry", type=int, default=2)
args = ap.parse_args()

n_chunks = -(-args.n_universes // args.per_job)
dagdir = f"{DATA}/dag/{args.name}"; logdir = f"{DATA}/logs/{args.name}"; outdir = f"{DATA}/simulations/full_run"
for d in (dagdir, logdir, outdir):
    os.makedirs(d, exist_ok=True)

sh = f"{dagdir}/run_chunk.sh"
with open(sh, "w") as f:
    f.write(f"""#!/bin/bash
# $1 = chunk index
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
{PY} {CODE}/run_chunk.py --chunk $1 --per_chunk {args.per_job} --master_seed {args.master_seed} \\
    --n_pool {args.n_pool} --outdir {outdir}
""")
os.chmod(sh, os.stat(sh).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

with open(f"{dagdir}/sources_chunk.sub", "w") as f:
    f.write(f"""executable     = {sh}
arguments      = $(chunk)
getenv         = true
log            = {logdir}/chunk_$(chunk).log
output         = {logdir}/chunk_$(chunk).out
error          = {logdir}/chunk_$(chunk).err
request_cpus   = 1
request_memory = {args.request_memory}
request_disk   = 500MB
notify_user    = s.singh.3@research.gla.ac.uk
notification   = Error
queue 1
""")

with open(f"{dagdir}/{args.name}.dag", "w") as f:
    for c in range(n_chunks):
        f.write(f"JOB chunk_{c:04d} {dagdir}/sources_chunk.sub\nVARS chunk_{c:04d} chunk=\"{c}\"\nRETRY chunk_{c:04d} {args.retry}\n")

print(f"""wrote DAG with {n_chunks} jobs x {args.per_job} universes = {n_chunks*args.per_job} universes (master seed {args.master_seed})
  dag dir : {dagdir}
  outputs : {outdir}/chunk_XXXX/univ_XXXXXX.npz
  logs    : {logdir}
estimate: ~{args.per_job*5.5/60:.1f} min/job, ~{args.n_universes*5.5/3600:.1f} core-hours total, ~{args.n_universes*165/1e6:.1f} GB disk
submit  : cd {dagdir} && condor_submit_dag {args.name}.dag""")
