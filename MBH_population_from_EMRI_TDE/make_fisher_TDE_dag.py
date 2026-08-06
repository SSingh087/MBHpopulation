#!/usr/bin/env python
"""
Writes fisher_TDE_{SURVEY}.dag ONLY. Assumes SUBMIT_FILES/fisher_TDE.sub
and SUBMIT_FILES/merge.sub (or whatever you point --WORKER_SUB /
--MERGE_SUB at) already exist -- static files, not generated here.

No header comments are written, on purpose: this file should never be
hand-edited (sed/tail'd, etc.) after generation. If the DAG is wrong,
fix the CLI args and regenerate -- don't patch the .dag by hand, since
that's exactly what caused the "Unknown Node batch0_ZTF" failure (an
off-by-one line strip that ate the first JOB line along with the
header).

Ends with a self-check: parses its own output and confirms every node
referenced in the PARENT/CHILD line has a matching JOB line, so a
malformed DAG fails loudly here instead of silently at condor_submit_dag
time.

Usage:
    python make_fisher_TDE_dag.py --GALAXIES 100000 --SURVEY ZTF --BATCH_SIZE 15 --mbh_mass_method use_sigma --nucleation_method hannah
"""

import os, re, argparse
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES",      type=int, required=True)
parser.add_argument("--SURVEY",        type=str, required=True, choices=["ZTF", "LSST"])
parser.add_argument("--BATCH_SIZE",    type=int, default=15, help="events per DAG node (10-20 recommended)")
parser.add_argument("--WORKER_SUB",    type=str, default="SUBMIT_FILES/fisher_TDE.sub")
parser.add_argument("--MERGE_SUB",     type=str, default="SUBMIT_FILES/merge.sub")
parser.add_argument("--mbh_mass_method", type=str, required=False, default="M_sigma", choices=["use_sigma", "use_lgMgal"])
parser.add_argument("--nucleation_method", type=str, required=False, default="hannah", choices=["hannah", "neumayer"])
args = parser.parse_args()

DATA_DIR = "./data"
loc      = f"{DATA_DIR}/fisher_data/{args.GALAXIES}"
os.makedirs(loc, exist_ok=True)
os.makedirs("condor_logs", exist_ok=True)

SNR_PATH = f"{DATA_DIR}/astrophysical_data/{args.GALAXIES}/TDE_SNRs_{args.SURVEY}_{args.mbh_mass_method}_{args.nucleation_method}.h5"

with h5py.File(SNR_PATH, "r") as f:
    n_total = f["z_gal"].shape[0]

if n_total == 0:
    raise RuntimeError(f"{SNR_PATH} has 0 detection-passed events -- nothing to build a DAG for.")

n_batches = (n_total + args.BATCH_SIZE - 1) // args.BATCH_SIZE
print(f"{n_total} detection-passed {args.SURVEY} events -> {n_batches} batches of up to {args.BATCH_SIZE}")

dag_path = f"fisher_TDE_{args.SURVEY}_{args.mbh_mass_method}_{args.nucleation_method}.dag"   # written to CWD, matching your existing convention

lines = []
job_nodes = set()

for b in range(n_batches):
    start = b * args.BATCH_SIZE
    end = min(start + args.BATCH_SIZE, n_total)
    node = f"batch{b}_{args.SURVEY}_{args.mbh_mass_method}_{args.nucleation_method}"
    job_nodes.add(node)
    lines.append(f"JOB {node} {args.WORKER_SUB}")
    lines.append(f'VARS {node} survey="{args.SURVEY}" mbh_mass_method="{args.mbh_mass_method}" nucleation_method="{args.nucleation_method}" start_idx="{start}" end_idx="{end}" batch_id="{b}"')
    lines.append(f"RETRY {node} 2")
    lines.append("")

merge_node = f"merge_{args.SURVEY}_{args.mbh_mass_method}_{args.nucleation_method}"
job_nodes.add(merge_node)
lines.append(f"JOB {merge_node} {args.MERGE_SUB}")
lines.append(f'VARS {merge_node} survey="{args.SURVEY}" mbh_mass_method="{args.mbh_mass_method}" nucleation_method="{args.nucleation_method}"')
parent_line = "PARENT " + " ".join(f"batch{b}_{args.SURVEY}_{args.mbh_mass_method}_{args.nucleation_method}" for b in range(n_batches)) + f" CHILD {merge_node}"
lines.append(parent_line)

with open(dag_path, "w") as f:
    f.write("\n".join(lines) + "\n")

# ---------------------------------------------------------------
# Self-check: re-read what we just wrote, confirm every node named in
# a PARENT/CHILD line was actually declared via a JOB line above it.
# ---------------------------------------------------------------
with open(dag_path) as f:
    written = f.read().splitlines()

declared = set()
referenced = set()
for i, line in enumerate(written):
    if line.startswith("JOB "):
        declared.add(line.split()[1])
    elif line.startswith("PARENT "):
        # PARENT n1 n2 ... CHILD c1 c2
        toks = line.split()
        child_idx = toks.index("CHILD")
        referenced.update(toks[1:child_idx])
        referenced.update(toks[child_idx + 1:])

missing = referenced - declared
if missing:
    raise RuntimeError(
        f"[SELF-CHECK FAILED] {dag_path} references node(s) {missing} in a "
        f"PARENT/CHILD line with no matching JOB line. Not submitting-safe -- "
        f"aborting before you hit this at condor_submit_dag time."
    )

print(f"[SELF-CHECK OK] all {len(referenced)} PARENT/CHILD-referenced nodes have matching JOB lines "
      f"({len(declared)} JOB nodes total).")
# print(f"[WROTE] {dag_path}  ({len(written)} lines)")
# print(f"\nSubmit with:\n  condor_submit_dag {dag_path}")