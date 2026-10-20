#!/usr/bin/env python
"""
Write the Fisher DAG of one source and scenario: one node per batch of detected events,
then one merge node that builds fisher_posteriors_<source>_<scenario>.(h5|pt).

The number of batches comes from the detected catalogue written by
hierarchical_inference/prepare_data_MEDIAN.py, so this runs as a PRE script of a SUBDAG
(see hierarchical_inference/gen_dag.py), after the best universe is known.

Usage:
    python make_fisher_dag.py --GALAXIES 100000 --SOURCE EMRI --SCENARIO use_sigma_hannah_uniform
    python make_fisher_dag.py --GALAXIES 100000 --SOURCE TDE_ZTF --SCENARIO use_sigma_hannah_uniform --BATCH_SIZE 15
"""
import argparse

import h5py

import os, sys
sys.path.insert(0, os.path.abspath("./"))
from utils import INFERENCE_DATA_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES", type=int, required=True)
parser.add_argument("--SOURCE", type=str, required=True, choices=["EMRI", "TDE_ZTF", "TDE_LSST"])
parser.add_argument("--SCENARIO", type=str, required=True,
                    help="{mbh_mass_method}_{nucleation_method}_{occupation_method}")
parser.add_argument("--BATCH_SIZE", type=int, default=15, help="events per DAG node")
parser.add_argument("--WORKER_SUB", type=str, default="SUBMIT_FILES/fisher.sub")
parser.add_argument("--MERGE_SUB", type=str, default="SUBMIT_FILES/merge_fisher.sub")
args = parser.parse_args()

catalog = f"{INFERENCE_DATA_DIR}/{args.GALAXIES}/detected_catalog_{args.SOURCE}_{args.SCENARIO}.h5"
with h5py.File(catalog, "r") as f:
    n_total = f["z_gal"].shape[0]

tag = f"{args.SOURCE}_{args.SCENARIO}"
dag_path = f"fisher_{tag}.dag"
n_batches = (n_total + args.BATCH_SIZE - 1) // args.BATCH_SIZE
print(f"{n_total} detected {args.SOURCE} events -> {n_batches} batches of up to {args.BATCH_SIZE}")

lines = []
for b in range(n_batches):
    start, end = b * args.BATCH_SIZE, min((b + 1) * args.BATCH_SIZE, n_total)
    node = f"batch{b}_{tag}"
    lines += [f"JOB {node} {args.WORKER_SUB}",
              f'VARS {node} source="{args.SOURCE}" scenario="{args.SCENARIO}" start_idx="{start}" '
              f'end_idx="{end}" batch_id="{b}"',
              f"RETRY {node} 2", ""]

merge = f"merge_{tag}"
lines += [f"JOB {merge} {args.MERGE_SUB}", f'VARS {merge} source="{args.SOURCE}" scenario="{args.SCENARIO}"']
if n_batches:
    lines.append("PARENT " + " ".join(f"batch{b}_{tag}" for b in range(n_batches)) + f" CHILD {merge}")

with open(dag_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {dag_path}")
