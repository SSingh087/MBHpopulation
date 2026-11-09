#!/usr/bin/env python
"""make_sf_brute_mc_dag.py -- DAG for sf_brute_mc.py: 40 Lambda x 2 chunks x 1e8 Monte Carlo samples (seeded per Lambda and chunk), then merge+report.
  python make_sf_brute_mc_dag.py ; cd <DATA>/dag/sf_brute_mc && condor_submit_dag -maxidle 100 sf_brute_mc.dag     (submit ONCE; never -f)"""
import os
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"; D = f"{DATA}/dag/sf_brute_mc"; L = f"{DATA}/logs/sf_brute_mc"; os.makedirs(D, exist_ok=True); os.makedirs(L, exist_ok=True)
CODE = "/home/2673888s/EMRI_TDE_flow_inference/MBHpopulation/MBH_population_flow_inference/detectability"; PY = "/home/2673888s/.conda/envs/redback_few/bin/python"
open(f"{D}/run.sh", "w").write(f"#!/bin/bash\nexport OMP_NUM_THREADS=2 MKL_NUM_THREADS=2\ncd {CODE}\n{PY} -u -W ignore sf_brute_mc.py \"$@\"\n"); os.chmod(f"{D}/run.sh", 0o755)
sub = f"universe = vanilla\nexecutable = {D}/run.sh\narguments = $(args)\ngetenv = True\nlog = {L}/sf_brute_mc.log\noutput = {L}/$(node).out\nerror = {L}/$(node).err\nrequest_cpus = 2\nrequest_memory = 8GB\nrequest_disk = 2GB\nrequest_gpus = 1\nrequire_gpus = Capability >= 7.0 && Capability < 9.0\npriority = 15\n\nqueue 1\n"
open(f"{D}/node.sub", "w").write(sub)
lines, nodes = [], []
for k in range(40):
    for c in range(2):
        n = f"mc_{k:02d}_{c}"; nodes.append(n); lines += [f"JOB {n} {D}/node.sub", f'VARS {n} node="{n}" args="--lam {k} --chunk {c} --n 1e8"', f"RETRY {n} 2"]
lines += [f"JOB report {D}/node.sub", 'VARS report node="report" args="--report"', "RETRY report 1", "PARENT " + " ".join(nodes) + " CHILD report"]
open(f"{D}/sf_brute_mc.dag", "w").write("\n".join(lines) + "\n"); print(len(nodes), "MC nodes + report ->", D)
