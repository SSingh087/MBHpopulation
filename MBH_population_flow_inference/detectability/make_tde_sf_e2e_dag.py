#!/usr/bin/env python
"""make_tde_sf_e2e_dag.py -- DAG for tde_sf_endtoend.py (TDE selection-function end-to-end test). Not submitted here.
  python make_tde_sf_e2e_dag.py ; cd <DATA>/dag/tde_sf_e2e && condor_submit_dag -maxidle 300 tde_sf_e2e.dag     (submit ONCE; never -f)
ZTF: 5 Lambda x 20 chunks x 10000 draws (2e5 per Lambda); LSST: 5 Lambda x 100 chunks x 1000 draws (1e5 per Lambda); then report.  Parallel over events/chunks only (seeded per chunk)."""
import os
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"; D = f"{DATA}/dag/tde_sf_e2e"; L = f"{DATA}/logs/tde_sf_e2e"; os.makedirs(D, exist_ok=True); os.makedirs(L, exist_ok=True)
CODE = "/home/2673888s/EMRI_TDE_flow_inference/MBHpopulation/MBH_population_flow_inference/detectability"; PY = "/home/2673888s/.conda/envs/redback_few/bin/python"
open(f"{D}/run.sh", "w").write(f"#!/bin/bash\nexport OMP_NUM_THREADS=2 MKL_NUM_THREADS=2\ncd {CODE}\n{PY} -u -W ignore tde_sf_endtoend.py \"$@\"\n"); os.chmod(f"{D}/run.sh", 0o755)
sub = lambda cpus, mem: f"universe = vanilla\nexecutable = {D}/run.sh\narguments = $(args)\ngetenv = True\nlog = {L}/tde_sf_e2e.log\noutput = {L}/$(node).out\nerror = {L}/$(node).err\nrequest_cpus = {cpus}\nrequest_memory = {mem}\nrequest_disk = 2GB\npriority = 15\n\nqueue 1\n"
open(f"{D}/node.sub", "w").write(sub(4, "8GB")); open(f"{D}/report.sub", "w").write(sub(1, "4GB"))
lines, nodes = [], []
for sv, nch, nd in (("ZTF", 20, 10000), ("LSST", 100, 1000)):
    for lam in range(5):
        for c in range(nch):
            n = f"{sv}_{lam}_{c:03d}"; nodes.append(n)
            lines += [f"JOB {n} {D}/node.sub", f'VARS {n} node="{n}" args="--survey {sv} --lam {lam} --chunk {c} --n_draw {nd} --workers 4"', f"RETRY {n} 2"]
lines += [f"JOB report {D}/report.sub", 'VARS report node="report" args="--report"', "RETRY report 1", "PARENT " + " ".join(nodes) + " CHILD report"]
open(f"{D}/tde_sf_e2e.dag", "w").write("\n".join(lines) + "\n"); print(len(nodes), "nodes ->", D)
