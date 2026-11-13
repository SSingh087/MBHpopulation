"""One-job GPU DAG: runs emri_snr_diag.py on the emri_fisher_test bank.  Output: logs/emri_snr_diag/diag.out"""
import os
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"; CODE = os.path.dirname(os.path.abspath(__file__))
name = "emri_snr_diag"; dag = f"{DATA}/dag/{name}"; logs = f"{DATA}/logs/{name}"
for d in (dag, logs): os.makedirs(d, exist_ok=True)
PY = "/home/2673888s/.conda/envs/redback_few/bin/python"
open(f"{dag}/diag.sh", "w").write(f"#!/bin/bash\nexport PYTHONPATH=/home/2673888s/poplar\n{PY} {CODE}/emri_snr_diag.py --gpu 1 --bank_dir {DATA}/simulations/emri_fisher_test\n"); os.chmod(f"{dag}/diag.sh", 0o755)
open(f"{dag}/diag.sub", "w").write(f"universe = vanilla\nexecutable = {dag}/diag.sh\nrequest_cpus = 2\nrequest_memory = 12GB\nrequest_gpus = 1\n"
    f"require_gpus = Capability >= 7.0 && Capability < 9.0\ngetenv = True\noutput = {logs}/diag.out\nerror = {logs}/diag.err\nlog = {logs}/{name}.log\nqueue\n")
open(f"{dag}/{name}.dag", "w").write(f"JOB diag {dag}/diag.sub\n")
print(f"wrote {dag}/{name}.dag\nsubmit: condor_submit_dag {dag}/{name}.dag\nresults: {logs}/diag.out")
