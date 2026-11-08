"""Write (do NOT submit) a one-node DAG that runs check_fixed_points.py (mixture proposal) on a free GPU.
   python make_fp_check_dag.py ; cd <DATA>/dag/fp_check && condor_submit_dag fp_check.dag   (submit once)
Output: logs/fp_check/check.out (unbuffered, watch with tail -f), logs/check_fixed_points_cuda.json"""
import os
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"; HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/home/2673888s/.conda/envs/redback_few/bin/python"; dag, logs = f"{DATA}/dag/fp_check", f"{DATA}/logs/fp_check"
os.makedirs(dag, exist_ok=True); os.makedirs(logs, exist_ok=True)
open(f"{dag}/run.sh", "w").write(f"#!/bin/bash\nexport OMP_NUM_THREADS=2\ncd {HERE}\n{PY} -u -W ignore check_fixed_points.py --device cuda --proposal mixture --L_mix 32 --repl 4 --k_list 8 32 128 --m_list 14 16 18 20\n"); os.chmod(f"{dag}/run.sh", 0o755)
open(f"{dag}/run.sub", "w").write(f"universe = vanilla\nexecutable = {dag}/run.sh\ngetenv = True\nlog = {logs}/check.log\noutput = {logs}/check.out\nerror = {logs}/check.err\nrequest_cpus = 2\nrequest_memory = 16GB\nrequest_disk = 2GB\npriority = 20\nrequest_gpus = 1\nrequire_gpus = Capability >= 7.0 && Capability < 9.0\nqueue 1\n")
open(f"{dag}/fp_check.dag", "w").write(f"JOB check {dag}/run.sub\nRETRY check 1\nPRIORITY check 20\n")
print(f"wrote {dag}/fp_check.dag\nsubmit once: cd {dag} && condor_submit_dag fp_check.dag\nwatch: tail -f {logs}/check.out")
