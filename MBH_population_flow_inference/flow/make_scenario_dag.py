#!/usr/bin/env python
"""
One top-level DAG per scenario: simulate N universes (CPU chunks) -> consolidate (CPU) -> flow sweep (GPU jobs + select + held-out + acceptance).
Everything under <DATA>/scenarios/<scenario>/.   Writes only; submit with condor_submit_dag.

  python make_scenario_dag.py --scenario sigmaHV_legacy_a075 --nucleation hannah --occupation varying --alpha_Re 0.75 --legacy 1
  cd <DATA>/scenarios/<scenario>/dag/top && condor_submit_dag -maxjobs 150 top.dag
"""
import argparse, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"; PY = "/home/2673888s/.conda/envs/redback_few/bin/python"
ap = argparse.ArgumentParser()
ap.add_argument("--scenario", required=True); ap.add_argument("--nucleation", required=True); ap.add_argument("--occupation", required=True)
ap.add_argument("--alpha_Re", type=float, required=True); ap.add_argument("--legacy", type=int, required=True)
ap.add_argument("--n_universes", type=int, default=10000); ap.add_argument("--per_job", type=int, default=50)
a = ap.parse_args()
R = f"{DATA}/scenarios/{a.scenario}"; top = f"{R}/dag/top"; os.makedirs(top, exist_ok=True); os.makedirs(f"{R}/logs/top", exist_ok=True)
spec = f"{a.nucleation},{a.occupation},{a.alpha_Re},{a.legacy}"
run = lambda cmd: subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
run([PY, f"{HERE}/../sources/make_dag.py", "--scenario", a.scenario, "--name", "sources", "--nucleation", a.nucleation, "--occupation", a.occupation,
     "--alpha_Re", str(a.alpha_Re), "--legacy", str(a.legacy), "--n_universes", str(a.n_universes), "--per_job", str(a.per_job)])
run([PY, f"{HERE}/make_sweep_dag.py", "--scenario", a.scenario, "--spec", spec, "--name", "flow_sweep"])
sh = f"{top}/consolidate.sh"
open(sh, "w").write(f"#!/bin/bash\nexport EMRI_SCENARIO_ROOT={R}\ncd {HERE}\n{PY} build_dataset.py\n"); os.chmod(sh, 0o755)
open(f"{top}/consolidate.sub", "w").write(f"""executable = {sh}
getenv = true
log = {R}/logs/top/consolidate.log
output = {R}/logs/top/consolidate.out
error = {R}/logs/top/consolidate.err
request_cpus = 2
request_memory = 8GB
request_disk = 2GB
queue 1
""")
open(f"{top}/top.dag", "w").write(f"""SPLICE sources {R}/dag/sources/sources.dag
JOB consolidate {top}/consolidate.sub
RETRY consolidate 1
SPLICE sweep {R}/dag/flow_sweep/flow_sweep.dag
PARENT sources CHILD consolidate
PARENT consolidate CHILD sweep
""")
print(f"wrote {top}/top.dag  (spec {spec})")
