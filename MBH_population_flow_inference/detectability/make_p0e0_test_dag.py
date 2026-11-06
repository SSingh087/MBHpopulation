"""make_p0e0_test_dag.py -- small TEST DAG for the p0/e0 emulator pipeline (not submitted by this script).
  gen_00..gen_15 (<=50 rows or 15 min each; bisection + hang-proof child) -> train (40 epochs, plots in plots/detectability/_test_p0e0)
Outputs: logs/p0e0_test/{timing.out,train.out,gen_*.out}"""
import os
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"; CODE = os.path.dirname(os.path.abspath(__file__))
name = "p0e0_test2"; dag = f"{DATA}/dag/{name}"; logs = f"{DATA}/logs/{name}"; out = f"{DATA}/simulations/p0e0_test2"
for d in (dag, logs, out): os.makedirs(d, exist_ok=True)
PY = "/home/2673888s/.conda/envs/redback_few/bin/python"; N = 16
def wrapper(fn, body): open(f"{dag}/{fn}", "w").write("#!/bin/bash\nexport PYTHONPATH=/home/2673888s/poplar\nexport MPLBACKEND=Agg\n" + body + "\n"); os.chmod(f"{dag}/{fn}", 0o755)
wrapper("gen.sh", f'{PY} {CODE}/gen_traj_chunk.py --chunk "$1" --n_keep 50 --max_wall_s 900 --out_dir {out}')
wrapper("train.sh", f'{PY} {CODE}/train_p0e0_emulator.py --data_dir {out} --out_name _test_p0e0 --epochs 40')
sub = lambda ex, args, cpus, mem, gpus, tag: (f"universe = vanilla\nexecutable = {dag}/{ex}\narguments = {args}\nrequest_cpus = {cpus}\nrequest_memory = {mem}\nrequest_gpus = {gpus}\n"
                                              f"getenv = True\noutput = {logs}/{tag}.out\nerror = {logs}/{tag}.err\nlog = {logs}/{name}.log\nqueue\n")
open(f"{dag}/gen.sub", "w").write(sub("gen.sh", "$(chunk)", 1, "2GB", 0, "gen_$(chunk)"))
open(f"{dag}/train.sub", "w").write(sub("train.sh", "", 2, "8GB", 1, "train"))
with open(f"{dag}/{name}.dag", "w") as f:
    for c in range(N): f.write(f"JOB gen_{c:02d} {dag}/gen.sub\nVARS gen_{c:02d} chunk=\"{c}\"\n")
    f.write(f"JOB train {dag}/train.sub\nPARENT " + " ".join(f"gen_{c:02d}" for c in range(N)) + " CHILD train\n")
print(f"wrote {dag}/{name}.dag\nsubmit: condor_submit_dag {dag}/{name}.dag\nresults: {logs}/train.out  plots/detectability/_test_p0e0/")
