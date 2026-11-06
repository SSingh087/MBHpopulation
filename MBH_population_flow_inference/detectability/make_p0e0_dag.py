"""make_p0e0_dag.py -- DAG: N parallel forward-FEW data chunks -> train the p0/e0 emulator (GPU).
Usage: python detectability/make_p0e0_dag.py --n_chunks 100 --n_keep 2000 ; then condor_submit_dag <dag>
(Not submitted by this script.)"""
import argparse, os
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"; CODE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--n_chunks", type=int, default=100); ap.add_argument("--n_keep", type=int, default=2000)
ap.add_argument("--epochs", type=int, default=300); a = ap.parse_args()
name = "p0e0_emulator"; dag = f"{DATA}/dag/{name}"; logs = f"{DATA}/logs/{name}"; out = f"{DATA}/simulations/p0e0_traj"
for d in (dag, logs, out): os.makedirs(d, exist_ok=True)
PY = "/home/2673888s/.conda/envs/redback_few/bin/python"
def wrapper(fn, body): open(f"{dag}/{fn}", "w").write("#!/bin/bash\nexport PYTHONPATH=/home/2673888s/poplar\nexport MPLBACKEND=Agg\n" + body + "\n"); os.chmod(f"{dag}/{fn}", 0o755)
wrapper("gen.sh", f'{PY} {CODE}/gen_traj_chunk.py --chunk "$1" --n_keep {a.n_keep} --out_dir {out}')
wrapper("train.sh", f'{PY} {CODE}/train_p0e0_emulator.py --epochs {a.epochs} --out_name {name}')
sub = lambda ex, args, res, tag: (f"universe = vanilla\nexecutable = {dag}/{ex}\narguments = {args}\nrequest_cpus = {res[0]}\nrequest_memory = {res[1]}\nrequest_gpus = {res[2]}\n"
                                  f"getenv = True\noutput = {logs}/{tag}.out\nerror = {logs}/{tag}.err\nlog = {logs}/{name}.log\nqueue\n")
open(f"{dag}/gen.sub", "w").write(sub("gen.sh", "$(chunk)", (1, "2GB", 0), "gen_$(chunk)"))
open(f"{dag}/train.sub", "w").write(sub("train.sh", "", (2, "8GB", 1), "train"))
with open(f"{dag}/{name}.dag", "w") as f:
    for c in range(a.n_chunks): f.write(f"JOB gen_{c:04d} {dag}/gen.sub\nVARS gen_{c:04d} chunk=\"{c}\"\nRETRY gen_{c:04d} 2\n")
    f.write(f"JOB train {dag}/train.sub\nPARENT " + " ".join(f"gen_{c:04d}" for c in range(a.n_chunks)) + " CHILD train\n")
print(f"wrote {dag}/{name}.dag  ({a.n_chunks} data chunks x {a.n_keep} rows -> train)\nsubmit: condor_submit_dag {dag}/{name}.dag")
