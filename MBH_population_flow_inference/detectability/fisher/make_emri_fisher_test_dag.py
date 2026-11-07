"""make_emri_fisher_test_dag.py -- small GPU TEST DAG for the EMRI Fisher bank (not submitted here).
  fish_00..fish_07: emri_fisher_bank.py (4 events each, with a C2 re-evaluation) on the p0e0_test2 trajectory rows -> summary node
Look at: logs/emri_fisher_test/summary.out (timing, failures, cond numbers) and plots/detectability/emri_fisher/test_*.png"""
import os, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--n_chunks", type=int, default=8); ap.add_argument("--n_events", type=int, default=4)
ap.add_argument("--live_dangerously", type=int, default=0); ap.add_argument("--max_wall_s", type=float, default=2400); args = ap.parse_args()
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"; CODE = os.path.dirname(os.path.abspath(__file__))
name = "emri_fisher_test"; dag = f"{DATA}/dag/{name}"; logs = f"{DATA}/logs/{name}"; out = f"{DATA}/simulations/emri_fisher_test"
for d in (dag, logs, out): os.makedirs(d, exist_ok=True)
PY = "/home/2673888s/.conda/envs/redback_few/bin/python"
def wrapper(fn, body): open(f"{dag}/{fn}", "w").write("#!/bin/bash\nexport PYTHONPATH=/home/2673888s/poplar\nexport MPLBACKEND=Agg\n" + body + "\n"); os.chmod(f"{dag}/{fn}", 0o755)
wrapper("fish.sh", f'{PY} {CODE}/emri_fisher_bank.py --chunk "$1" --n_events {args.n_events} --live_dangerously {args.live_dangerously} --max_wall_s {args.max_wall_s} --traj_dir {DATA}/simulations/p0e0_test2 --out_dir {out}')
wrapper("summary.sh", f'{PY} {CODE}/summarize_emri_fisher.py --bank_dir {out} --tag test')
GPUREQ = "require_gpus = Capability >= 7.0 && Capability < 9.0\n"   # the RTX 5090 (cc 12.0) is not supported by the installed cupy/nvrtc
sub = lambda ex, a, cpus, mem, gpus, tag: (f"universe = vanilla\nexecutable = {dag}/{ex}\narguments = {a}\nrequest_cpus = {cpus}\nrequest_memory = {mem}\nrequest_gpus = {gpus}\n"
                                           + (GPUREQ if gpus else "") + f"getenv = True\noutput = {logs}/{tag}.out\nerror = {logs}/{tag}.err\nlog = {logs}/{name}.log\nqueue\n")
open(f"{dag}/fish.sub", "w").write(sub("fish.sh", "$(chunk)", 2, "12GB", 1, "fish_$(chunk)"))
open(f"{dag}/summary.sub", "w").write(sub("summary.sh", "", 1, "4GB", 0, "summary"))
with open(f"{dag}/{name}.dag", "w") as f:
    for c in range(args.n_chunks): f.write(f'JOB fish_{c:02d} {dag}/fish.sub\nVARS fish_{c:02d} chunk="{c}"\n')
    f.write(f"JOB summary {dag}/summary.sub\nPARENT " + " ".join(f"fish_{c:02d}" for c in range(args.n_chunks)) + " CHILD summary\n")
print(f"wrote {dag}/{name}.dag\nsubmit: condor_submit_dag {dag}/{name}.dag")
