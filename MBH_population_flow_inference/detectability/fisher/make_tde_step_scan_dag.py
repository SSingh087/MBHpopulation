"""make_tde_step_scan_dag.py -- DAG that finds STABLE finite-difference steps for the tde_fallback Fisher (not submitted here).
  scan_{ZTF,LSST}_{00..NN}: tde_fisher_bank.py --scan 1 (F_kk vs step multiplier for each of the 7 parameters, n events each)
  -> plot: plot_tde_step_scan.py (plateau plots + recommended step per parameter in plots/detectability/tde_fisher/)
Outputs: simulations/tde_fisher_scan/*.npz, logs/tde_step_scan/*.out"""
import os, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--n_chunks", type=int, default=8); ap.add_argument("--n_events", type=int, default=8)
ap.add_argument("--max_wall_s", type=float, default=3000); args = ap.parse_args()
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"; CODE = os.path.dirname(os.path.abspath(__file__))
name = "tde_step_scan"; dag = f"{DATA}/dag/{name}"; logs = f"{DATA}/logs/{name}"; out = f"{DATA}/simulations/tde_fisher_scan"
for d in (dag, logs, out): os.makedirs(d, exist_ok=True)
PY = "/home/2673888s/.conda/envs/redback_few/bin/python"
def wrapper(fn, body): open(f"{dag}/{fn}", "w").write("#!/bin/bash\nexport MPLBACKEND=Agg\n" + body + "\n"); os.chmod(f"{dag}/{fn}", 0o755)
wrapper("scan.sh", f'{PY} {CODE}/tde_fisher_bank.py --survey "$1" --chunk "$2" --n_events {args.n_events} --scan 1 --max_wall_s {args.max_wall_s} --out_dir {out}')
wrapper("plot.sh", f'{PY} {CODE}/plot_tde_step_scan.py --scan_dir {out}')
sub = lambda ex, a, cpus, mem, tag: (f"universe = vanilla\nexecutable = {dag}/{ex}\narguments = {a}\nrequest_cpus = {cpus}\nrequest_memory = {mem}\n"
                                     f"getenv = True\noutput = {logs}/{tag}.out\nerror = {logs}/{tag}.err\nlog = {logs}/{name}.log\nqueue\n")
open(f"{dag}/scan.sub", "w").write(sub("scan.sh", "$(survey) $(chunk)", 1, "3GB", "scan_$(survey)_$(chunk)"))
open(f"{dag}/plot.sub", "w").write(sub("plot.sh", "", 1, "4GB", "plot"))
nodes = []
with open(f"{dag}/{name}.dag", "w") as f:
    for s in ("ZTF", "LSST"):
        for c in range(args.n_chunks):
            n = f"scan_{s}_{c:02d}"; nodes.append(n); f.write(f'JOB {n} {dag}/scan.sub\nVARS {n} survey="{s}" chunk="{c}"\n')
    f.write(f"JOB plot {dag}/plot.sub\nPARENT {' '.join(nodes)} CHILD plot\n")
print(f"wrote {dag}/{name}.dag\nsubmit: condor_submit_dag {dag}/{name}.dag\nresults: plots/detectability/tde_fisher/step_scan_*.png, logs/tde_step_scan/plot.out")
