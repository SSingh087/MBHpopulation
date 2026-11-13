"""make_tde_bank_dag.py -- TDE Fisher bank DAG (not submitted here).
  bank_{ZTF,LSST}_{00..N}  (CPU, independent, each seeded by (master_seed, chunk, survey); split by EVENTS only)
        --> summary        (PARENT = every bank node, nothing else)
Default is a SMALL TEST (2 surveys x 4 chunks x 10 events).  Production: --n_chunks 200 --n_events 50 (decide after the test).
Outputs: simulations/tde_fisher_bank[_test]/*.npz, logs/tde_fisher_bank[_test]/, plots/detectability/tde_fisher/bank_*.png"""
import os, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--n_chunks", type=int, default=4); ap.add_argument("--n_events", type=int, default=10)
ap.add_argument("--max_wall_s", type=float, default=2400); ap.add_argument("--production", action="store_true"); ap.add_argument("--proposal", choices=["prior", "mixed"], default="prior"); ap.add_argument("--catalogue_fraction", type=float, default=0.7); ap.add_argument("--master_seed", type=int, default=None, help="default 11 (prior bank), 12 (catalogue-mixed bank) so the two banks never repeat events"); a = ap.parse_args()
a.master_seed = a.master_seed if a.master_seed is not None else (12 if a.proposal == "mixed" else 11)
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"; CODE = os.path.dirname(os.path.abspath(__file__))
tag = ("tde_fisher_bank" if a.production else "tde_fisher_bank_test") + ("_cat" if a.proposal == "mixed" else "")
dag = f"{DATA}/dag/{tag}"; logs = f"{DATA}/logs/{tag}"; out = f"{DATA}/simulations/{tag}"
for d in (dag, logs, out): os.makedirs(d, exist_ok=True)
PY = "/home/2673888s/.conda/envs/redback_few/bin/python"
def wrapper(fn, body): open(f"{dag}/{fn}", "w").write("#!/bin/bash\nexport MPLBACKEND=Agg\n" + body + "\n"); os.chmod(f"{dag}/{fn}", 0o755)
wrapper("bank.sh", f'{PY} {CODE}/tde_fisher_bank.py --survey "$1" --chunk "$2" --n_events {a.n_events} --max_wall_s {a.max_wall_s} --master_seed {a.master_seed} --proposal {a.proposal} --catalogue_fraction {a.catalogue_fraction} --out_dir {out}')
wrapper("summary.sh", f'{PY} {CODE}/summarize_tde_fisher.py --bank_dir {out} --tag {"prod" if a.production else "test"}{"_cat" if a.proposal == "mixed" else ""}')
sub = lambda ex, args, mem, name: (f"universe = vanilla\nexecutable = {dag}/{ex}\narguments = {args}\nrequest_cpus = 1\nrequest_memory = {mem}\n"
                                   f"getenv = True\noutput = {logs}/{name}.out\nerror = {logs}/{name}.err\nlog = {logs}/{tag}.log\nqueue\n")
open(f"{dag}/bank.sub", "w").write(sub("bank.sh", "$(survey) $(chunk)", "3GB", "bank_$(survey)_$(chunk)"))
open(f"{dag}/summary.sub", "w").write(sub("summary.sh", "", "4GB", "summary"))
nodes = []
with open(f"{dag}/{tag}.dag", "w") as f:
    for s in ("ZTF", "LSST"):
        for c in range(a.n_chunks):
            n = f"bank_{s}_{c:03d}"; nodes.append(n)
            f.write(f'JOB {n} {dag}/bank.sub\nVARS {n} survey="{s}" chunk="{c}"\nRETRY {n} 2\n')
    f.write(f"JOB summary {dag}/summary.sub\nPARENT {' '.join(nodes)} CHILD summary\n")
print(f"wrote {dag}/{tag}.dag  ({len(nodes)} bank nodes + 1 summary)\nsubmit (once!): condor_submit_dag {dag}/{tag}.dag\nresults: {logs}/summary.out, plots/detectability/tde_fisher/bank_*.png")
