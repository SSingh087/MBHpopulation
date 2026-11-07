"""make_emri_fisher_bank_dag.py -- EMRI Fisher bank, production DAG (not submitted here).
  fish_000..fish_099 (GPU, independent; each seeded by (master_seed, chunk); split by EVENTS only; 20 events each)  -->  summary (PARENT = every fish node, nothing else)
Settings: Fisher SNR fixed at 100 (errors are SNR-normalised, sigma*rho, and rescaled with the SNR network at application time), C2 re-evaluation at a noisy
estimate, catalogue-mixed proposal (70% of events built from source-catalogue EMRIs), use_rk4=False, normalised-inverse covariance (fisher_cov.py).
GPU: any card with compute capability 7-9 (2080 Ti, 3090, 4090); no memory restriction -- cupy's pool is released after every event and a failed event is
recorded as a flagged row, the node is retried twice.  Outputs: simulations/emri_fisher_bank/chunk_*.npz, logs/emri_fisher_bank/, plots/detectability/emri_fisher/"""
import os, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--n_chunks", type=int, default=100); ap.add_argument("--n_events", type=int, default=20)
ap.add_argument("--target_snr", type=float, default=100.0); ap.add_argument("--catalogue_fraction", type=float, default=0.7)
ap.add_argument("--master_seed", type=int, default=31, help="31: distinct from the tests (21) so no test event is repeated")
ap.add_argument("--max_wall_s", type=float, default=14400); a = ap.parse_args()
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"; CODE = os.path.dirname(os.path.abspath(__file__))
name = "emri_fisher_bank"; dag = f"{DATA}/dag/{name}"; logs = f"{DATA}/logs/{name}"; out = f"{DATA}/simulations/{name}"
for d in (dag, logs, out): os.makedirs(d, exist_ok=True)
PY = "/home/2673888s/.conda/envs/redback_few/bin/python"
def wrapper(fn, body): open(f"{dag}/{fn}", "w").write("#!/bin/bash\nexport PYTHONPATH=/home/2673888s/poplar\nexport MPLBACKEND=Agg\n" + body + "\n"); os.chmod(f"{dag}/{fn}", 0o755)
wrapper("fish.sh", f'{PY} {CODE}/emri_fisher_bank.py --chunk "$1" --n_events {a.n_events} --target_snr {a.target_snr} --proposal mixed --catalogue_fraction {a.catalogue_fraction} '
                   f'--master_seed {a.master_seed} --max_wall_s {a.max_wall_s} --traj_dir {DATA}/simulations/p0e0_traj --out_dir {out}')
wrapper("summary.sh", f'{PY} {CODE}/summarize_emri_fisher.py --bank_dir {out} --tag bank')
GPUREQ = "require_gpus = Capability >= 7.0 && Capability < 9.0\n"     # the RTX 5090 (cc 12.0) is unsupported by the installed cupy/torch
sub = lambda ex, args, cpus, mem, gpus, tag: (f"universe = vanilla\nexecutable = {dag}/{ex}\narguments = {args}\nrequest_cpus = {cpus}\nrequest_memory = {mem}\nrequest_gpus = {gpus}\n"
                                              + (GPUREQ if gpus else "") + f"getenv = True\noutput = {logs}/{tag}.out\nerror = {logs}/{tag}.err\nlog = {logs}/{name}.log\nqueue\n")
open(f"{dag}/fish.sub", "w").write(sub("fish.sh", "$(chunk)", 2, "12GB", 1, "fish_$(chunk)"))
open(f"{dag}/summary.sub", "w").write(sub("summary.sh", "", 1, "4GB", 0, "summary"))
nodes = []
with open(f"{dag}/{name}.dag", "w") as f:
    for c in range(a.n_chunks):
        n = f"fish_{c:03d}"; nodes.append(n); f.write(f'JOB {n} {dag}/fish.sub\nVARS {n} chunk="{c}"\nRETRY {n} 2\n')
    f.write(f"JOB summary {dag}/summary.sub\nPARENT {' '.join(nodes)} CHILD summary\n")
print(f"wrote {dag}/{name}.dag  ({a.n_chunks} GPU nodes x {a.n_events} events = {a.n_chunks * a.n_events} events + 1 summary)\nsubmit (once, no -f): condor_submit_dag {dag}/{name}.dag\nresults: {logs}/summary.out, plots/detectability/emri_fisher/bank_*.png")
