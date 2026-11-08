"""make_tde_bank_scale_dag.py -- LARGE TDE Fisher bank (bank 3) for the Fisher-error flows.  Not submitted here.
Same generator/proposal as bank 2 (catalogue-mixed, 70% catalogue / 30% wide prior), new master seed 13.
Parallel over EVENTS only: node bank_<S>_<c> = one multi-core job (request_cpus = --cpus, a worker pool over its events) of --n_events attempted events, seeded by (master_seed, chunk, survey).
SCALING: only the chunk index changes, so the bank is extended by a further DAG ("wave") with --chunk_start = previous chunk_start + n_chunks,
the SAME master seed and out dir -> events never repeat; the summary and the trainers read every chunk file in the dir.
  wave 1:  python make_tde_bank_scale_dag.py --chunk_start 0    --n_chunks 340             (~250k usable per survey)
  wave 2+: python make_tde_bank_scale_dag.py --chunk_start 340 --n_chunks 340 --wave 2    (and so on up to ~1M)
Yield ~0.37-0.40 usable per attempted event; ZTF ~9 min / 2000-event chunk on 8 cpus, LSST ~25 min.  Submit each DAG ONCE."""
import os, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--chunk_start", type=int, default=0); ap.add_argument("--n_chunks", type=int, default=340); ap.add_argument("--n_events", type=int, default=2000); ap.add_argument("--cpus", type=int, default=8, help="request_cpus = pool workers inside each job")
ap.add_argument("--wave", type=int, default=1); ap.add_argument("--master_seed", type=int, default=13); ap.add_argument("--max_wall_s", type=float, default=7200)
ap.add_argument("--catalogue_fraction", type=float, default=0.7); ap.add_argument("--surveys", nargs="+", default=["ZTF", "LSST"]); a = ap.parse_args()
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"; CODE = os.path.dirname(os.path.abspath(__file__))
bank = "tde_fisher_bank_big"; tag = f"{bank}_w{a.wave}"
dag, logs, out = f"{DATA}/dag/{tag}", f"{DATA}/logs/{tag}", f"{DATA}/simulations/{bank}"
for d in (dag, logs, out): os.makedirs(d, exist_ok=True)
PY = "/home/2673888s/.conda/envs/redback_few/bin/python"
def wrapper(fn, body):
    open(f"{dag}/{fn}", "w").write("#!/bin/bash\nexport MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n" + body + "\n"); os.chmod(f"{dag}/{fn}", 0o755)
wrapper("bank.sh", f'{PY} {CODE}/tde_fisher_bank.py --survey "$1" --chunk "$2" --n_events {a.n_events} --max_wall_s {a.max_wall_s} --workers {a.cpus} --master_seed {a.master_seed} --proposal mixed --catalogue_fraction {a.catalogue_fraction} --out_dir {out}')
wrapper("summary.sh", f'{PY} {CODE}/summarize_tde_fisher.py --bank_dir {out} --tag big_w{a.wave}')
sub = lambda ex, args, mem, name, cpus=1: (f"universe = vanilla\nexecutable = {dag}/{ex}\narguments = {args}\nrequest_cpus = {cpus}\nrequest_memory = {mem}\n"
                                   f"getenv = True\noutput = {logs}/{name}.out\nerror = {logs}/{name}.err\nlog = {logs}/{tag}.log\nqueue\n")
open(f"{dag}/bank.sub", "w").write(sub("bank.sh", "$(survey) $(chunk)", f"{3*a.cpus}GB", "bank_$(survey)_$(chunk)", a.cpus))
open(f"{dag}/summary.sub", "w").write(sub("summary.sh", "", "8GB", "summary"))
nodes = []
with open(f"{dag}/{tag}.dag", "w") as f:
    for c in range(a.chunk_start, a.chunk_start + a.n_chunks):
        for s in a.surveys:                       # interleave surveys so both finish together
            n = f"bank_{s}_{c:05d}"; nodes.append(n)
            f.write(f'JOB {n} {dag}/bank.sub\nVARS {n} survey="{s}" chunk="{c}"\nRETRY {n} 2\n')
    f.write(f"JOB summary {dag}/summary.sub\nPARENT {' '.join(nodes)} CHILD summary\n")
n_att = a.n_chunks * a.n_events
print(f"wrote {dag}/{tag}.dag: {len(nodes)} {a.cpus}-cpu nodes ({a.n_chunks} chunks x {len(a.surveys)} surveys x {a.n_events} attempted events) + summary\n"
      f"expected usable per survey ~{int(0.38*n_att)}; core-hours ZTF ~{a.n_chunks*9*a.cpus/60:.0f}, LSST ~{a.n_chunks*25*a.cpus/60:.0f}\n"
      f"submit ONCE: cd {dag} && condor_submit_dag -maxidle 600 {tag}.dag\nout: {out}  logs: {logs}")
