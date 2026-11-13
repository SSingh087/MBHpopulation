"""make_sf_validation_dag.py -- ONE DAG for the selection-function (SF) validation (EMRI + TDE).  Not submitted here.

  python make_sf_validation_dag.py ; cd <DATA>/dag/sf_validation && condor_submit_dag sf_validation.dag     (submit ONCE; never -f)

DAG (all nodes run sf_stage.py <stage>):
  resid            (CPU)  residual model of the SNR network from the 57k OUT-OF-SAMPLE injections: r = log10(rho_net/rho_true) per (lgM, e0, rho_net) cell
    -> table       (GPU)  2^19 fixed points drawn from a mixture of flow densities (+10% box), P_naive and P_calibrated at each point (K nuisance draws)
         -> zbreak        (GPU)  alpha for 6 Lambda: naive vs calibrated, share of alpha by z / lgM / e_plunge   [where does the SF come from?]
         -> brute         (GPU)  1e8-sample brute force at 2 Lambda vs fixed points (bias at the 0.15% level)
         -> mu            (GPU)  mu = Lambda_tot*alpha over 300 prior universes: expected EMRI detections (sanity vs the old paper)
         -> flowsim_0..9  (GPU)  alpha for all 10,000 universes (1000 per node)
              -> flowsim_merge (GPU)  sum of P over the SIMULATED sources vs sum Lambda_tot*alpha_flow: does the flow's low-z tail carry the right mass?
  tde              (GPU)  TDE 2-D grid SF (classifier P_cut / P_soft, ZTF z<0.5, LSST z<4): alpha for universes, flow mass at z<zmax and detected-count tail test vs the simulator
  report           collects the JSONs (PARENT of everything)
Outputs: <DATA>/simulations/sf/*.json, logs/sf_validation/<stage>.out (unbuffered; watch with tail -f)."""
import argparse, os
ap = argparse.ArgumentParser(); ap.add_argument("--n_chunks", type=int, default=10); ap.add_argument("--priority", type=int, default=15); a = ap.parse_args()
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"; HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/home/2673888s/.conda/envs/redback_few/bin/python"; name = "sf_validation"
dag, logs = f"{DATA}/dag/{name}", f"{DATA}/logs/{name}"; os.makedirs(dag, exist_ok=True); os.makedirs(logs, exist_ok=True); os.makedirs(f"{DATA}/simulations/sf", exist_ok=True)
open(f"{dag}/run.sh", "w").write(f"#!/bin/bash\nexport OMP_NUM_THREADS=2 MKL_NUM_THREADS=2\ncd {HERE}\n{PY} -u -W ignore sf_stage.py \"$@\"\n"); os.chmod(f"{dag}/run.sh", 0o755)
def sub(tag, gpu, mem, cpus):
    open(f"{dag}/{tag}.sub", "w").write(f"universe = vanilla\nexecutable = {dag}/run.sh\narguments = $(args)\ngetenv = True\nlog = {logs}/{name}.log\noutput = {logs}/$(node).out\nerror = {logs}/$(node).err\n"
        f"request_cpus = {cpus}\nrequest_memory = {mem}\nrequest_disk = 2GB\npriority = {a.priority}\n" + ("request_gpus = 1\nrequire_gpus = Capability >= 7.0 && Capability < 9.0\n" if gpu else "") + "queue 1\n")
sub("cpu", False, "8GB", 4); sub("gpu", True, "16GB", 2)
L, nodes = [], []
def node(n, kind, args, parents=()):
    L.extend([f"JOB {n} {dag}/{kind}.sub", f'VARS {n} node="{n}" args="{args}"', f"RETRY {n} 1", f"PRIORITY {n} {a.priority}"]); nodes.append(n)
    if parents: L.append(f"PARENT {' '.join(parents)} CHILD {n}")
node("resid", "cpu", "resid")
node("table", "gpu", "table", ["resid"])
for st in ("zbreak", "brute", "mu"): node(st, "gpu", st, ["table"])
fs = []
for k in range(a.n_chunks): n = f"flowsim_{k}"; node(n, "gpu", f"flowsim --chunk {k} --n_chunks {a.n_chunks}", ["table"]); fs.append(n)
node("flowsim_merge", "gpu", "flowsim_merge", fs)
node("tde", "gpu", "tde")
node("report", "cpu", "report", [n for n in nodes if n != "report"])
open(f"{dag}/{name}.dag", "w").write("\n".join(L) + "\n")
print(f"wrote {dag}/{name}.dag: {len(nodes)} nodes (GPU: table, zbreak, brute, mu, {a.n_chunks} flowsim, flowsim_merge, tde)\nsubmit ONCE: cd {dag} && condor_submit_dag {name}.dag\nwatch: tail -f {logs}/<node>.out   results: {DATA}/simulations/sf/*.json and logs/{name}/report.out")
