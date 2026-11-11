#!/usr/bin/env python
"""make_tde_inference_dag.py -- DAG for the TDE-only population inference at the FIDUCIAL Lambda, ONE realisation (same layout as the Paper-I run_inference.dag).

  mock_ZTF, mock_LSST (CPU)  ->  infer_ZTF, infer_LSST (one GPU nessai job each)  ->  plots (violins + cached PPDs)
The number of detected events per survey is Poisson around mu = Lambda_tot x alpha at the fiducial Lambda (ZTF 313.1, LSST 17893.7; Lambda_tot_tde = 160,798 from 8 simulator universes, alpha_soft from the flow grid).
No joint ZTF+LSST analysis: LSST has ~57x more events.  EMRI nodes can be added to the same tree later.
  python make_tde_inference_dag.py --seed 1 --nlive 1000       (writes files only; submit with condor_submit_dag)
"""
import argparse, os
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"; PY = "/home/2673888s/.conda/envs/redback_few/bin/python"
MU = {"ZTF": 313.13, "LSST": 17893.7}
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--nlive", type=int, default=1000); ap.add_argument("--priority", type=int, default=0); a = ap.parse_args()
tag = f"tde_fid_mu_seed{a.seed}"; dd = f"{DATA}/dag/{tag}"; run = f"{DATA}/inference_runs/{tag}"; logs = f"{DATA}/logs/{tag}"; mk = f"{DATA}/simulations/inference_mocks"
for d in (dd, run, logs, mk): os.makedirs(d, exist_ok=True)
mock = lambda sv: f"{mk}/tde_{sv}_fid_mu_seed{a.seed}.npz"
def sub(name, script, args, gpu_mem=None):
    res = "request_cpus = 2\n" + ("" if gpu_mem is None else f"request_gpus = 1\nrequire_gpus = Capability >= 7.0 && Capability < 9.0 && GlobalMemoryMb >= {gpu_mem}\n")
    open(f"{dd}/{name}.sub", "w").write(f"universe = vanilla\nexecutable = {PY}\narguments = {HERE}/{script} {args}\n{res}request_memory = 24GB\npriority = {a.priority}\ngetenv = True\n"
                                        f"environment = \"OMP_NUM_THREADS=2\"\noutput = {logs}/{name}.out\nerror = {logs}/{name}.err\nlog = {logs}/{tag}.log\nqueue\n")
dag = []
for sv in ("ZTF", "LSST"):
    sub(f"mock_{sv}", "make_tde_mock.py", f"--survey {sv} --seed {a.seed} --mu {MU[sv]} --out {mock(sv)} --device cpu"); dag.append(f"JOB mock_{sv} {dd}/mock_{sv}.sub")
    sub(f"infer_{sv}", "run_nessai_tde.py", f"--mock {mock(sv)} --out {run}/{sv} --nlive {a.nlive} --seed {a.seed} --threads 2 --device cuda", 10000 if sv == "LSST" else 4000)   # LSST kernel matrix ~5.5 GB + flow
    dag += [f"JOB infer_{sv} {dd}/infer_{sv}.sub", f"PARENT mock_{sv} CHILD infer_{sv}", f"RETRY infer_{sv} 2"]
sub("plots", "plot_tde_posteriors.py", f"--base {run} --nppd 300 --device cuda", 10000); dag += [f"JOB plots {dd}/plots.sub", "PARENT infer_ZTF infer_LSST CHILD plots"]
open(f"{dd}/{tag}.dag", "w").write("\n".join(dag) + "\n"); print(f"wrote {dd}/{tag}.dag")
