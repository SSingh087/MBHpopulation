#!/usr/bin/env python
"""make_tde_inference_dag.py -- DAG for the TDE-only population inference at the FIDUCIAL Lambda, ONE realisation (same layout as the Paper-I run_inference.dag).

  mock_ZTF, mock_LSST (CPU)  ->  infer_ZTF, infer_LSST (CPU nessai job each, low priority); test_ZTF/LSST = likelihood pull/profile plots  ->  ppd_<survey> (CPU, caches that survey's PPD as soon as its inference ends) -> plots (violins + PPD figure from the cache)
The number of detected events per survey is Poisson around mu = Lambda_tot x alpha at the fiducial Lambda (ZTF 313.1, LSST 17893.7; Lambda_tot_tde = 160,798 from 8 simulator universes, alpha_soft from the flow grid).
No joint ZTF+LSST analysis: LSST has ~57x more events.  EMRI nodes can be added to the same tree later.
  python make_tde_inference_dag.py --seed 1 --nlive 1000       (writes files only; submit with condor_submit_dag)
"""
import argparse, os
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"; PY = "/home/2673888s/.conda/envs/redback_few/bin/python"
MU = {"ZTF": 313.13, "LSST": 17893.7}
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--nlive", type=int, default=1000); ap.add_argument("--priority", type=int, default=10); ap.add_argument("--threads", type=int, default=16); ap.add_argument("--gpu", action="store_true", help="run infer_* and plots on a GPU (LSST needs >=10 GB: the wiay 2080 Ti or any 3090/4090)"); a = ap.parse_args()
tag = f"tde_fid_mu_seed{a.seed}"; dd = f"{DATA}/dag/{tag}"; run = f"{DATA}/inference_runs/{tag}"; logs = f"{DATA}/logs/{tag}"; mk = f"{DATA}/simulations/inference_mocks"
for d in (dd, run, logs, mk): os.makedirs(d, exist_ok=True)
mock = lambda sv: f"{mk}/tde_{sv}_fid_mu_seed{a.seed}.npz"
dev = "cuda" if a.gpu else "cpu"
MEM = {"ZTF": "16GB", "LSST": "40GB"}      # LSST kernel matrix ~18k x 76,800 float32 = 5.5 GB, plus the flow pass and nessai
def sub(name, script, args, mem="16GB", cpus=None, gpu_mem=None):
    cpus = cpus or (2 if gpu_mem else a.threads)
    g = f"request_gpus = 1\nrequire_gpus = Capability >= 7.0 && Capability < 9.0 && GlobalMemoryMb >= {gpu_mem}\n" if gpu_mem else ""
    open(f"{dd}/{name}.sub", "w").write(f"universe = vanilla\nexecutable = {PY}\narguments = {HERE}/{script} {args}\nrequest_cpus = {cpus}\nrequest_memory = {mem}\n{g}priority = {a.priority}\ngetenv = True\n"
                                        f"environment = \"OMP_NUM_THREADS={cpus} MKL_NUM_THREADS={cpus} MALLOC_ARENA_MAX=4 KMP_STACKSIZE=8m\"\noutput = {logs}/{name}.out\nerror = {logs}/{name}.err\nlog = {logs}/{tag}.log\nqueue\n")
dag = []
for sv in ("ZTF", "LSST"):
    sub(f"mock_{sv}", "make_tde_mock.py", f"--survey {sv} --seed {a.seed} --mu {MU[sv]} --out {mock(sv)} --device cpu", MEM[sv], 4); dag.append(f"JOB mock_{sv} {dd}/mock_{sv}.sub")
    sub(f"test_{sv}", "test_tde_likelihood.py", f"--mock {mock(sv)} --tag {tag}_{sv} --device cpu", MEM[sv], 8); dag += [f"JOB test_{sv} {dd}/test_{sv}.sub", f"PARENT mock_{sv} CHILD test_{sv}"]
    sub(f"infer_{sv}", "run_nessai_tde.py", f"--mock {mock(sv)} --out {run}/{sv} --nlive {a.nlive} --seed {a.seed} --threads {2 if a.gpu else a.threads} --device {dev}", MEM[sv], gpu_mem=(10000 if sv == "LSST" else 4000) if a.gpu else None)
    dag += [f"JOB infer_{sv} {dd}/infer_{sv}.sub", f"PARENT mock_{sv} CHILD infer_{sv}", f"RETRY infer_{sv} 2"]
    sub(f"ppd_{sv}", "plot_tde_posteriors.py", f"--base {run} --nppd 300 --device cpu --cache_only {sv}", MEM[sv], cpus=8); dag += [f"JOB ppd_{sv} {dd}/ppd_{sv}.sub", f"PARENT infer_{sv} CHILD ppd_{sv}"]
sub("plots", "plot_tde_posteriors.py", f"--base {run} --nppd 300 --device cpu", "16GB", cpus=8); dag += [f"JOB plots {dd}/plots.sub", "PARENT ppd_ZTF ppd_LSST CHILD plots"]
open(f"{dd}/{tag}.dag", "w").write("\n".join(dag) + "\n"); print(f"wrote {dd}/{tag}.dag")
