"""Learning-curve DAG for the TDE Fisher-error flow: {ZTF,LSST} x {cfg 05 (winner), cfg 11 (small)} x train fractions -> plot.  Not submitted here.
Train split subsampled (same data_seed => same val/test); test split untouched."""
import os, stat, json
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"; HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/home/2673888s/.conda/envs/redback_few/bin/python"; name = "tde_fisher_lc"
dag, logs = f"{DATA}/dag/{name}", f"{DATA}/logs/{name}"; os.makedirs(dag, exist_ok=True); os.makedirs(logs, exist_ok=True)
CFG = {"05": "--n_layers 8 --hidden 256 --bins 16 --ctx_hidden 64 --batch 512 --lr 0.00104372 --seed 5",
       "11": "--n_layers 6 --hidden 64 --bins 12 --ctx_hidden 64 --batch 2048 --lr 0.0025773 --seed 11"}
FR = [0.1, 0.25, 0.5, 1.0]
def sh(fn, body):
    open(f"{dag}/{fn}", "w").write("#!/bin/bash\nexport OMP_NUM_THREADS=2 MKL_NUM_THREADS=2\ncd " + HERE + "\n" + body + "\n"); os.chmod(f"{dag}/{fn}", 0o755)
sh("run_lc.sh", f'case $2 in {" ".join(f"{k}) C=\'{v}\';;" for k, v in CFG.items())} esac\n{PY} train_tde_fisher_flow.py --survey $1 --epochs 300 --patience 25 --no_eval --train_frac $3 --out_name tde_fisher_lc/$1_$2_f$3 $C')
sh("run_plot.sh", f"{PY} learning_curve_tde_fisher.py")
def sub(n, ex, arg, gpu, mem):
    open(f"{dag}/{n}.sub", "w").write(f"universe = vanilla\nexecutable = {dag}/{ex}\narguments = {arg}\ngetenv = True\nlog = {logs}/{n}_$(tag).log\noutput = {logs}/{n}_$(tag).out\nerror = {logs}/{n}_$(tag).err\n"
        f"request_cpus = 2\nrequest_memory = {mem}\nrequest_disk = 2GB\n" + ("request_gpus = 1\nrequire_gpus = Capability >= 7.0 && Capability < 9.0\n" if gpu else "") + "queue 1\n")
sub("lc", "run_lc.sh", "$(survey) $(cfg) $(frac)", True, "8GB"); sub("plot", "run_plot.sh", "", False, "2GB")
L, nodes = [], []
for S in ["ZTF", "LSST"]:
    for c in CFG:
        for f in FR:
            n = f"lc_{S}_{c}_p{int(round(f*100))}"; nodes.append(n)
            L += [f"JOB {n} {dag}/lc.sub", f'VARS {n} survey="{S}" cfg="{c}" frac="{f}" tag="{S}_{c}_f{f}"', f"RETRY {n} 1"]
L += [f"JOB plot {dag}/plot.sub", 'VARS plot tag="plot"', f"PARENT {' '.join(nodes)} CHILD plot"]
open(f"{dag}/{name}.dag", "w").write("\n".join(L) + "\n")
print(f"wrote {dag}/{name}.dag: {len(nodes)} GPU trainings + plot\nsubmit ONCE: cd {dag} && condor_submit_dag {name}.dag\nresult: plots/detectability/tde_fisher_flow/learning_curve.png")
