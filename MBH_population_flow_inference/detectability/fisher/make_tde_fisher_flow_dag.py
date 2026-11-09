#!/usr/bin/env python
"""Write (do NOT submit) the DAG: smoke test -> hyper-parameter sweep (GPU, per survey) -> select winner -> evaluate winner on the test split.

    python make_tde_fisher_flow_dag.py --n_configs 12
    cd <DATA>/dag/tde_fisher_flow && condor_submit_dag tde_fisher_flow.dag        # submit ONCE

  smoke_ZTF (2 epochs, train+test eval, writes models/.../smoke_*)  -> PARENT of every sweep node
  train_<S>_<cfg> (S in ZTF, LSST; config 00 = baseline)           -> select_<S> (rank by VALIDATION nll, copy winner, sweep plots)
  select_<S> -> final_<S> (--eval_only on the untouched TEST split: test NLL, PIT calibration, sigma scatter plots)
Outputs: models/detectability/tde_fisher_sweep_<S>/cfg_XX, models/detectability/tde_fisher_flow_<S> (winner),
         plots/detectability/tde_fisher_flow/{sweep_<S>.png, tde_fisher_flow_<S>_diag.png}, logs/tde_fisher_flow/, logs/tde_fisher_flow_ranking_<S>.csv
"""
import argparse, json, os, stat
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"
PY = "/home/2673888s/.conda/envs/redback_few/bin/python"
ap = argparse.ArgumentParser()
ap.add_argument("--n_configs", type=int, default=12); ap.add_argument("--name", default="tde_fisher_flow"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="", help="suffix for model/plot names (e.g. _big) so existing winners are not overwritten"); ap.add_argument("--banks", nargs="+", default=["tde_fisher_bank", "tde_fisher_bank_cat"])
ap.add_argument("--batches", type=int, nargs="+", default=[512, 1024, 2048]); ap.add_argument("--base_batch", type=int, default=1024)
ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--patience", type=int, default=25)
args = ap.parse_args()
args.name = args.name + args.tag; TG = args.tag; BK = " ".join(args.banks)
dagdir, logdir = f"{DATA}/dag/{args.name}", f"{DATA}/logs/{args.name}"
os.makedirs(dagdir, exist_ok=True); os.makedirs(logdir, exist_ok=True)
SPACE = dict(n_layers=[4, 6, 8], hidden=[64, 128, 256], bins=[8, 12, 16], ctx_hidden=[64, 128], batch=args.batches)
BASE = dict(n_layers=6, hidden=128, bins=8, ctx_hidden=128, batch=args.base_batch, lr=1e-3)
rng = np.random.default_rng(args.seed)
cfgs, seen = [dict(id="00", **BASE, seed=0)], {tuple(BASE.values())}
while len(cfgs) < args.n_configs:
    c = {k: int(rng.choice(v)) for k, v in SPACE.items()}; c["lr"] = float(np.exp(rng.uniform(np.log(3e-4), np.log(3e-3))))
    key = tuple(c[k] for k in BASE)
    if key in seen: continue
    seen.add(key); cfgs.append(dict(id=f"{len(cfgs):02d}", **c, seed=len(cfgs)))
json.dump(cfgs, open(f"{dagdir}/configs.json", "w"), indent=1)

def script(name, body):
    p = f"{dagdir}/{name}"; open(p, "w").write("#!/bin/bash\nexport OMP_NUM_THREADS=2 MKL_NUM_THREADS=2\ncd " + HERE + "\n" + body + "\n")
    os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH); return p
T = f"{PY} train_tde_fisher_flow.py"
sh_train = script("run_train.sh", f'CFG=$(case $2 in {" ".join(f"{c["id"]}) echo --n_layers {c["n_layers"]} --hidden {c["hidden"]} --bins {c["bins"]} --ctx_hidden {c["ctx_hidden"]} --batch {c["batch"]} --lr {c["lr"]:.6g} --seed {c["seed"]};;" for c in cfgs)} esac)\n'
                  f'{T} --survey $1 --epochs {args.epochs} --patience {args.patience} --no_eval --banks {BK} --out_name tde_fisher_sweep{TG}_$1/cfg_$2 $CFG')
sh_smoke = script("run_smoke.sh", f'{T} --survey ZTF --epochs 2 --banks {BK} --out_name smoke_tde_fisher_flow{TG}_ZTF && {T} --survey ZTF --eval_only --banks {BK} --out_name smoke_tde_fisher_flow{TG}_ZTF')
sh_select = script("run_select.sh", f'{PY} select_tde_fisher_flow.py --survey $1 --tag {TG}')
sh_final = script("run_final.sh", f'{T} --survey $1 --eval_only --banks {BK} --out_name tde_fisher_flow{TG}_$1')
def sub(name, exe, arg, gpu, mem):
    open(f"{dagdir}/{name}.sub", "w").write(f"""universe = vanilla
executable = {exe}
arguments = {arg}
getenv = True
log = {logdir}/{name}_$(tag).log
output = {logdir}/{name}_$(tag).out
error = {logdir}/{name}_$(tag).err
request_cpus = 2
request_memory = {mem}
request_disk = 2GB
{"request_gpus = 1" + chr(10) + "require_gpus = Capability >= 7.0 && Capability < 9.0" if gpu else ""}
queue 1
""")
sub("train", sh_train, "$(survey) $(cfg)", True, "8GB"); sub("smoke", sh_smoke, "", True, "8GB")
sub("select", sh_select, "$(survey)", False, "4GB"); sub("final", sh_final, "$(survey)", True, "8GB")
L = [f"JOB smoke_ZTF {dagdir}/smoke.sub", 'VARS smoke_ZTF tag="ZTF"']
for S in ["ZTF", "LSST"]:
    names = []
    for c in cfgs:
        n = f"train_{S}_{c['id']}"; names.append(n)
        L += [f"JOB {n} {dagdir}/train.sub", f'VARS {n} survey="{S}" cfg="{c["id"]}" tag="{S}_{c["id"]}"', f"RETRY {n} 1", f"PARENT smoke_ZTF CHILD {n}"]
    L += [f"JOB select_{S} {dagdir}/select.sub", f'VARS select_{S} survey="{S}" tag="{S}"', f"PARENT {' '.join(names)} CHILD select_{S}",
          f"JOB final_{S} {dagdir}/final.sub", f'VARS final_{S} survey="{S}" tag="{S}"', f"PARENT select_{S} CHILD final_{S}"]
open(f"{dagdir}/{args.name}.dag", "w").write("\n".join(L) + "\n")
print(f"wrote {dagdir}/{args.name}.dag: 1 smoke + 2 x {args.n_configs} GPU trainings + 2 select (CPU) + 2 final (GPU)\nsubmit ONCE: cd {dagdir} && condor_submit_dag {args.name}.dag")
