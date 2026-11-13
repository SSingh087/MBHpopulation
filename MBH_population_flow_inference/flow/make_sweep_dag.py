#!/usr/bin/env python
"""
Write (do NOT submit) the HTCondor DAG for a hyperparameter sweep of the EMRI and TDE flows.

    python make_sweep_dag.py --n_configs 12
    cd <DATA>/dag/flow_sweep && condor_submit_dag flow_sweep.dag

DAG:
    train_<pop>_<cfg>  (one GPU job per hyperparameter configuration; cfg 00 = baseline)
        -> select_<pop>   : ranks the configs, writes sweep plots, copies the winner to models/flow_<pop>_best/ and plots its diagnostics
    heldout_<lambda>      : (CPU) simulate fresh held-out universes at each acceptance-test Lambda
        -> acceptance     : (after both selects + all heldout) winner flows vs held-out simulations, corner/marginal/KS plots

Everything written under /data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference/ :
    dag/<name>/        DAG, .sub files, wrapper scripts, sweep_<pop>.json (the sampled configurations)
    logs/<name>/       condor logs;   logs/flow/ ranking csv, acceptance json
    models/sweep_<pop>/cfg_XX/ (all runs)   models/flow_<pop>_best/ (winner)
    plots/sweep_<pop>/ (comparison plots)   plots/flow_<pop>_best/ (winner diagnostics)   plots/acceptance_best/
"""
import argparse, json, os, stat
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA0 = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"
PY = "/home/2673888s/.conda/envs/redback_few/bin/python"
TEST_LAMBDA_NAMES = ["fiducial", "heavy_corner", "light_corner", "highA_steep", "lowA_shallow"]     # keep in sync with acceptance.py

SPACE = dict(n_layers=[4, 6, 8, 10], hidden=[32, 64, 128, 256], bins=[6, 8, 12, 16], ctx_hidden=[32, 64, 128], batch=[2048, 4096, 8192])
LR_RANGE = (3e-4, 3e-3)                                                                              # log-uniform
BASELINE = dict(n_layers=6, hidden=64, bins=8, ctx_hidden=64, lr=1e-3, batch=4096)

ap = argparse.ArgumentParser()
ap.add_argument("--n_configs", type=int, default=12, help="configurations per population (config 00 is the baseline)")
ap.add_argument("--pops", nargs="+", default=["emri", "tde"])
ap.add_argument("--name", default="flow_sweep")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n_univ_heldout", type=int, default=20)
ap.add_argument("--emri_epochs", type=int, default=200)
ap.add_argument("--emri_steps", type=int, default=300)
ap.add_argument("--tde_epochs", type=int, default=120)
ap.add_argument("--tde_steps", type=int, default=150)
ap.add_argument("--patience", type=int, default=20)
ap.add_argument("--scenario", default=None, help="organised run under DATA/scenarios/<scenario>/ (needs --spec)")
ap.add_argument("--spec", default=None, help="nucleation,occupation,alpha_Re e.g. hannah,varying,0.75 (for the held-out simulations)")
args = ap.parse_args()
assert not args.scenario or args.spec, "--scenario needs --spec nucleation,occupation,alpha_Re"
DATA = f"{DATA0}/scenarios/{args.scenario}" if args.scenario else DATA0
ENV = (f"export EMRI_SCENARIO_ROOT={DATA}\n" + (f"export EMRI_SCENARIO_SPEC={args.spec}\n" if args.spec else "")) if args.scenario else ""

dagdir, logdir = f"{DATA}/dag/{args.name}", f"{DATA}/logs/{args.name}"
for d in (dagdir, logdir, f"{DATA}/logs/flow"): os.makedirs(d, exist_ok=True)
rng = np.random.default_rng(args.seed)

# ---- sample configurations (random search; unique; config 00 = baseline)
sweeps = {}
for pop in args.pops:
    cfgs, seen = [dict(id="00", **BASELINE, seed=0)], {tuple(BASELINE.values())}
    while len(cfgs) < args.n_configs:
        c = {k: int(rng.choice(v)) for k, v in SPACE.items()}
        c["lr"] = float(np.exp(rng.uniform(np.log(LR_RANGE[0]), np.log(LR_RANGE[1]))))
        key = tuple(c[k] for k in BASELINE)
        if key in seen: continue
        seen.add(key); cfgs.append(dict(id=f"{len(cfgs):02d}", **c, seed=len(cfgs)))
    train = dict(epochs=args.emri_epochs if pop == "emri" else args.tde_epochs,
                 max_steps_per_epoch=args.emri_steps if pop == "emri" else args.tde_steps, patience=args.patience)
    sweeps[pop] = dict(configs=cfgs, train=train)
    json.dump(sweeps[pop], open(f"{dagdir}/sweep_{pop}.json", "w"), indent=1)

def script(name, body):
    p = f"{dagdir}/{name}"
    open(p, "w").write("#!/bin/bash\nexport OMP_NUM_THREADS=2 MKL_NUM_THREADS=2\n" + ENV + "cd " + HERE + "\n" + body + "\n")
    os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH); return p

sh_train = script("run_train.sh", f'{PY} train_flow.py --pop $1 --config_json {dagdir}/sweep_$1.json --config_id $2 --out_name sweep_$1/cfg_$2 --no_diag --skip_done')
sh_select = script("run_select.sh", f'{PY} select_best.py --pop $1')
sh_held = script("run_heldout.sh", f'{PY} acceptance.py --sim_only $1 --n_univ {args.n_univ_heldout} --workers 8')
sh_acc = script("run_acceptance.sh", f'{PY} acceptance.py --tag _best --n_univ {args.n_univ_heldout}')

def sub(name, exe, arg, gpu, cpus, mem, disk="2GB"):
    open(f"{dagdir}/{name}.sub", "w").write(f"""executable     = {exe}
arguments      = {arg}
getenv         = true
log            = {logdir}/{name}_$(tag).log
output         = {logdir}/{name}_$(tag).out
error          = {logdir}/{name}_$(tag).err
request_cpus   = {cpus}
request_memory = {mem}
request_disk   = {disk}
{"request_GPUs   = 1" if gpu else ""}
notify_user    = s.singh.3@research.gla.ac.uk
notification   = Error
queue 1
""")
sub("train", sh_train, "$(pop) $(cfg)", True, 2, "10GB")
sub("select", sh_select, "$(pop)", True, 2, "8GB")
sub("heldout", sh_held, "$(lam)", False, 8, "8GB")
sub("acceptance", sh_acc, "", True, 2, "8GB")

lines = []
for pop in args.pops:
    names = []
    for c in sweeps[pop]["configs"]:
        n = f"train_{pop}_{c['id']}"; names.append(n)
        lines += [f"JOB {n} {dagdir}/train.sub", f'VARS {n} pop="{pop}" cfg="{c["id"]}" tag="{pop}_{c["id"]}"', f"RETRY {n} 1"]
    lines += [f"JOB select_{pop} {dagdir}/select.sub", f'VARS select_{pop} pop="{pop}" tag="{pop}"', f"RETRY select_{pop} 1",
              f"PARENT {' '.join(names)} CHILD select_{pop}"]
held = []
for l in TEST_LAMBDA_NAMES:
    n = f"heldout_{l}"; held.append(n)
    lines += [f"JOB {n} {dagdir}/heldout.sub", f'VARS {n} lam="{l}" tag="{l}"', f"RETRY {n} 1"]
lines += [f"JOB acceptance {dagdir}/acceptance.sub", 'VARS acceptance tag="all"', "RETRY acceptance 1",
          f"PARENT {' '.join('select_' + p for p in args.pops)} {' '.join(held)} CHILD acceptance"]
open(f"{dagdir}/{args.name}.dag", "w").write("\n".join(lines) + "\n")

print(f"""wrote {dagdir}/{args.name}.dag : {len(args.pops)} populations x {args.n_configs} configs = {len(args.pops)*args.n_configs} GPU training jobs,
  + {len(args.pops)} select jobs, {len(held)} held-out simulation jobs (8 CPUs), 1 acceptance job
configs  : {dagdir}/sweep_<pop>.json   (random search over {list(SPACE)} + lr; config 00 = baseline)
estimate (baseline config, measured): EMRI ~8 s/epoch, TDE ~13 s/epoch (150 steps); larger configs are slower.
          Worst case per job at the baseline size: EMRI ~{args.emri_epochs*8/60:.0f} min, TDE ~{args.tde_epochs*13/60:.0f} min (early stopping usually ends sooner).
held-out simulation: {args.n_univ_heldout} universes x ~6 s per Lambda on 8 CPUs.
submit   : cd {dagdir} && condor_submit_dag {args.name}.dag
results  : plots/sweep_<pop>/, plots/flow_<pop>_best/, plots/acceptance_best/ ; winner model: models/flow_<pop>_best/best.pt""")
