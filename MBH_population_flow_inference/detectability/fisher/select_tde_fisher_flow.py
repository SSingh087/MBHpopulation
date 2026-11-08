#!/usr/bin/env python
"""Rank the sweep configs of the TDE Fisher-error flow by VALIDATION nll (test split untouched), copy the winner to
models/detectability/tde_fisher_flow_<survey>/, write ranking.csv and comparison plots."""
import argparse, glob, json, os, shutil
import numpy as np
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"
ap = argparse.ArgumentParser(); ap.add_argument("--survey", required=True); args = ap.parse_args()
S = args.survey
sweep = f"{DATA}/models/detectability/tde_fisher_sweep_{S}"; pdir = f"{DATA}/plots/detectability/tde_fisher_flow"; os.makedirs(pdir, exist_ok=True)
rows = []
for d in sorted(glob.glob(f"{sweep}/cfg_*")):
    f = f"{d}/summary.json"
    if not os.path.exists(f): print("missing", d); continue
    s = json.load(open(f)); h = s["hyper"]
    rows.append(dict(id=os.path.basename(d)[4:], val=s["best_val"], ep=s["best_epoch"], epochs=s["epochs_run"], n_params=s["n_params"], wall_min=s["wall_s"] / 60,
                     n_layers=h["n_layers"], hidden=h["hidden"], bins=h["K"], ctx_hidden=h["ctx_hidden"], lr=s["lr"], batch=s["batch"], dir=d))
rows = [r for r in rows if np.isfinite(r["val"])]; rows.sort(key=lambda r: r["val"])
with open(f"{DATA}/logs/tde_fisher_flow_ranking_{S}.csv", "w") as fh:
    keys = [k for k in rows[0] if k != "dir"]; fh.write(",".join(keys) + "\n")
    for r in rows: fh.write(",".join(str(r[k]) for k in keys) + "\n")
for r in rows: print(f"cfg {r['id']}  val {r['val']:.4f}  ep {r['ep']}/{r['epochs']}  L{r['n_layers']} H{r['hidden']} K{r['bins']} C{r['ctx_hidden']} lr {r['lr']:.1e} b{r['batch']}  {r['n_params']} params {r['wall_min']:.1f} min")
win = rows[0]; dst = f"{DATA}/models/detectability/tde_fisher_flow_{S}"
if os.path.exists(dst): shutil.rmtree(dst)
shutil.copytree(win["dir"], dst); print("winner", win["id"], "->", dst)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 3, figsize=(17, 9))
a = ax[0, 0]; a.bar(range(len(rows)), [r["val"] for r in rows], color=["C3" if i == 0 else "C0" for i in range(len(rows))])
a.set_xticks(range(len(rows))); a.set_xticklabels([r["id"] for r in rows]); a.set_ylim(rows[0]["val"] - 0.1, rows[-1]["val"] + 0.1); a.set_ylabel("best val NLL"); a.set_title("configs ranked (red = winner)")
for r in rows:
    h = np.loadtxt(f"{r['dir']}/history.csv", delimiter=",").reshape(-1, 3)
    ax[0, 1].plot(h[:, 0], h[:, 2], lw=2.5 if r is win else 0.8, color="C3" if r is win else None, label=r["id"] if r is win else None)
ax[0, 1].set_ylim(win["val"] - 0.3, win["val"] + 3); ax[0, 1].set_xlabel("epoch"); ax[0, 1].set_ylabel("val NLL"); ax[0, 1].legend()
ax[0, 2].scatter([r["n_params"] for r in rows], [r["val"] for r in rows]); ax[0, 2].set_xscale("log"); ax[0, 2].set_xlabel("parameters"); ax[0, 2].set_ylabel("val NLL")
for a, k in zip(ax[1], ["lr", "n_layers", "hidden"]):
    a.scatter([r[k] for r in rows], [r["val"] for r in rows]); a.set_xlabel(k); a.set_ylabel("val NLL")
    if k == "lr": a.set_xscale("log")
fig.suptitle(f"TDE Fisher-error flow sweep, {S}"); fig.tight_layout(); fig.savefig(f"{pdir}/sweep_{S}.png", dpi=120)
