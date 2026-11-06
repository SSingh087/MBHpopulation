#!/usr/bin/env python
"""
Pick the best flow of a hyperparameter sweep (lowest validation NLL) and produce the sweep-comparison plots.

  python select_best.py --pop emri
Reads  <DATA>/models/sweep_<pop>/cfg_*/{summary.json,history.csv}
Writes <DATA>/models/flow_<pop>_best/{best.pt,best_config.json,history.csv}   (copy of the winner; used by acceptance.py --tag _best)
       <DATA>/plots/sweep_<pop>/*.png            (curves, NLL vs hyperparameters, NLL vs cost, ranking)
       <DATA>/plots/flow_<pop>_best/*.png        (loss curves + validation diagnostics of the winner)
       <DATA>/logs/flow/sweep_<pop>_ranking.csv
Note: the winner is chosen on the validation universes, so its validation NLL is slightly optimistic;
acceptance.py then tests it on freshly simulated universes that no model has seen.
"""
import argparse, csv, glob, json, os, shutil, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, POPS
import plotting

ap = argparse.ArgumentParser()
ap.add_argument("--pop", choices=list(POPS), required=True)
ap.add_argument("--sweep", default=None, help="sweep folder name under models/ (default sweep_<pop>)")
ap.add_argument("--best", default=None, help="folder name for the winner (default flow_<pop>_best)")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()
pop = args.pop
sweep_name, best_name = args.sweep or f"sweep_{pop}", args.best or f"flow_{pop}_best"
sdir, pdir = f"{DATA}/models/{sweep_name}", f"{DATA}/plots/{sweep_name}"
bdir, bpdir = f"{DATA}/models/{best_name}", f"{DATA}/plots/{best_name}"
for d in (pdir, bdir, bpdir, f"{DATA}/logs/flow"): os.makedirs(d, exist_ok=True)

runs = []
for f in sorted(glob.glob(f"{sdir}/cfg_*/summary.json")):
    s = json.load(open(f)); s["dir"] = os.path.dirname(f)
    h = np.genfromtxt(f"{s['dir']}/history.csv", delimiter=",", names=True); s["hist"] = {k: np.atleast_1d(h[k]) for k in h.dtype.names}
    runs.append(s)
missing = sorted(set(os.path.basename(os.path.dirname(f)) for f in glob.glob(f"{sdir}/cfg_*/")) - set(os.path.basename(r["dir"]) for r in runs))
if not runs: sys.exit(f"no finished runs in {sdir}")
good = [r for r in runs if np.isfinite(r["best_val"])]
good.sort(key=lambda r: r["best_val"])
best = good[0]
print(f"{len(runs)} finished runs ({len(missing)} without summary: {missing}); best = cfg {best['config_id']}  val NLL {best['best_val']:.4f}")

# ---------- ranking table
cols = ["rank", "config_id", "best_val", "best_epoch", "epochs_run", "final_train", "n_params", "wall_min", "n_layers", "hidden", "bins", "ctx_hidden", "lr", "batch"]
with open(f"{DATA}/logs/flow/{sweep_name}_ranking.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(cols)
    for i, r in enumerate(good):
        hp = r["hyperparameters"]; w.writerow([i + 1, r["config_id"], f"{r['best_val']:.5f}", r["best_epoch"], r["epochs_run"], f"{r['final_train']:.5f}", r["n_params"],
                                               f"{r['wall_s']/60:.1f}", hp["n_layers"], hp["hidden"], hp["bins"], hp["ctx_hidden"], f"{hp['lr']:.2e}", hp["batch"]])
print("rank cfg  val_NLL   params  min   layers hidden bins ctx   lr      batch")
for i, r in enumerate(good):
    hp = r["hyperparameters"]; print(f"{i+1:3d}  {r['config_id']:>3s}  {r['best_val']:.4f}  {r['n_params']:7d}  {r['wall_s']/60:4.1f}  {hp['n_layers']:3d}   {hp['hidden']:4d}  {hp['bins']:3d}  {hp['ctx_hidden']:3d}  {hp['lr']:.1e}  {hp['batch']}")

gmin = good[0]["best_val"]
cmap = plt.get_cmap("viridis")
# ---------- 01: validation curves of every config
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
for i, r in enumerate(good):
    col = "crimson" if r is best else cmap(i / max(len(good) - 1, 1)); lw = 2.5 if r is best else 1.0
    ax[0].plot(r["hist"]["epoch"], r["hist"]["val"], color=col, lw=lw, label=f"cfg {r['config_id']}" if (r is best or i < 6) else None)
    ax[1].plot(r["hist"]["epoch"], np.array(r["hist"]["val"]) - gmin + 1e-3, color=col, lw=lw)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("validation NLL"); ax[0].set_ylim(gmin - 0.02, np.percentile(np.concatenate([r["hist"]["val"] for r in good]), 60)); ax[0].legend(fontsize=7, title="best 6 (red = winner)")
ax[1].set_yscale("log"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation NLL - best + 1e-3"); ax[1].set_title("distance from the best config")
fig.suptitle(f"{pop.upper()} sweep: validation curves of all {len(good)} configurations"); fig.tight_layout(); fig.savefig(f"{pdir}/01_validation_curves.png", dpi=130); plt.close(fig)

# ---------- 02: NLL vs each hyperparameter
hps = ["n_layers", "hidden", "bins", "ctx_hidden", "lr", "batch"]
fig, ax = plt.subplots(1, len(hps), figsize=(3.4 * len(hps), 3.6))
for a, k in zip(ax, hps):
    xs = np.array([r["hyperparameters"][k] for r in good]); ys = np.array([r["best_val"] for r in good])
    a.scatter(xs, ys, c=ys, cmap="viridis_r", s=45, edgecolor="k", lw=.4); a.scatter(best["hyperparameters"][k], best["best_val"], marker="*", s=300, color="crimson", zorder=5)
    if k == "lr": a.set_xscale("log")
    a.set_xlabel(k); a.set_ylim(gmin - 0.01, min(ys.max(), gmin + 0.3) + 0.01)
ax[0].set_ylabel("best validation NLL")
fig.suptitle(f"{pop.upper()} sweep: validation NLL vs each hyperparameter (star = winner; y-axis clipped at best+0.3)"); fig.tight_layout(); fig.savefig(f"{pdir}/02_nll_vs_hyperparameters.png", dpi=130); plt.close(fig)

# ---------- 03: NLL vs cost
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, key, lab in ((ax[0], "n_params", "parameters"), (ax[1], "wall_s", "training wall time [s]")):
    a.scatter([r[key] for r in good], [r["best_val"] for r in good], s=45, edgecolor="k", lw=.4)
    for r in good: a.annotate(r["config_id"], (r[key], r["best_val"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    a.scatter(best[key], best["best_val"], marker="*", s=300, color="crimson", zorder=5); a.set_xscale("log"); a.set_xlabel(lab); a.set_ylabel("best validation NLL"); a.set_ylim(gmin - 0.01, gmin + 0.3)
ax[2].scatter([r["final_train"] for r in good], [r["best_val"] for r in good], s=45, edgecolor="k", lw=.4)
allv = [r['final_train'] for r in good] + [r['best_val'] for r in good]; lo, hi = min(allv) - 0.02, np.percentile(allv, 90) + 0.05; ax[2].plot([lo, hi], [lo, hi], "k:", lw=1); ax[2].set_xlim(lo, hi); ax[2].set_ylim(lo, hi)
ax[2].set_xlabel("final-epoch training NLL"); ax[2].set_ylabel("best validation NLL"); ax[2].set_title("generalisation (below line: val better than train)")
fig.suptitle(f"{pop.upper()} sweep: accuracy vs cost"); fig.tight_layout(); fig.savefig(f"{pdir}/03_nll_vs_cost.png", dpi=130); plt.close(fig)

# ---------- 04: ranking
fig, ax = plt.subplots(figsize=(10, 0.42 * len(good) + 1.5))
y = np.arange(len(good))[::-1]
ax.barh(y, [r["best_val"] - gmin for r in good], color=["crimson" if r is best else "C0" for r in good])
for yi, r in zip(y, good):
    hp = r["hyperparameters"]; ax.text(0.0005, yi, f"cfg {r['config_id']}: L{hp['n_layers']} h{hp['hidden']} K{hp['bins']} c{hp['ctx_hidden']} lr{hp['lr']:.0e} b{hp['batch']}  ({r['n_params']/1e3:.0f}k par, {r['best_epoch']} ep)", va="center", fontsize=7)
ax.set_yticks([]); ax.set_xlabel("best validation NLL - winner's NLL"); ax.set_title(f"{pop.upper()}: ranking (winner NLL = {gmin:.4f}); differences below ~0.003 are within noise")
fig.tight_layout(); fig.savefig(f"{pdir}/04_ranking.png", dpi=130); plt.close(fig)

# ---------- winner: copy + plots + diagnostics
shutil.copy(f"{best['dir']}/best.pt", f"{bdir}/best.pt"); shutil.copy(f"{best['dir']}/history.csv", f"{bdir}/history.csv")
json.dump({k: v for k, v in best.items() if k not in ("hist",)}, open(f"{bdir}/best_config.json", "w"), indent=1, default=float)
plotting.loss_curves({k: list(v) for k, v in best["hist"].items()}, f"{bpdir}/01_loss_curves.png", title=f"{pop.upper()} winner cfg {best['config_id']}: {best['hyperparameters']}")
from diagnostics import run_diagnostics
run_diagnostics(pop, f"{bdir}/best.pt", bpdir, torch.device(args.device), title_extra=f" (winner cfg {best['config_id']})")
print("winner copied to", bdir, "| sweep plots:", pdir, "| winner plots:", bpdir)
