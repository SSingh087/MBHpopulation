#!/usr/bin/env python
"""
Train the conditional spline flow p(x | Lambda) for one source population (EMRI or TDE).

  python train_flow.py --pop emri                                        # defaults
  python train_flow.py --pop emri --config_json sweep_emri.json --config_id 03   # one config of a sweep (see make_sweep_dag.py)
  python train_flow.py --pop emri --max_universes 400 --epochs 3 --out_name test # quick test

Every universe has equal weight (event weight ~ 1/n_events_in_universe).
Outputs under <DATA>/models/<out_name>/ : best.pt, last.pt, history.csv, summary.json ; <DATA>/plots/<out_name>/ : loss curves
(+ validation diagnostics unless --no_diag).
"""
import argparse, csv, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, POPS, XTransform, LamScaler, load_consolidated
from flows import ConditionalFlow
import plotting

ap = argparse.ArgumentParser()
ap.add_argument("--pop", choices=list(POPS), required=True)
ap.add_argument("--config_json", default=None, help="sweep file written by make_sweep_dag.py")
ap.add_argument("--config_id", default=None)
ap.add_argument("--out_name", default=None, help="subfolder name under models/ and plots/ (default flow_<pop>)")
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--patience", type=int, default=20)
ap.add_argument("--max_steps_per_epoch", type=int, default=300)
# hyperparameters (overridden by --config_json/--config_id)
ap.add_argument("--n_layers", type=int, default=6)
ap.add_argument("--hidden", type=int, default=64)
ap.add_argument("--bins", type=int, default=8)
ap.add_argument("--ctx_hidden", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--tail", type=float, default=6.0)
ap.add_argument("--max_universes", type=int, default=None, help="quick tests: use only the first N universes")
ap.add_argument("--max_val_events", type=int, default=100_000)
ap.add_argument("--no_diag", action="store_true", help="skip the validation diagnostic plots (used inside sweeps)")
ap.add_argument("--skip_done", action="store_true", help="exit if summary.json already exists (DAG retries)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()

if args.config_json:
    sw = json.load(open(args.config_json))
    cfg = next(c for c in sw["configs"] if c["id"] == args.config_id)
    for k, v in sw.get("train", {}).items(): setattr(args, k, v)
    for k in ("n_layers", "hidden", "bins", "ctx_hidden", "lr", "batch", "tail"):
        if k in cfg: setattr(args, k, cfg[k])
    if "seed" in cfg: args.seed = cfg["seed"]
out_name = args.out_name or f"flow_{args.pop}"
mdir, pdir = f"{DATA}/models/{out_name}", f"{DATA}/plots/{out_name}"
if args.skip_done and os.path.exists(f"{mdir}/summary.json"):
    print("already done:", mdir); sys.exit(0)
os.makedirs(mdir, exist_ok=True); os.makedirs(pdir, exist_ok=True)
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = torch.device(args.device)
info = POPS[args.pop]
hp = dict(n_layers=args.n_layers, hidden=args.hidden, bins=args.bins, ctx_hidden=args.ctx_hidden, lr=args.lr, batch=args.batch, tail=args.tail, seed=args.seed)
print("hyperparameters:", hp, flush=True)

# ---------------- data
meta, x, uid = load_consolidated(args.pop)
lam_all = torch.tensor(meta["lam"]); is_val_u = torch.tensor(meta["is_val"])
if args.max_universes:
    keep = uid < args.max_universes; x, uid = x[keep], uid[keep]
x = torch.tensor(x); uid = torch.tensor(uid, dtype=torch.long)
n_u_events = torch.bincount(uid, minlength=len(lam_all)).float()
ev_val = is_val_u[uid]
x_tr, u_tr, x_va, u_va = x[~ev_val], uid[~ev_val], x[ev_val], uid[ev_val]
if len(x_va) > args.max_val_events:
    g = torch.Generator().manual_seed(1234)                      # same validation subset for every config
    s = torch.randperm(len(x_va), generator=g)[: args.max_val_events]; x_va, u_va = x_va[s], u_va[s]
xt = XTransform(info["bounds"]).fit(x_tr)
lam_s = LamScaler().forward(lam_all)
def weights(u):
    w = 1.0 / n_u_events[u]; return w / w.mean()
X_tr, U_tr, W_tr = xt.forward(x_tr).to(dev), u_tr.to(dev), weights(u_tr).to(dev)
X_va, U_va, W_va = xt.forward(x_va).to(dev), u_va.to(dev), weights(u_va).to(dev)
LAM_S = lam_s.to(dev)
print(f"{args.pop}: train events {len(X_tr)}, val events {len(X_va)}, device {dev}", flush=True)

# ---------------- model
d = len(info["names"])
flow = ConditionalFlow(d, 6, n_layers=args.n_layers, hidden=args.hidden, K=args.bins, tail=args.tail, ctx_hidden=args.ctx_hidden).to(dev)
n_params = sum(p.numel() for p in flow.parameters()); print("parameters:", n_params, flush=True)
opt = torch.optim.Adam(flow.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
steps = int(min(args.max_steps_per_epoch, max(1, len(X_tr) // args.batch)))

@torch.no_grad()
def val_loss():
    flow.eval(); tot, wsum = 0.0, 0.0
    for i in range(0, len(X_va), 20000):
        nll = -flow.log_prob(X_va[i:i + 20000], LAM_S[U_va[i:i + 20000]])
        tot += (W_va[i:i + 20000] * nll).sum().item(); wsum += W_va[i:i + 20000].sum().item()
    flow.train(); return tot / wsum

hist = dict(epoch=[], train=[], val=[], lr=[], time=[]); best, best_ep, bad = 1e9, 0, 0; t_start = time.time(); stopped = False
for ep in range(1, args.epochs + 1):
    t0 = time.time(); run, n_ok = 0.0, 0
    for _ in range(steps):
        idx = torch.randint(len(X_tr), (args.batch,), device=dev)
        nll = -flow.log_prob(X_tr[idx], LAM_S[U_tr[idx]])
        loss = (W_tr[idx] * nll).sum() / W_tr[idx].sum()
        if not torch.isfinite(loss): continue
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0); opt.step()
        run += loss.item(); n_ok += 1
    v = val_loss(); sched.step(v)
    hist["epoch"].append(ep); hist["train"].append(run / max(n_ok, 1)); hist["val"].append(v)
    hist["lr"].append(opt.param_groups[0]["lr"]); hist["time"].append(time.time() - t0)
    ckpt = dict(model=flow.state_dict(), cfg=dict(d=d, pop=args.pop, n_layers=args.n_layers, hidden=args.hidden, K=args.bins, tail=args.tail, ctx_hidden=args.ctx_hidden),
                x_mean=xt.mean, x_std=xt.std, epoch=ep, val=v)
    torch.save(ckpt, f"{mdir}/last.pt")
    if v < best - 1e-4: best, best_ep, bad = v, ep, 0; torch.save(ckpt, f"{mdir}/best.pt")
    else: bad += 1
    print(f"epoch {ep:3d}  train {hist['train'][-1]:.4f}  val {v:.4f}  best {best:.4f}  lr {hist['lr'][-1]:.1e}  {hist['time'][-1]:.1f}s", flush=True)
    if not np.isfinite(v): print("non-finite validation loss, abort"); break
    if bad >= args.patience: print("early stopping"); stopped = True; break
with open(f"{mdir}/history.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(hist.keys()); w.writerows(zip(*hist.values()))
summary = dict(pop=args.pop, hyperparameters=hp, best_val=best, best_epoch=best_ep, epochs_run=len(hist["epoch"]), early_stopped=stopped,
               final_train=hist["train"][-1], n_params=n_params, wall_s=time.time() - t_start, train_events=len(X_tr), val_events=len(X_va),
               config_id=args.config_id)
json.dump(summary, open(f"{mdir}/summary.json", "w"), indent=1)
plotting.loss_curves(hist, f"{pdir}/01_loss_curves.png", title=f"{args.pop.upper()} {out_name}: {hp}")
if not args.no_diag:
    from diagnostics import run_diagnostics
    run_diagnostics(args.pop, f"{mdir}/best.pt", pdir, dev, args.max_val_events)
print("final: best val", best, "at epoch", best_ep, "->", mdir)
