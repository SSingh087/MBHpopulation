#!/usr/bin/env python
"""
Train the conditional spline flow p(x | Lambda) for one source population (EMRI or TDE).

  python train_flow.py --pop emri                       # full run
  python train_flow.py --pop emri --max_universes 400 --epochs 3 --tag quick     # quick test

Every universe has equal weight (event weight ~ 1/n_events_in_universe), so the flow learns the
prior-averaged conditional density, not one dominated by high-count universes.
Outputs: <DATA>/models/flow_<pop><tag>/{best.pt,last.pt,history.csv}  and  <DATA>/plots/flow_<pop><tag>/*.png
"""
import argparse, csv, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, POPS, XTransform, LamScaler, load_consolidated
from flows import ConditionalFlow
import plotting

ap = argparse.ArgumentParser()
ap.add_argument("--pop", choices=list(POPS), required=True)
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--patience", type=int, default=20)
ap.add_argument("--max_steps_per_epoch", type=int, default=300)
ap.add_argument("--n_layers", type=int, default=6)
ap.add_argument("--hidden", type=int, default=64)
ap.add_argument("--bins", type=int, default=8)
ap.add_argument("--tail", type=float, default=6.0)
ap.add_argument("--max_universes", type=int, default=None, help="quick tests: use only the first N universes")
ap.add_argument("--max_val_events", type=int, default=100_000)
ap.add_argument("--tag", default="")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = torch.device(args.device)
name = f"flow_{args.pop}{args.tag}"
mdir, pdir = f"{DATA}/models/{name}", f"{DATA}/plots/{name}"
os.makedirs(mdir, exist_ok=True); os.makedirs(pdir, exist_ok=True)
info = POPS[args.pop]

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
    s = torch.randperm(len(x_va))[: args.max_val_events]; x_va, u_va = x_va[s], u_va[s]
xt = XTransform(info["bounds"]).fit(x_tr)
ls = LamScaler()
lam_s = ls.forward(lam_all)                                           # (U,6) scaled context per universe
nonempty = (n_u_events > 0)
def weights(u):  # equal weight per (non-empty) universe, mean 1
    w = 1.0 / n_u_events[u]; return w / w.mean()
w_tr, w_va = weights(u_tr), weights(u_va)
print(f"{args.pop}: train events {len(x_tr)} ({len(torch.unique(u_tr))} univ), val events {len(x_va)} ({len(torch.unique(u_va))} univ), device {dev}", flush=True)
X_tr, U_tr, W_tr = xt.forward(x_tr).to(dev), u_tr.to(dev), w_tr.to(dev)
X_va, U_va, W_va = xt.forward(x_va).to(dev), u_va.to(dev), w_va.to(dev)
LAM_S = lam_s.to(dev)

# ---------------- model
d = len(info["names"])
flow = ConditionalFlow(d, 6, n_layers=args.n_layers, hidden=args.hidden, K=args.bins, tail=args.tail).to(dev)
print("parameters:", sum(p.numel() for p in flow.parameters()), flush=True)
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

hist = dict(epoch=[], train=[], val=[], lr=[], time=[]); best, bad = 1e9, 0
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
    ckpt = dict(model=flow.state_dict(), cfg=dict(d=d, n_layers=args.n_layers, hidden=args.hidden, K=args.bins, tail=args.tail, pop=args.pop),
                x_mean=xt.mean, x_std=xt.std, epoch=ep, val=v)
    torch.save(ckpt, f"{mdir}/last.pt")
    if v < best - 1e-4: best, bad = v, 0; torch.save(ckpt, f"{mdir}/best.pt")
    else: bad += 1
    print(f"epoch {ep:3d}  train {hist['train'][-1]:.4f}  val {v:.4f}  best {best:.4f}  lr {hist['lr'][-1]:.1e}  {hist['time'][-1]:.1f}s", flush=True)
    if bad >= args.patience: print("early stopping"); break
with open(f"{mdir}/history.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(hist.keys()); w.writerows(zip(*hist.values()))
plotting.loss_curves(hist, f"{pdir}/01_loss_curves.png", title=f"{args.pop.upper()} flow training ({len(X_tr)} train events)")

# ---------------- diagnostics on validation universes (flow = best checkpoint)
ck = torch.load(f"{mdir}/best.pt", map_location=dev); flow.load_state_dict(ck["model"]); flow.eval()
n = len(X_va); samples = []
for i in range(0, n, 50000):
    samples.append(flow.sample(LAM_S[U_va[i:i + 50000]]))
u_flow = torch.cat(samples)
x_flow = xt.inverse(u_flow).cpu().numpy(); x_val = x_va.numpy(); lam_ev = meta["lam"][u_va.numpy()]
plotting.pooled_marginals(x_val, x_flow, info["labels"], f"{pdir}/02_val_pooled_marginals.png", title=f"{args.pop.upper()}: validation universes, one flow draw per event (conditioned on its Lambda)")
plotting.lambda_sweep(x_val, x_flow, lam_ev, info["labels"], f"{pdir}/03_val_lambda_sweep.png", title=f"{args.pop.upper()}: x mean/std vs Lambda on validation universes")
sub = np.random.default_rng(0).choice(n, min(n, 20000), replace=False)
plotting.corner_overlay(x_val[sub], x_flow[sub], info["labels"], f"{pdir}/04_val_corner.png", title=f"{args.pop.upper()}: validation events vs flow draws")
print("final: best val", best, "plots ->", pdir)
