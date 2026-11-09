#!/usr/bin/env python
"""
Train the conditional flow p(Fisher error | TDE event parameters) for one survey, from the pooled TDE Fisher banks
(bank 1 = wide prior, bank 2 = catalogue-mixed).  Conditional model => the proposal density of the banks does not bias it.

Target (28 numbers per event) from the stored 7x7 posterior covariance C (data Fisher + nuisance priors, order
[z, log10 M6, log10 eta, log10 tvisc, bb, log10 rph0, lphoto]):
    C = D R D,  D = diag(sigma),  R = Lr Lr^T (Cholesky of the correlation matrix, unit-norm rows)
    y = [log sigma_i (7), Lr_ij for i>j (21)]       -> sample => unit-normalise rows of Lr => valid C
Context (13): lgM, z, log10 eta, log10 tvisc, bb, log10 rph0, lphoto, log10 alpha, beta, sin ra, cos ra, dec, (t0-t0min)/range.
Trimming (training time only): ok==1, sigma_lgM <= 3 dex, normalised cond <= 1e12.
Split: random 85/7.5/7.5 train/val/test over events (seeded), test never touched for model selection.

  python train_tde_fisher_flow.py --survey ZTF                       # production
  python train_tde_fisher_flow.py --survey ZTF --epochs 3 --out_name test_tde_err_ZTF    # quick test
Outputs: models/detectability/tde_fisher_flow_<survey>/{best.pt,summary.json,history.csv}, plots/detectability/tde_fisher_flow/<survey>_*.png
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "flow"))
from flows import ConditionalFlow

DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"
BANKS = ["tde_fisher_bank", "tde_fisher_bank_cat"]
NP = 7
TRI = np.tril_indices(NP, -1)

ap = argparse.ArgumentParser()
ap.add_argument("--banks", nargs="+", default=BANKS, help="bank directories under simulations/ to pool")
ap.add_argument("--survey", choices=["ZTF", "LSST"], required=True)
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--patience", type=int, default=25)
ap.add_argument("--n_layers", type=int, default=6)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--bins", type=int, default=8)
ap.add_argument("--ctx_hidden", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--tail", type=float, default=6.0)
ap.add_argument("--max_sigma_lgM", type=float, default=3.0)
ap.add_argument("--max_cond", type=float, default=1e12)
ap.add_argument("--seed", type=int, default=0, help="network init / batch order")
ap.add_argument("--data_seed", type=int, default=0, help="train/val/test split (keep fixed across a sweep)")
ap.add_argument("--train_frac", type=float, default=1.0, help="use only this fraction of the training split (learning curve); val/test unchanged")
ap.add_argument("--no_eval", action="store_true", help="train only (sweep nodes); test split untouched")
ap.add_argument("--eval_only", action="store_true", help="skip training, evaluate best.pt in --out_name on the test split")
ap.add_argument("--out_name", default=None)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()
out_name = args.out_name or f"tde_fisher_flow_{args.survey}"
mdir = f"{DATA}/models/detectability/{out_name}"; out_name = os.path.basename(out_name); pdir = f"{DATA}/plots/detectability/tde_fisher_flow"
os.makedirs(mdir, exist_ok=True); os.makedirs(pdir, exist_ok=True)
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = torch.device(args.device)


def encode_C(C):
    sig = np.sqrt(np.einsum("nii->ni", C))
    R = C / sig[:, :, None] / sig[:, None, :]
    L = np.linalg.cholesky(R + 1e-12 * np.eye(NP))
    L = L / np.linalg.norm(L, axis=2, keepdims=True)
    return np.concatenate([np.log(sig), L[:, TRI[0], TRI[1]]], 1)


def decode_y(y):
    sig = np.exp(y[:, :NP]); L = np.zeros((len(y), NP, NP))
    L[:, TRI[0], TRI[1]] = y[:, NP:]
    # diagonal fixed by unit row norm (positive); if the off-diagonals overshoot, rescale the row
    for i in range(NP):
        off = np.linalg.norm(L[:, i, :i], axis=1)
        s = np.where(off > 0.999, 0.999 / np.maximum(off, 1e-12), 1.0); L[:, i, :i] *= s[:, None]
        L[:, i, i] = np.sqrt(1 - np.linalg.norm(L[:, i, :i], axis=1) ** 2)
    R = L @ L.transpose(0, 2, 1)
    return R * sig[:, :, None] * sig[:, None, :]


def context(d, t0min, trange):
    return np.stack([d["lgM"], d["z"], np.log10(d["eta"]), np.log10(d["tvisc"]), d["bb"], np.log10(d["rph0"]), d["lphoto"],
                     np.log10(d["alpha"]), d["beta"], np.sin(d["ra"]), np.cos(d["ra"]), d["dec"], (d["t0"] - t0min) / trange], 1)


# ---------------- data
keys = ["lgM", "z", "ra", "dec", "t0", "eta", "tvisc", "bb", "rph0", "lphoto", "alpha", "beta", "C", "ok", "cond", "npts"]
cols = {k: [] for k in keys}; src = []
for bi, b in enumerate(args.banks):
    for f in sorted(glob.glob(f"{DATA}/simulations/{b}/{args.survey}_chunk_*.npz")):
        d = np.load(f)
        for k in keys: cols[k].append(d[k])
        src.append(np.full(len(d["ok"]), bi))
D = {k: np.concatenate(v) for k, v in cols.items()}; src = np.concatenate(src)
n_all = len(D["ok"])
sig_lgM = np.sqrt(D["C"][:, 1, 1])
keep = (D["ok"] == 1) & (sig_lgM <= args.max_sigma_lgM) & (D["cond"] <= args.max_cond) & np.all(np.isfinite(D["C"]), (1, 2))
print(f"{args.survey}: attempted {n_all}, ok {int((D['ok']==1).sum())}, kept after trimming {int(keep.sum())}", flush=True)
D = {k: v[keep] for k, v in D.items()}; src = src[keep]
# drop numerically non-positive-definite correlation matrices (needed for the Cholesky encoding); count them
_s = np.sqrt(np.einsum("nii->ni", D["C"])); _R = D["C"] / _s[:, :, None] / _s[:, None, :]
pd_ok = np.linalg.eigvalsh(_R).min(1) > 1e-9
print(f"  dropped {int((~pd_ok).sum())} events with non-positive-definite correlation matrix (min eig <= 1e-9)", flush=True)
D = {k: v[pd_ok] for k, v in D.items()}; src = src[pd_ok]
t0min, trange = float(D["t0"].min()), float(D["t0"].max() - D["t0"].min())
ctx = context(D, t0min, trange); Y = encode_C(D["C"])
ok_row = np.all(np.isfinite(Y), 1); ctx, Y, src = ctx[ok_row], Y[ok_row], src[ok_row]
n = len(Y); perm = np.random.default_rng(args.data_seed).permutation(n)
n_te = n_va = int(0.075 * n)
idx_te, idx_va, idx_tr = perm[:n_te], perm[n_te:n_te + n_va], perm[n_te + n_va:]
idx_tr = idx_tr[: max(1, int(args.train_frac * len(idx_tr)))]
cm, cs = ctx[idx_tr].mean(0), ctx[idx_tr].std(0) + 1e-9
ym, ys = Y[idx_tr].mean(0), Y[idx_tr].std(0) + 1e-9
T = lambda a, i: torch.tensor(a[i], dtype=torch.float32, device=dev)
C_tr, Y_tr = T((ctx - cm) / cs, idx_tr), T((Y - ym) / ys, idx_tr)
C_va, Y_va = T((ctx - cm) / cs, idx_va), T((Y - ym) / ys, idx_va)
C_te, Y_te = T((ctx - cm) / cs, idx_te), T((Y - ym) / ys, idx_te)
print(f"train {len(idx_tr)}  val {len(idx_va)}  test {len(idx_te)}  device {dev}", flush=True)

if args.eval_only:
    _c = torch.load(f"{mdir}/best.pt", map_location="cpu")["cfg"]
    args.n_layers, args.hidden, args.bins, args.ctx_hidden, args.tail = _c["n_layers"], _c["hidden"], _c["K"], _c["ctx_hidden"], _c["tail"]
flow = ConditionalFlow(Y.shape[1], ctx.shape[1], n_layers=args.n_layers, hidden=args.hidden, K=args.bins, tail=args.tail, ctx_hidden=args.ctx_hidden).to(dev)
n_params = sum(p.numel() for p in flow.parameters()); print("parameters:", n_params, flush=True)
opt = torch.optim.Adam(flow.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
steps = max(1, len(Y_tr) // args.batch)


@torch.no_grad()
def nll_of(Yx, Cx):
    flow.eval(); tot = 0.0
    for i in range(0, len(Yx), 20000): tot += -flow.log_prob(Yx[i:i + 20000], Cx[i:i + 20000]).sum().item()
    flow.train(); return tot / len(Yx)


hist, best, best_ep, bad, t_start = [], 1e9, 0, 0, time.time()
cfg = dict(d=Y.shape[1], ctx_dim=ctx.shape[1], n_layers=args.n_layers, hidden=args.hidden, K=args.bins, tail=args.tail, ctx_hidden=args.ctx_hidden)
for ep in range(1, 1 if args.eval_only else args.epochs + 1):
    perm_tr = torch.randperm(len(Y_tr), device=dev); run = 0.0; n_ok = 0
    for s in range(steps):
        ii = perm_tr[s * args.batch:(s + 1) * args.batch]
        loss = -flow.log_prob(Y_tr[ii], C_tr[ii]).mean()
        if not torch.isfinite(loss): continue
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0); opt.step(); run += loss.item(); n_ok += 1
    v = nll_of(Y_va, C_va); sched.step(v); hist.append((ep, run / max(n_ok, 1), v))
    ck = dict(model=flow.state_dict(), cfg=cfg, ctx_mean=cm, ctx_std=cs, y_mean=ym, y_std=ys, t0min=t0min, trange=trange, survey=args.survey, epoch=ep, val=v)
    if v < best - 1e-4: best, best_ep, bad = v, ep, 0; torch.save(ck, f"{mdir}/best.pt")
    else: bad += 1
    print(f"epoch {ep:3d} train {hist[-1][1]:.4f} val {v:.4f} best {best:.4f} lr {opt.param_groups[0]['lr']:.1e}", flush=True)
    if not np.isfinite(v): print("non-finite validation loss"); break
    if bad >= args.patience: print("early stopping"); break
if args.eval_only:
    hist = np.loadtxt(f"{mdir}/history.csv", delimiter=",").reshape(-1, 3).tolist(); old = json.load(open(f"{mdir}/summary.json"))
    best, best_ep = old["best_val"], old["best_epoch"]
else:
    np.savetxt(f"{mdir}/history.csv", np.array(hist), delimiter=",", header="epoch,train,val")
    json.dump(dict(survey=args.survey, best_val=best, best_epoch=best_ep, n_params=n_params, wall_s=time.time() - t_start, hyper=cfg, lr=args.lr, batch=args.batch,
                   epochs_run=len(hist), n_kept=int(n), n_train=len(idx_tr), train_frac=args.train_frac), open(f"{mdir}/summary.json", "w"), indent=1)
if args.no_eval: print("train only ->", mdir); sys.exit(0)

# ---------------- test-set evaluation (best checkpoint)
ck = torch.load(f"{mdir}/best.pt", map_location=dev); flow.load_state_dict(ck["model"]); flow.eval()
test_nll = nll_of(Y_te, C_te)
NS = 200
with torch.no_grad():
    S = torch.stack([flow.sample(C_te) for _ in range(NS)], 0).cpu().numpy() * ys + ym      # (NS, n_te, 28)
Yt = Y[idx_te]
pit = (S < Yt[None]).mean(0)                                                                  # PIT per target dimension
med = np.median(S, 0)
names = ["z", "lgM", "eta", "tvisc", "bb", "rph0", "lphoto"]
res = dict(survey=args.survey, n_attempted=n_all, n_kept=int(n), n_train=len(idx_tr), n_val=len(idx_va), n_test=len(idx_te), best_val=best,
           best_epoch=best_ep, test_nll=test_nll, n_params=n_params, wall_s=time.time() - t_start, hyper=cfg,
           median_abs_dex_err_sigma={names[i]: float(np.median(np.abs(med[:, i] - Yt[:, i])) / np.log(10)) for i in range(NP)},
           intrinsic_spread_dex_sigma={names[i]: float(np.std(Yt[:, i]) / np.log(10)) for i in range(NP)},
           pit_ks={names[i]: float(np.max(np.abs(np.sort(pit[:, i]) - np.linspace(0, 1, len(pit))))) for i in range(NP)})
# does the sampled covariance reproduce the true one? compare sigma_lgM and a conditional-number-free check of correlations
Cs = decode_y(S[0]); Ct = D["C"][ok_row][idx_te]
res["sampled_sigma_lgM_vs_true_dex_median_abs"] = float(np.median(np.abs(np.log10(np.sqrt(Cs[:, 1, 1]) / np.sqrt(Ct[:, 1, 1])))))
json.dump(res, open(f"{mdir}/test_summary.json", "w"), indent=1)
print(json.dumps(res, indent=1))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
h = np.array(hist).reshape(-1, 3)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
ax[0].plot(h[:, 0], h[:, 1], label="train"); ax[0].plot(h[:, 0], h[:, 2], label="val"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("NLL"); ax[0].legend()
for i in range(NP): ax[1].hist(pit[:, i], bins=20, histtype="step", density=True, label=names[i])
ax[1].set_title("PIT of true log sigma (flat = calibrated)"); ax[1].legend(fontsize=7)
for k, (i, nm) in enumerate([(1, "lgM"), (0, "z")]):
    ax[2].scatter(Yt[:, i] / np.log(10), med[:, i] / np.log(10), s=2, alpha=0.3, label=f"log10 sigma_{nm}")
lo, hi = ax[2].get_xlim(); ax[2].plot([lo, hi], [lo, hi], "k--"); ax[2].set_xlabel("true"); ax[2].set_ylabel("flow median"); ax[2].legend()
fig.suptitle(f"TDE Fisher-error flow, {args.survey}: test NLL {test_nll:.2f}"); fig.tight_layout(); fig.savefig(f"{pdir}/{out_name}_diag.png", dpi=120)
print("done ->", mdir)
