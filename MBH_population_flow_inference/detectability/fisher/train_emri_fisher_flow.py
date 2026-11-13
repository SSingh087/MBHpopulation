#!/usr/bin/env python
"""
train_emri_fisher_flow.py -- learning curve of the EMRI Fisher-error flow: p(error | EMRI event parameters), trained on the EMRI Fisher bank (fixed Fisher SNR = 100).

Target per event (105 numbers) from the 14x14 posterior covariance C (normalised-inverse of the Fisher matrix, fisher_cov.py), in 'natural' coordinates
(ln M, ln mu, ln p0, ln dist, and a, e0, Y0, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 as they are):   C = D R D,  R = Lr Lr^T  ->
    y = [log sigma_i (14), Lr_ij for i>j (91)]        (sigma at SNR 100; at application sigma(rho) = sigma_100 * 100/rho)
Context (18): log10 M, log10 mu, a, p0, e0, Y0, qS, sin/cos phiS, qK, sin/cos phiK, sin/cos of the three initial phases, T_obs.   (distance is NOT a context: sigma*rho does not depend on it)
The script trains at several training-set fractions with the SAME validation/test split and writes the learning curve (NLL vs data), a calibration check (PIT of log sigma) and the plots.
  python train_emri_fisher_flow.py --fracs 0.1 0.2 0.4 0.7 1.0 --seeds 0 1 2
"""
import argparse, glob, json, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, f"{ROOT}/flow"); sys.path.insert(0, HERE)
from flows import ConditionalFlow
from fisher_cov import covariance
DATA = os.environ.get("EMRI_FLOW_DATA", "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference")
ap = argparse.ArgumentParser(); ap.add_argument("--bank_dir", nargs="+", default=[f"{DATA}/simulations/emri_fisher_bank"], help="one or more bank folders, read together"); ap.add_argument("--fracs", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.7, 1.0])
ap.add_argument("--seeds", type=int, nargs="+", default=[0]); ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--patience", type=int, default=30)
ap.add_argument("--n_layers", type=int, default=4); ap.add_argument("--hidden", type=int, default=64); ap.add_argument("--bins", type=int, default=8); ap.add_argument("--ctx_hidden", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--batch", type=int, default=128); ap.add_argument("--wd", type=float, default=1e-5); ap.add_argument("--max_cond", type=float, default=1e12)
ap.add_argument("--out_name", default="emri_fisher_flow_lc"); ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=8); a = ap.parse_args()
torch.set_num_threads(a.threads); dev = torch.device(a.device); NP = 14; TRI = np.tril_indices(NP, -1)
mdir = f"{DATA}/models/detectability/{a.out_name}"; pdir = f"{DATA}/plots/detectability/emri_fisher"; os.makedirs(mdir, exist_ok=True); os.makedirs(pdir, exist_ok=True)
LOGP = [0, 1, 3, 6]      # M, mu, p0, dist -> ln coordinates (sigma = relative error)


def encode(C):
    sig = np.sqrt(np.einsum("nii->ni", C)); R = C / sig[:, :, None] / sig[:, None, :]; L = np.linalg.cholesky(R + 1e-12 * np.eye(NP)); L = L / np.linalg.norm(L, axis=2, keepdims=True)
    return np.concatenate([np.log(sig), L[:, TRI[0], TRI[1]]], 1)


def decode(y):
    sig = np.exp(y[:, :NP]); L = np.zeros((len(y), NP, NP)); L[:, TRI[0], TRI[1]] = y[:, NP:]
    for i in range(NP):
        off = np.linalg.norm(L[:, i, :i], axis=1); s = np.where(off > 0.999, 0.999 / np.maximum(off, 1e-12), 1.0); L[:, i, :i] *= s[:, None]; L[:, i, i] = np.sqrt(1 - np.linalg.norm(L[:, i, :i], axis=1) ** 2)
    return L @ L.transpose(0, 2, 1) * sig[:, :, None] * sig[:, None, :]


# ---------------- data
Dl = [np.load(f) for f in sorted(f for b in a.bank_dir for f in glob.glob(f"{b}/chunk_*.npz"))]; cat = lambda k: np.concatenate([d[k] for d in Dl])
ok = cat("ok").astype(bool); th = cat("theta")[ok]; Tb = cat("Tb")[ok]; F = cat("F1")[ok]; n_att = len(ok)
cv = [covariance(f) for f in F]; C = np.array([c[0] for c in cv]); condn = np.array([c[1] for c in cv])
J = np.ones((len(th), NP)); J[:, LOGP] = 1.0 / th[:, LOGP]; C = C * J[:, :, None] * J[:, None, :]            # natural coordinates
sig = np.sqrt(np.einsum("nii->ni", C)); Rr = C / sig[:, :, None] / sig[:, None, :]
keep = (condn <= a.max_cond) & np.all(np.isfinite(C), (1, 2)) & (np.linalg.eigvalsh(np.nan_to_num(Rr)).min(1) > 1e-9)
print(f"attempted {n_att}, Fisher ok {int(ok.sum())}, kept {int(keep.sum())} (cond <= {a.max_cond:g}, finite, positive-definite correlation)", flush=True)
th, Tb, C = th[keep], Tb[keep], C[keep]; Y = encode(C)
ctx = np.stack([np.log10(th[:, 0]), np.log10(th[:, 1]), th[:, 2], th[:, 3], th[:, 4], th[:, 5], th[:, 7], np.sin(th[:, 8]), np.cos(th[:, 8]), th[:, 9], np.sin(th[:, 10]), np.cos(th[:, 10]),
                np.sin(th[:, 11]), np.cos(th[:, 11]), np.sin(th[:, 12]), np.cos(th[:, 12]), np.sin(th[:, 13]), np.cos(th[:, 13]), Tb], 1)
n = len(Y); perm = np.random.default_rng(0).permutation(n); n_te = n_va = int(0.1 * n); i_te, i_va, i_tr_all = perm[:n_te], perm[n_te:n_te + n_va], perm[n_te + n_va:]
print(f"split: train {len(i_tr_all)} / val {n_va} / test {n_te}; context {ctx.shape[1]}, targets {Y.shape[1]}", flush=True)
names = ["M", "mu", "a", "p0", "e0", "Y0", "dist", "qS", "phiS", "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0"]
T = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)


def train_one(frac, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(100 + seed); itr = rng.permutation(i_tr_all)[: max(8, int(frac * len(i_tr_all)))]
    cm, cs, ym, ys = ctx[itr].mean(0), ctx[itr].std(0) + 1e-9, Y[itr].mean(0), Y[itr].std(0) + 1e-9
    Ct, Yt, Cv, Yv, Cte, Yte = T((ctx[itr] - cm) / cs), T((Y[itr] - ym) / ys), T((ctx[i_va] - cm) / cs), T((Y[i_va] - ym) / ys), T((ctx[i_te] - cm) / cs), T((Y[i_te] - ym) / ys)
    fl = ConditionalFlow(Y.shape[1], ctx.shape[1], n_layers=a.n_layers, hidden=a.hidden, K=a.bins, tail=6.0, ctx_hidden=a.ctx_hidden).to(dev)
    opt = torch.optim.AdamW(fl.parameters(), lr=a.lr, weight_decay=a.wd); sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
    nll = lambda Yx, Cx: -fl.log_prob(Yx, Cx).mean().item()
    best, bad, state, hist = 1e9, 0, None, []
    for ep in range(a.epochs):
        fl.train(); p = torch.randperm(len(Yt), device=dev)
        for s in range(0, len(Yt), a.batch):
            ii = p[s:s + a.batch]; loss = -fl.log_prob(Yt[ii], Ct[ii]).mean()
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(fl.parameters(), 5.0); opt.step()
        fl.eval()
        with torch.no_grad(): v = nll(Yv, Cv)
        hist.append(v); sch.step(v)
        if v < best - 1e-4: best, bad, state = v, 0, {k: x.clone() for k, x in fl.state_dict().items()}
        else: bad += 1
        if bad >= a.patience or not np.isfinite(v): break
    fl.load_state_dict(state); fl.eval()
    with torch.no_grad():
        te = nll(Yte, Cte); S = torch.stack([fl.sample(Cte) for _ in range(100)], 0).cpu().numpy() * ys + ym
    Yt_ = Y[i_te]; pit = (S[:, :, :NP] < Yt_[None, :, :NP]).mean(0); med = np.median(S[:, :, :NP], 0)
    return dict(frac=frac, seed=seed, n_train=len(itr), val=best, test=te, epochs=len(hist),
                pit_ks=[float(np.max(np.abs(np.sort(pit[:, i]) - np.linspace(0, 1, len(pit))))) for i in range(NP)],
                med_abs_dex=[float(np.median(np.abs(med[:, i] - Yt_[:, i])) / np.log(10)) for i in range(NP)]), fl, (cm, cs, ym, ys)


R = []; t0 = time.time()
for fr in a.fracs:
    for sd in a.seeds:
        r, fl, norm = train_one(fr, sd); R.append(r); print(f"frac {fr:.2f} seed {sd}: n_train {r['n_train']} epochs {r['epochs']} val {r['val']:.2f} test {r['test']:.2f}   [{(time.time()-t0)/60:.1f} min]", flush=True)
        if fr == max(a.fracs) and sd == a.seeds[0]: torch.save(dict(model=fl.state_dict(), norm=norm, cfg=dict(d=Y.shape[1], ctx_dim=ctx.shape[1], n_layers=a.n_layers, hidden=a.hidden, K=a.bins, tail=6.0, ctx_hidden=a.ctx_hidden)), f"{mdir}/best.pt")
spread = [float(np.std(Y[i_tr_all, i]) / np.log(10)) for i in range(NP)]
json.dump(dict(runs=R, n_kept=int(n), intrinsic_spread_dex_logsigma=dict(zip(names, spread)), args=vars(a)), open(f"{mdir}/learning_curve.json", "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2)); fr = sorted(set(r["frac"] for r in R)); nt = [np.mean([r["n_train"] for r in R if r["frac"] == f]) for f in fr]
for key, c in (("val", "C0"), ("test", "C3")):
    m = [np.mean([r[key] for r in R if r["frac"] == f]) for f in fr]; e = [np.std([r[key] for r in R if r["frac"] == f]) for f in fr]; ax[0].errorbar(nt, m, e, fmt="o-", color=c, label=key)
ax[0].set_xscale("log"); ax[0].set_xlabel("training events"); ax[0].set_ylabel("NLL per event (lower is better)"); ax[0].legend(); ax[0].set_title("learning curve")
last = [r for r in R if r["frac"] == max(fr)]; md = np.mean([r["med_abs_dex"] for r in last], 0); ks = np.mean([r["pit_ks"] for r in last], 0); x = np.arange(NP)
ax[1].bar(x - 0.2, md, 0.4, label="median |error of predicted log sigma| [dex]"); ax[1].bar(x + 0.2, spread, 0.4, label="intrinsic spread of log sigma [dex]"); ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=70, fontsize=7); ax[1].legend(fontsize=7); ax[1].set_title("what the flow predicts (full data)")
ax[2].bar(x, ks); ax[2].axhline(1.36 / np.sqrt(n_te), color="r", ls="--", label="5% KS level"); ax[2].set_xticks(x); ax[2].set_xticklabels(names, rotation=70, fontsize=7); ax[2].set_ylabel("PIT-KS"); ax[2].legend(fontsize=7); ax[2].set_title("calibration of log sigma")
fig.tight_layout(); fig.savefig(f"{pdir}/{a.out_name}.png", dpi=130); fig.savefig(f"{pdir}/{a.out_name}.pdf"); print("saved", pdir, mdir)
