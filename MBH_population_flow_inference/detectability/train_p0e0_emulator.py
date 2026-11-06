"""train_p0e0_emulator.py -- MLP emulator (lgM, a, Y0, e_plunge, T) -> (e0, log10 p0), trained on forward-FEW rows.
Also checks what matters downstream: the SNR network's output (log10 SNR*D_L) with true vs emulated (e0, p0).
Writes models/detectability/p0e0_emulator/{model.pt,summary.json}, plots/detectability/p0e0_emulator/*.png"""
import argparse, glob, json, os, sys
import numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"
ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default=f"{DATA}/simulations/p0e0_traj"); ap.add_argument("--out_name", default="p0e0_emulator")
ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--hidden", type=int, default=256); ap.add_argument("--layers", type=int, default=4); ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--snr_model", default=f"{DATA}/models/detectability/emri_snr/model.pth")
a = ap.parse_args()
mdir = f"{DATA}/models/detectability/{a.out_name}"; pdir = f"{DATA}/plots/detectability/{a.out_name}"
os.makedirs(mdir, exist_ok=True); os.makedirs(pdir, exist_ok=True)
fs = sorted(glob.glob(f"{a.data_dir}/chunk_*.npz")); d = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ["lgM", "a", "Y0", "e_plunge", "T", "e0", "p0"]}
n = len(d["lgM"]); print(f"{len(fs)} chunks, {n} rows")
X = np.stack([d["lgM"], d["a"], d["Y0"], d["e_plunge"], d["T"]], 1); Y = np.stack([d["e0"], np.log10(d["p0"])], 1)
perm = np.random.default_rng(a.seed).permutation(n); ntr, nva = int(.8 * n), int(.1 * n)
tr, va, te = perm[:ntr], perm[ntr:ntr + nva], perm[ntr + nva:]
xm, xs, ym, ys = X[tr].mean(0), X[tr].std(0), Y[tr].mean(0), Y[tr].std(0)
tt = lambda v, m, s: torch.tensor((v - m) / s, dtype=torch.float32, device=a.device)
Xt, Yt = tt(X, xm, xs), tt(Y, ym, ys)
torch.manual_seed(a.seed)
layers = []; i = 5
for _ in range(a.layers): layers += [nn.Linear(i, a.hidden), nn.SiLU()]; i = a.hidden
net = nn.Sequential(*layers, nn.Linear(i, 2)).to(a.device)
opt = torch.optim.Adam(net.parameters(), lr=a.lr); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
hist = []; best = (1e9, None); trI = torch.tensor(tr, device=a.device)
for ep in range(a.epochs):
    net.train(); p = trI[torch.randperm(len(trI), device=a.device)]; tl = 0; nb = 0
    for b in range(0, len(p), a.batch):
        ix = p[b:b + a.batch]; loss = nn.functional.smooth_l1_loss(net(Xt[ix]), Yt[ix], beta=0.05)
        opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item(); nb += 1
    sch.step(); net.eval()
    with torch.no_grad(): vl = nn.functional.smooth_l1_loss(net(Xt[va]), Yt[va], beta=0.05).item()
    hist.append((ep, tl / nb, vl))
    if vl < best[0]: best = (vl, {k: v.clone() for k, v in net.state_dict().items()}, ep)
    if ep % 20 == 0 or ep == a.epochs - 1: print(f"ep {ep} train {tl/nb:.5f} val {vl:.5f}", flush=True)
net.load_state_dict(best[1]); net.eval()
torch.save(dict(state=best[1], hidden=a.hidden, layers=a.layers, xm=xm, xs=xs, ym=ym, ys=ys, epoch=best[2]), f"{mdir}/model.pt")
with torch.no_grad(): P = net(Xt[te]).cpu().numpy() * ys + ym
T_ = Y[te]; e0p, lp0p = P[:, 0], P[:, 1]; e0t, lp0t = T_[:, 0], T_[:, 1]
res = dict(n_rows=n, n_test=len(te), best_epoch=best[2],
           e0_mae=float(np.mean(np.abs(e0p - e0t))), e0_p90=float(np.percentile(np.abs(e0p - e0t), 90)),
           log10p0_mae=float(np.mean(np.abs(lp0p - lp0t))), log10p0_p90=float(np.percentile(np.abs(lp0p - lp0t), 90)))
# ---- downstream: SNR network with true vs emulated (e0, p0)
try:
    sys.path.insert(0, "/home/2673888s/LISA_selection_function/lisa_snr_package"); os.environ.setdefault("POPLAR_PATH", "/home/2673888s/poplar")
    from snr_model import load_snr_model, build_features
    sm = load_snr_model(a.snr_model); Xte = X[te]; mu = np.full(len(te), 10.0)
    def snr_dl(e0, p0): return sm.run_on_dataset(build_features(np.stack([10 ** Xte[:, 0], mu, Xte[:, 1], e0, p0, Xte[:, 2], Xte[:, 4]], 1)), n_batches=4).cpu().numpy().ravel()
    s_true, s_emu = snr_dl(d["e0"][te], d["p0"][te]), snr_dl(e0p, 10 ** lp0p)
    dl = np.abs(np.log10(s_emu) - np.log10(s_true)); res.update(snr_log10_median=float(np.median(dl)), snr_log10_p90=float(np.percentile(dl, 90)))
    snr_ok = True
except Exception as ex:
    print("SNR downstream check skipped:", ex); snr_ok = False
json.dump(res, open(f"{mdir}/summary.json", "w"), indent=1); print(json.dumps(res, indent=1))
# ---- plots
h = np.array(hist); fig, ax = plt.subplots(figsize=(6, 4)); ax.semilogy(h[:, 0], h[:, 1], label="train"); ax.semilogy(h[:, 0], h[:, 2], label="val"); ax.legend(); ax.set_xlabel("epoch"); ax.set_ylabel("smooth-L1 (standardised)"); fig.tight_layout(); fig.savefig(f"{pdir}/01_loss.png", dpi=120); plt.close(fig)
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
for k, (t, p_, nm) in enumerate([(e0t, e0p, "e0"), (lp0t, lp0p, "log10 p0")]):
    ax[k].hexbin(t, p_, gridsize=60, bins="log", mincnt=1); lo, hi = t.min(), t.max(); ax[k].plot([lo, hi], [lo, hi], "r", lw=1)
    ax[k].set_xlabel(f"true {nm}"); ax[k].set_ylabel(f"emulated {nm}"); ax[k].set_title(f"{nm}: MAE {np.mean(np.abs(p_-t)):.4f}")
fig.tight_layout(); fig.savefig(f"{pdir}/02_pred_vs_true.png", dpi=120); plt.close(fig)
fig, ax = plt.subplots(2, 5, figsize=(17, 6)); names = ["lgM", "a", "Y0", "e_plunge", "T [yr]"]
for r, (t, p_, nm) in enumerate([(e0t, e0p, "e0 resid"), (lp0t, lp0p, "log10 p0 resid")]):
    for c in range(5):
        ax[r, c].hexbin(X[te][:, c], p_ - t, gridsize=40, bins="log", mincnt=1); ax[r, c].axhline(0, c="r", lw=.8); ax[r, c].set_xlabel(names[c]); ax[r, 0].set_ylabel(nm)
fig.tight_layout(); fig.savefig(f"{pdir}/03_residuals_vs_inputs.png", dpi=110); plt.close(fig)
fig, ax = plt.subplots(1, 3 if snr_ok else 2, figsize=(5 * (3 if snr_ok else 2), 4))
ax[0].hist(np.abs(e0p - e0t), bins=60, log=True); ax[0].set_xlabel("|e0 error|"); ax[1].hist(np.abs(lp0p - lp0t), bins=60, log=True); ax[1].set_xlabel("|log10 p0 error|")
if snr_ok: ax[2].hist(dl, bins=60, log=True); ax[2].axvline(0.083, c="r", ls="--", label="SNR emulator's own median error (0.083)"); ax[2].set_xlabel("|log10 SNR(emulated e0,p0) - log10 SNR(true)|"); ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{pdir}/04_error_histograms_and_snr_impact.png", dpi=120); plt.close(fig)
fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
for c, (x_, nm) in enumerate([(d["e_plunge"], "e_plunge"), (d["T"], "T [yr]"), (d["e0"], "e0"), (np.log10(d["p0"]), "log10 p0")]): ax[c].hist(x_, bins=60); ax[c].set_xlabel(nm)
fig.suptitle(f"emulator training data ({n} forward-FEW rows)"); fig.tight_layout(); fig.savefig(f"{pdir}/00_training_data.png", dpi=110); plt.close(fig)
print("plots ->", pdir)
