#!/usr/bin/env python
"""Overview plots of the consolidated training set (run after build_dataset.py). Output: <DATA>/plots/dataset/"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, LAMBDA_NAMES, LAM_LABELS, POPS, FIDUCIAL, load_consolidated

out = f"{DATA}/plots/dataset"; os.makedirs(out, exist_ok=True)
meta, xe, ue = load_consolidated("emri"); _, xt, ut = load_consolidated("tde")
lam = meta["lam"]; U = len(lam); val = meta["is_val"]
print(U, "universes,", len(xe), "EMRI events,", len(xt), "TDE events")
lg = lambda v, k: np.log(v) if LAMBDA_NAMES[k] == "delta" else v
lab = lambda k: ("ln " if LAMBDA_NAMES[k] == "delta" else "") + LAM_LABELS[k]

# 1. Lambda coverage corner
sub = np.random.default_rng(0).choice(U, 2500, replace=False)
fig, ax = plt.subplots(6, 6, figsize=(14, 14))
for i in range(6):
    for j in range(6):
        a = ax[i, j]
        if j > i: a.axis("off"); continue
        if i == j: a.hist(lg(lam[:, i], i), bins=30, color="C0", alpha=.7)
        else: a.scatter(lg(lam[sub, j], j), lg(lam[sub, i], i), s=1.5, alpha=.4)
        if i == 5: a.set_xlabel(lab(j))
        if j == 0 and i > 0: a.set_ylabel(lab(i))
fig.suptitle(f"Lambda coverage of the {U} training universes (delta log-uniform)"); fig.tight_layout(); fig.savefig(f"{out}/01_lambda_coverage.png", dpi=110); plt.close(fig)

# 2. Expected counts vs each Lambda parameter
fig, ax = plt.subplots(2, 6, figsize=(17, 6))
for j in range(6):
    for r, (key, nm) in enumerate((("Lambda_tot", "EMRI"), ("Lambda_tot_tde", "TDE"))):
        ax[r, j].scatter(lam[:, j], meta[key], s=2, alpha=.25); ax[r, j].set_yscale("log")
        if LAMBDA_NAMES[j] == "delta": ax[r, j].set_xscale("log")
        ax[r, j].axvline(FIDUCIAL[LAMBDA_NAMES[j]], color="r", ls="--", lw=1)
        ax[r, j].set_xlabel(LAM_LABELS[j]); ax[r, 0].set_ylabel(f"expected {nm} sources")
fig.suptitle("Expected source counts vs Lambda (red dashed: fiducial)"); fig.tight_layout(); fig.savefig(f"{out}/02_expected_counts_vs_lambda.png", dpi=110); plt.close(fig)

# 3. Counts and importance-sampling quality
fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
ax[0].hist(meta["n_emri"], bins=50); ax[0].set_xlabel("realised EMRI sources per universe"); ax[0].axvline(5, color="r", ls=":")
ax[1].hist(np.log10(meta["n_tde_total"]), bins=50); ax[1].set_xlabel(r"$\log_{10}$ realised TDE sources per universe")
ax[2].hist(meta["n_eff"], bins=50); ax[2].set_xlabel(r"$n_{\rm eff}$ EMRI (importance weights)"); ax[2].axvline(30, color="r", ls=":")
ax[3].hist(meta["n_eff_tde"], bins=50); ax[3].set_xlabel(r"$n_{\rm eff}$ TDE")
fig.suptitle(f"empty EMRI universes: {(meta['n_emri']==0).sum()}, <5 events: {(meta['n_emri']<5).sum()}, n_eff<30: {(meta['n_eff']<30).sum()}")
fig.tight_layout(); fig.savefig(f"{out}/03_counts_and_neff.png", dpi=110); plt.close(fig)

# 4/5. Pooled marginals and sensitivity to each Lambda parameter
for pop, x, uid in (("emri", xe, ue), ("tde", xt, ut)):
    info = POPS[pop]; d = x.shape[1]; labels = info["labels"]
    fig, ax = plt.subplots(1, d, figsize=(3.8 * d, 3.2), squeeze=False)
    for i in range(d):
        ax[0, i].hist(x[:, i], bins=60, density=True, color="C0", alpha=.7); ax[0, i].set_xlabel(labels[i]); ax[0, i].set_yticks([])
    fig.suptitle(f"{pop.upper()}: pooled marginals over all training universes ({len(x)} events)"); fig.tight_layout(); fig.savefig(f"{out}/04_{pop}_pooled_marginals.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(d, 6, figsize=(17, 2.7 * d), squeeze=False); ev_lam = lam[uid]
    for j in range(6):
        v = lg(ev_lam[:, j], j); q1, q3 = np.quantile(v, [0.25, 0.75])
        for i in range(d):
            lo, hi = x[:, i].min(), x[:, i].max()
            ax[i, j].hist(x[v <= q1, i], bins=40, range=(lo, hi), density=True, histtype="step", lw=2, color="C0", label="lowest 25% of param")
            ax[i, j].hist(x[v >= q3, i], bins=40, range=(lo, hi), density=True, histtype="step", lw=2, color="C3", label="highest 25%")
            ax[i, j].set_yticks([])
            if i == 0: ax[i, j].set_title(LAM_LABELS[j])
            if j == 0: ax[i, j].set_ylabel(labels[i])
    ax[0, 0].legend(fontsize=7)
    fig.suptitle(f"{pop.upper()}: dependence of each x distribution on each Lambda parameter (what the flow must learn)")
    fig.tight_layout(); fig.savefig(f"{out}/05_{pop}_sensitivity_to_lambda.png", dpi=110); plt.close(fig)

# 6. Train / validation consistency
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].hist(lam[~val, 0], bins=30, density=True, alpha=.5, label=f"train ({(~val).sum()})"); ax[0].hist(lam[val, 0], bins=30, density=True, alpha=.5, label=f"val ({val.sum()})"); ax[0].set_xlabel("A"); ax[0].legend()
ax[1].hist(np.log10(meta["Lambda_tot"][~val] + 1e-3), bins=30, density=True, alpha=.5); ax[1].hist(np.log10(meta["Lambda_tot"][val] + 1e-3), bins=30, density=True, alpha=.5); ax[1].set_xlabel(r"$\log_{10}$ expected EMRIs")
fig.suptitle("train vs validation universes"); fig.tight_layout(); fig.savefig(f"{out}/06_train_val_split.png", dpi=110); plt.close(fig)
print("plots in", out, sorted(os.listdir(out)))
