#!/usr/bin/env python
"""Compare the original and the z-extended ZTF classifier on the SAME held-out labelled events (tde_classifier_test.py sets: prior, cat, far; none used in training).
Writes <DATA>/plots/detectability/tde_clf_test/tde_clf_ext_compare.{png,pdf} and <DATA>/logs/tde_clf_ext_compare.json"""
import glob, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fisher")); from tde_classifier import TDEClassifier
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"; OUT = f"{DATA}/simulations/tde_clf_test"
C = {"old": TDEClassifier("ZTF"), "new (z<=2)": TDEClassifier("ZTFext")}; R = {}
Dset = {st: (lambda fs: {k: np.concatenate([np.load(f)[k] for f in fs]) for k in np.load(fs[0]).files})(sorted(glob.glob(f"{OUT}/ZTF_{st}_*.npz"))) for st in ("prior", "cat", "far")}
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3)); col = {"old": "#D55E00", "new (z<=2)": "#0072B2"}
for st, D in Dset.items():
    ok = D["y"] >= 0; D = {k: v[ok] for k, v in D.items()}; R[st] = {}
    for nm, clf in C.items():
        p = clf.prob(D); y = D["y"]; pred = p > 0.5
        R[st][nm] = dict(n=int(len(y)), prevalence=float(y.mean()), mean_p=float(p.mean()), mean_p_over_prev=float(p.mean() / max(y.mean(), 1e-9)), accuracy=float((pred == y).mean()),
                         auc=float(roc_auc_score(y, p)), brier=float(((p - y) ** 2).mean()), purity=float(y[pred].mean()) if pred.any() else None, completeness=float(pred[y == 1].mean()))
        if st == "far" or st == "cat":
            zz = np.linspace(0, 2, 21); bi = np.digitize(D["z_gal"], zz) - 1
            m = [p[bi == b].mean() if (bi == b).sum() > 20 else np.nan for b in range(20)]; ax[0 if st == "cat" else 1].plot(0.5 * (zz[1:] + zz[:-1]), m, "--", color=col[nm], label=f"{nm} p")
            if nm == "old": ax[0 if st == "cat" else 1].plot(0.5 * (zz[1:] + zz[:-1]), [y[bi == b].mean() if (bi == b).sum() > 20 else np.nan for b in range(20)], "o", ms=3, color="k", label="simulated")
for a_, t in zip(ax[:2], ("catalogue events, z<=0.5", "z in (0.5, 2] (outside old training range)")): a_.set_xlabel("$z$"); a_.set_ylabel("detected fraction"); a_.set_title(t, fontsize=8); a_.legend(fontsize=6)
names = list(R); x = np.arange(len(names)); w = 0.35
for i, nm in enumerate(C): ax[2].bar(x + (i - 0.5) * w, [R[s][nm]["mean_p_over_prev"] for s in names], w, color=col[nm], label=nm)
ax[2].axhline(1, color="k", lw=0.6); ax[2].set_xticks(x); ax[2].set_xticklabels(names); ax[2].set_ylabel("mean p / simulated fraction"); ax[2].legend(fontsize=6)
fig.tight_layout(); fig.savefig(f"{DATA}/plots/detectability/tde_clf_test/tde_clf_ext_compare.png", dpi=150); fig.savefig(f"{DATA}/plots/detectability/tde_clf_test/tde_clf_ext_compare.pdf")
json.dump(R, open(f"{DATA}/logs/tde_clf_ext_compare.json", "w"), indent=1)
for st in R:
    for nm, v in R[st].items(): print(st, nm, {k: (round(x_, 4) if isinstance(x_, float) else x_) for k, x_ in v.items()})
