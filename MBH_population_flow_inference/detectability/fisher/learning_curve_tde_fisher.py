"""Plot validation NLL vs training-set size for the TDE Fisher-error flow learning curves (models/detectability/tde_fisher_lc/<S>_<cfg>_f<frac>)."""
import glob, json, os
import numpy as np
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for a, S in zip(ax, ["ZTF", "LSST"]):
    for cfg, c in [("05", "C3"), ("11", "C0")]:
        pts = []
        for d in sorted(glob.glob(f"{DATA}/models/detectability/tde_fisher_lc/{S}_{cfg}_f*")):
            f = f"{d}/summary.json"
            if os.path.exists(f): s = json.load(open(f)); pts.append((s["n_train"], s["best_val"], s["n_params"]))
        if not pts: continue
        pts.sort(); n, v, npar = np.array(pts).T
        a.plot(n, v, "o-", color=c, label=f"cfg {cfg} ({int(npar[0])/1e6:.1f}M params)")
        for ni, vi in zip(n, v): print(f"{S} cfg {cfg}  n_train {int(ni):6d}  val NLL {vi:.3f}")
    a.set_xscale("log"); a.set_xlabel("training events"); a.set_title(S); a.legend()
ax[0].set_ylabel("best validation NLL (lower = better)")
fig.suptitle("TDE Fisher-error flow: learning curve (fixed val split)"); fig.tight_layout()
os.makedirs(f"{DATA}/plots/detectability/tde_fisher_flow", exist_ok=True); fig.savefig(f"{DATA}/plots/detectability/tde_fisher_flow/learning_curve.png", dpi=130)
