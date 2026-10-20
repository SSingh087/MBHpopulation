"""Pipeline impact plots from outputs/impact_*_N{N}_s{SEED}.json. Usage: python plot_impact.py [N] [SEED]"""
import os, sys, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_merger_rates import STYLE, INK, MUTED, OUT   # reuses style; also regenerates rate plots
N, SEED = (sys.argv[1] if len(sys.argv) > 1 else "100000"), (sys.argv[2] if len(sys.argv) > 2 else "42")
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
R = {k: json.load(open(f"{D}/impact_{k}_N{N}_s{SEED}.json")) for k in STYLE}
LAB = {"baseline": "Baseline\n(1+z)$^2$", "mundy17": "Mundy\n1.3", "mundy17_n5e4": "Mundy\n1.4", "oleary_9.7": "EMERGE\n≥9.7", "oleary_10.3": "EMERGE\n≥10.3", "oleary_11.0": "EMERGE\n≥11.0"}

fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))
for a, key, ttl in zip(ax[:2], ("Lambda_TDE", "Lambda_EMRI"), ("Expected TDEs (all z, lgM$_{BH}$ 4–7)", "Expected EMRIs")):
    v = [R[k][key] for k in STYLE]
    bars = a.bar(range(len(v)), v, color=[STYLE[k][0] for k in STYLE], width=0.7)
    for i, (b_, x) in enumerate(zip(bars, v)):
        a.text(i, x, f"{x / v[0]:.2f}×", ha="center", va="bottom", fontsize=9, color=INK)
    a.set_xticks(range(len(v))); a.set_xticklabels([LAB[k] for k in STYLE], fontsize=8.5)
    a.set_ylabel("expected count"); a.set_title(ttl, loc="left"); a.grid(axis="x", visible=False)
    a.set_ylim(0, max(v) * 1.12)
zb = np.array(R["baseline"]["z_bins"]); zc = 0.5 * (zb[1:] + zb[:-1])
for k in STYLE:
    c, ls, lw = STYLE[k]
    ax[2].step(zc, np.array(R[k]["Lambda_TDE_per_zbin"]), where="mid", color=c, ls=ls, lw=lw, label=LAB[k].replace("\n", " "))
ax[2].set_yscale("log"); ax[2].set_xlabel("galaxy redshift"); ax[2].set_ylabel("expected TDEs per z-bin")
ax[2].set_title("Where the TDEs are", loc="left"); ax[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/pipeline_impact_N{N}.png", dpi=150)

with open(f"{D}/summary_N{N}_s{SEED}.csv", "w") as f:
    f.write("model,Lambda_TDE,TDE_ratio,Lambda_EMRI,EMRI_ratio,active_fraction,censored_fraction,N_eff_TDE\n")
    for k in STYLE:
        r = R[k]; b = R["baseline"]
        f.write(f"{k},{r['Lambda_TDE']:.6g},{r['Lambda_TDE']/b['Lambda_TDE']:.4f},{r['Lambda_EMRI']:.6g},{r['Lambda_EMRI']/b['Lambda_EMRI']:.4f},{r['active_fraction']:.4f},{r['censored_frac_zLMM_at_zmax']:.4f},{r['n_eff_TDE']:.0f}\n")
print("saved")
