"""marginal_plots.py -- 1-D marginal distributions, simulator vs flow, each with a ratio panel.
(a) conditional marginals p(x_i | Lambda) at the 5 held-out test Lambdas (cached held-out sims, no new simulation),
(b) Lambda-marginalised (pooled over validation universes) p(x_i), linear and log y.
Quick (seconds). Output: plots/marginals<tag>/"""
import argparse, os, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import DATA, POPS, FIDUCIAL, LamScaler, lam_dict_to_array, load_consolidated
from diagnostics import load_flow
ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="_best"); ap.add_argument("--n_univ", type=int, default=20)
ap.add_argument("--max_n", type=int, default=100_000); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args(); dev = torch.device(a.device); ls = LamScaler()
out = f"{DATA}/plots/marginals{a.tag}"; os.makedirs(out, exist_ok=True)
TEST = {"fiducial": FIDUCIAL,
        "heavy_corner": dict(A=8.10, B=3.50, scatter=0.70, logM0=8.70, delta=10.0, theta=1.8),
        "light_corner": dict(A=7.65, B=5.60, scatter=0.40, logM0=10.2, delta=1.0, theta=0.5),
        "highA_steep": dict(A=8.10, B=5.50, scatter=0.40, logM0=8.60, delta=15.0, theta=0.3),
        "lowA_shallow": dict(A=7.60, B=3.40, scatter=0.70, logM0=9.80, delta=2.0, theta=1.2)}

def hist_ratio(ax, axr, xs, xf, label, logy=False, bins=40):
    lo, hi = np.percentile(np.concatenate([xs, xf]), [0.05, 99.95]); e = np.linspace(lo, hi, bins + 1); c = 0.5 * (e[1:] + e[:-1])
    hs, _ = np.histogram(xs, e); hf, _ = np.histogram(xf, e)
    ds, df = hs / len(xs) / np.diff(e), hf / len(xf) / np.diff(e)
    ax.step(c, ds, where="mid", lw=2, color="C0", label="simulated"); ax.step(c, df, where="mid", lw=2, color="C1", label="flow")
    ax.errorbar(c, ds, yerr=np.sqrt(np.maximum(hs, 1)) / len(xs) / np.diff(e), fmt="none", ecolor="C0", alpha=.5)
    if logy: ax.set_yscale("log"); ax.set_ylim(max(ds[ds > 0].min() * .5, 1e-5), ds.max() * 3)
    else: ax.set_yticks([])
    ok = hs > 0; r = np.full_like(ds, np.nan); r[ok] = df[ok] / ds[ok]; er = np.full_like(ds, np.nan); er[ok] = 1 / np.sqrt(hs[ok])
    axr.axhline(1, c="k", lw=.8); axr.errorbar(c[ok], r[ok], yerr=er[ok], fmt="o", ms=3, color="C1"); axr.set_ylim(0.5, 1.5)
    axr.set_xlabel(label); ax.set_xticklabels([])

for pop, info in POPS.items():
    mp_ = f"{DATA}/models/flow_{pop}{a.tag}/best.pt"
    if not os.path.exists(mp_): continue
    fl, xt, ck = load_flow(mp_, dev); d = len(info["names"]); rng = np.random.default_rng(0)
    def draw(ctx):
        with torch.no_grad(): u = torch.cat([fl.sample(ctx[i:i + 50000]) for i in range(0, len(ctx), 50000)])
        return xt.inverse(u).cpu().numpy()
    # (a) conditional marginals, one figure per test Lambda
    for name, lam in TEST.items():
        f = f"{DATA}/simulations/heldout/{name}_n{a.n_univ}.npz"
        if not os.path.exists(f): print("missing", f); continue
        xs = np.load(f)[pop]; xs = xs[rng.permutation(len(xs))[:a.max_n]]
        if len(xs) < 20: continue
        xf = draw(ls.forward(torch.tensor(lam_dict_to_array(lam))[None]).to(dev).repeat(len(xs), 1))
        fig, axs = plt.subplots(2, d, figsize=(3.6 * d, 4.6), gridspec_kw=dict(height_ratios=[3, 1], hspace=.05), squeeze=False)
        for i in range(d): hist_ratio(axs[0, i], axs[1, i], xs[:, i], xf[:, i], info["labels"][i]); axs[1, i].set_ylabel("flow/sim" if i == 0 else "")
        axs[0, 0].legend(fontsize=8); fig.suptitle(f"{pop.upper()} p(x | Lambda) @ {name}: N={len(xs)}  " + ", ".join(f"{k}={v:.2f}" for k, v in lam.items()), fontsize=9)
        fig.savefig(f"{out}/{pop}_cond_{name}.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    # (b) Lambda-marginalised, validation universes
    meta, x, uid = load_consolidated(pop); sel = meta["is_val"][uid]; x, uid = x[sel], uid[sel]
    s = rng.permutation(len(x))[:a.max_n]; x, uid = x[s], uid[s]
    xf = draw(ls.forward(torch.tensor(meta["lam"][uid])).to(dev))
    fig, axs = plt.subplots(3, d, figsize=(3.6 * d, 7), gridspec_kw=dict(height_ratios=[3, 3, 1], hspace=.08), squeeze=False)
    for i in range(d):
        hist_ratio(axs[0, i], axs[2, i], x[:, i], xf[:, i], info["labels"][i])
        hist_ratio(axs[1, i], axs[2, i], x[:, i], xf[:, i], info["labels"][i], logy=True); axs[1, i].set_xticklabels([])
        axs[2, i].set_ylabel("flow/sim" if i == 0 else "")
    axs[0, 0].set_ylabel("linear"); axs[1, 0].set_ylabel("log"); axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"{pop.upper()}: Lambda-marginalised p(x) (validation universes, prior-weighted by event counts)")
    fig.savefig(f"{out}/{pop}_marginalised_over_lambda.png", dpi=120, bbox_inches="tight"); plt.close(fig)
print("plots ->", out)
