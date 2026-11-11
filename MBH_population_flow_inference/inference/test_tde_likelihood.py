#!/usr/bin/env python
"""test_tde_likelihood.py -- checks of the TDE population likelihood on a mock catalogue (+ plots).
  1. pulls (obs-truth)/sigma of the mock (should be N(0,1) in lgM and z)   2. kernel mass inside the integration grid   3. lnL profile of each hyper-parameter through the truth
  python test_tde_likelihood.py --mock mock.npz --tag ztf_seed1
Plots -> <DATA>/plots/inference/<tag>_likelihood_tests.{png,pdf}; numbers -> <DATA>/logs/inference/<tag>_likelihood_tests.json
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tde_pop_lib import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("--mock", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--npt", type=int, default=21)
a = ap.parse_args(); M = np.load(a.mock, allow_pickle=True); survey = str(M["survey"]); obs, cov, tru, lt = M["obs"], M["cov"], M["truth"], M["lam_true"]
pop = TDEPopulation(survey, a.device); like = TDELikelihood(pop, obs, cov); R = {}
sg = np.sqrt(np.stack([cov[:, 0, 0], cov[:, 1, 1]], 1)); pull = (obs - tru) / sg
R["pull_std"] = pull.std(0).tolist(); R["pull_median_abs"] = np.median(np.abs(pull), 0).tolist(); R["frac_kernel_mass_in_grid_lt_0.5"] = float((like.mass_in_grid < 0.5).mean()); R["n_floored"] = like.n_floored
t = time.time(); [like.loglike(lt) for _ in range(5)]; R["seconds_per_eval"] = (time.time() - t) / 5
names = ["A", "B", "scatter", "logM0", "delta", "theta"]; fig, ax = plt.subplots(2, 5, figsize=(17, 6.4)); plt.rcParams.update({"font.size": 8})
for k, (nm, c) in enumerate(zip(["lgM", "z"], ["C0", "C3"])):
    ax[0, k].hist(np.clip(pull[:, k], -6, 6), bins=40, range=(-6, 6), density=True, color=c, alpha=0.6); xx = np.linspace(-6, 6, 200); ax[0, k].plot(xx, np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), "k"); ax[0, k].set_title(f"pull {nm}: std {pull[:, k].std():.2f}"); ax[0, k].set_xlabel("(obs - truth)/sigma")
ax[0, 2].hist(like.mass_in_grid, bins=30, color="grey"); ax[0, 2].set_title("kernel mass inside the grid"); ax[0, 2].set_xlabel("fraction")
ax[0, 3].scatter(tru[:, 1], obs[:, 1], s=4); ax[0, 3].plot([0, 0.5], [0, 0.5], "k"); ax[0, 3].set_ylim(-0.5, 1.5); ax[0, 3].set_xlabel("true z"); ax[0, 3].set_ylabel("observed z")
ax[0, 4].scatter(tru[:, 0], obs[:, 0], s=4); ax[0, 4].plot([4, 7], [4, 7], "k"); ax[0, 4].set_ylim(3, 9); ax[0, 4].set_xlabel("true lgM"); ax[0, 4].set_ylabel("observed lgM")
R["profile"] = {}
for k, nm in enumerate(names):
    lo, hi = PRIOR_BOX[nm]; xs = np.exp(np.linspace(np.log(lo), np.log(hi), a.npt)) if nm == "delta" else np.linspace(lo, hi, a.npt); r = []
    for v in xs: l = lt.copy(); l[k] = v; r.append(like.loglike(l)[0])
    r = np.array(r); r -= r.max(); R["profile"][nm] = dict(x=xs.tolist(), dlnL=r.tolist(), best=float(xs[r.argmax()]), truth=float(lt[k]))
    a_ = ax[1, k if k < 5 else 4] if k < 5 else None
    if k < 5: a_.plot(xs, r, "o-", ms=3); a_.axvline(lt[k], color="r"); a_.set_title(nm); a_.set_ylim(max(r.min(), -30), 1); a_.set_xlabel(nm); a_.set_ylabel("lnL - max (others at truth)")
    else: ax[1, 4].plot(xs, r, "o-", ms=3, color="C1"); ax[1, 4].axvline(lt[k], color="r"); ax[1, 4].set_title("theta (red = truth)"); ax[1, 4].set_ylim(max(r.min(), -30), 1)
fig.suptitle(f"{survey} mock ({like.n} events): likelihood tests; truth in red", y=1.0); fig.tight_layout()
od = f"{DATA}/plots/inference"; os.makedirs(od, exist_ok=True); os.makedirs(f"{DATA}/logs/inference", exist_ok=True)
fig.savefig(f"{od}/{a.tag}_likelihood_tests.png", dpi=130); fig.savefig(f"{od}/{a.tag}_likelihood_tests.pdf"); json.dump(R, open(f"{DATA}/logs/inference/{a.tag}_likelihood_tests.json", "w"), indent=1)
print({k: v for k, v in R.items() if k != "profile"}); [print(nm, "truth", p["truth"], "profile max at", round(p["best"], 3)) for nm, p in R["profile"].items()]
