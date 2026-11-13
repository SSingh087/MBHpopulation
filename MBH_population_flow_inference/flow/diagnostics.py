"""diagnostics.py -- validation-universe diagnostics for a trained flow checkpoint (used by training and by sweep selection)."""
import os
import numpy as np
import torch
from common import DATA, POPS, XTransform, LamScaler, load_consolidated
from flows import ConditionalFlow
import plotting


def load_flow(path, device):
    ck = torch.load(path, map_location=device); c = ck["cfg"]
    fl = ConditionalFlow(c["d"], 6, n_layers=c["n_layers"], hidden=c["hidden"], K=c["K"], tail=c["tail"],
                         ctx_hidden=c.get("ctx_hidden", 64)).to(device)
    fl.load_state_dict(ck["model"]); fl.eval()
    return fl, XTransform(POPS[c["pop"]]["bounds"], ck["x_mean"], ck["x_std"]), ck


def run_diagnostics(pop, model_path, plot_dir, device, max_val_events=100_000, title_extra="", seed=0):
    """One flow draw per validation event (conditioned on that event's own Lambda) vs the simulated events."""
    os.makedirs(plot_dir, exist_ok=True)
    info = POPS[pop]
    meta, x, uid = load_consolidated(pop)
    sel = meta["is_val"][uid]
    x, uid = x[sel], uid[sel]
    if len(x) > max_val_events:
        s = np.random.default_rng(seed).choice(len(x), max_val_events, replace=False); x, uid = x[s], uid[s]
    fl, xt, ck = load_flow(model_path, device)
    lam_s = LamScaler().forward(torch.tensor(meta["lam"][uid])).to(device)
    with torch.no_grad():
        u = torch.cat([fl.sample(lam_s[i:i + 50000]) for i in range(0, len(x), 50000)])
    x_flow = xt.inverse(u).cpu().numpy()
    tt = f"{pop.upper()}{title_extra}"
    plotting.pooled_marginals(x, x_flow, info["labels"], f"{plot_dir}/02_val_pooled_marginals.png", title=f"{tt}: validation universes, one flow draw per event")
    plotting.lambda_sweep(x, x_flow, meta["lam"][uid], info["labels"], f"{plot_dir}/03_val_lambda_sweep.png", title=f"{tt}: x mean/std vs Lambda (validation universes)")
    sub = np.random.default_rng(seed).choice(len(x), min(len(x), 20000), replace=False)
    plotting.corner_overlay(x[sub], x_flow[sub], info["labels"], f"{plot_dir}/04_val_corner.png", title=f"{tt}: validation events vs flow draws")
