"""plotting.py -- reusable diagnostic plots (corner overlays, Lambda-sweep of x moments, loss curves)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from common import LAMBDA_NAMES, LAM_LABELS

C_DATA, C_FLOW = "#1f77b4", "#ff7f0e"


def _levels(h, masses=(0.68, 0.95)):
    s = np.sort(h.ravel())[::-1]; c = np.cumsum(s) / s.sum()
    return sorted(s[np.searchsorted(c, m)] for m in masses)


def corner_overlay(a, b, labels, fname, la="simulated", lb="flow", title=None, bins=40):
    """a (n,d) vs b (m,d): histograms on the diagonal, 68/95% contours below."""
    d = a.shape[1]
    fig, ax = plt.subplots(d, d, figsize=(2.6 * d, 2.6 * d), squeeze=False)
    lims = [(min(a[:, i].min(), b[:, i].min()), max(a[:, i].max(), b[:, i].max())) for i in range(d)]
    for i in range(d):
        for j in range(d):
            x = ax[i, j]
            if j > i: x.axis("off"); continue
            if i == j:
                x.hist(a[:, i], bins=bins, range=lims[i], density=True, histtype="step", lw=2, color=C_DATA, label=f"{la} (n={len(a)})")
                x.hist(b[:, i], bins=bins, range=lims[i], density=True, histtype="step", lw=2, color=C_FLOW, label=f"{lb} (n={len(b)})")
                x.set_yticks([])
            else:
                for arr, col in ((a, C_DATA), (b, C_FLOW)):
                    h, xe, ye = np.histogram2d(arr[:, j], arr[:, i], bins=30, range=[lims[j], lims[i]])
                    h = gaussian_filter(h, 1.0)
                    if h.sum() > 0:
                        x.contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), h.T, levels=_levels(h) + [h.max() * 1.01], colors=[col], linewidths=[1.2, 1.8])
                x.set_xlim(lims[j]); x.set_ylim(lims[i])
            if i == d - 1: x.set_xlabel(labels[j])
            else: x.set_xticklabels([])
            if j == 0 and i > 0: x.set_ylabel(labels[i])
            elif j > 0: x.set_yticklabels([])
    ax[0, 0].legend(fontsize=8, loc="upper right")
    if title: fig.suptitle(title, y=1.0)
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)


def lambda_sweep(x_data, x_flow, lam_ev, labels, fname, nbins=6, title=None):
    """Mean +- std of each x variable versus each Lambda parameter (events binned by Lambda quantiles): data vs flow.
    lam_ev (n,6): the Lambda of each event; x_data / x_flow (n,d) are paired by event Lambda."""
    d = x_data.shape[1]
    fig, ax = plt.subplots(d, 6, figsize=(17, 2.6 * d), squeeze=False)
    for j in range(6):
        v = np.log(lam_ev[:, j]) if LAMBDA_NAMES[j] == "delta" else lam_ev[:, j]
        edges = np.quantile(v, np.linspace(0, 1, nbins + 1)); idx = np.clip(np.digitize(v, edges[1:-1]), 0, nbins - 1)
        cen = np.array([np.median(v[idx == k]) for k in range(nbins)])
        for i in range(d):
            for arr, col, lab in ((x_data, C_DATA, "simulated"), (x_flow, C_FLOW, "flow")):
                m = np.array([arr[idx == k, i].mean() for k in range(nbins)]); s = np.array([arr[idx == k, i].std() for k in range(nbins)])
                ax[i, j].plot(cen, m, "-o", color=col, ms=4, label=lab if (i == 0 and j == 0) else None)
                ax[i, j].fill_between(cen, m - s, m + s, color=col, alpha=0.15)
            if i == d - 1: ax[i, j].set_xlabel(("ln " if LAMBDA_NAMES[j] == "delta" else "") + LAM_LABELS[j])
            if j == 0: ax[i, j].set_ylabel(labels[i])
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(title or "x moments vs Lambda (line: mean, band: std)", y=1.0)
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)


def pooled_marginals(x_data, x_flow, labels, fname, title=None, bins=50):
    d = x_data.shape[1]
    fig, ax = plt.subplots(1, d, figsize=(3.6 * d, 3.2), squeeze=False)
    for i in range(d):
        lo, hi = min(x_data[:, i].min(), x_flow[:, i].min()), max(x_data[:, i].max(), x_flow[:, i].max())
        ax[0, i].hist(x_data[:, i], bins=bins, range=(lo, hi), density=True, histtype="step", lw=2, color=C_DATA, label="simulated")
        ax[0, i].hist(x_flow[:, i], bins=bins, range=(lo, hi), density=True, histtype="step", lw=2, color=C_FLOW, label="flow")
        ax[0, i].set_xlabel(labels[i]); ax[0, i].set_yticks([])
    ax[0, 0].legend(fontsize=8)
    if title: fig.suptitle(title)
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)


def loss_curves(hist, fname, title=None):
    ep = hist["epoch"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    ax[0].plot(ep, hist["train"], label="train"); ax[0].plot(ep, hist["val"], label="validation"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("NLL (standardised space)"); ax[0].legend()
    best = int(np.argmin(hist["val"])); ax[0].axvline(ep[best], color="k", ls=":", lw=1); ax[0].set_title(f"best val {hist['val'][best]:.4f} @ epoch {ep[best]}")
    lo = min(min(hist["train"]), min(hist["val"])); ax[1].plot(ep, np.array(hist["train"]) - lo + 1e-3, label="train"); ax[1].plot(ep, np.array(hist["val"]) - lo + 1e-3, label="val")
    ax[1].set_yscale("log"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("NLL - min + 1e-3"); ax[1].set_title("excess loss (log)"); ax[1].legend()
    ax[2].semilogy(ep, hist["lr"], color="C2"); ax[2].set_xlabel("epoch"); ax[2].set_ylabel("learning rate")
    ax2 = ax[2].twinx(); ax2.plot(ep, hist["time"], color="C3", alpha=.6); ax2.set_ylabel("epoch time [s]", color="C3")
    if title: fig.suptitle(title)
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)
