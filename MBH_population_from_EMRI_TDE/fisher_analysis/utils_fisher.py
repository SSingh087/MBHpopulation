"""Plotting helpers for Fisher results: corner, Fisher-matrix and covariance plots."""
import os
import numpy as np
import matplotlib.pyplot as plt
import corner


def plot_corner(samples, truths, params, filename, title=None):
    """
    Generate a corner plot of Fisher-sampled parameters.

    Args:
        samples (ndarray): shape (Nsamples, Nparams)
        truths (list): true parameter values (length Nparams)
        params (list): parameter names (length Nparams)
        filename (str): where to save the PDF/PNG
        title (str): optional figure title
    """
    fig = corner.corner(
        samples,
        labels=params,
        truths=truths,
        color="navy",
        truth_color="red",
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".4g",
        hist_kwargs={"density": True, "color": "black"},
    )

    if title:
        fig.suptitle(title, fontsize=14)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_fisher_matrix(F, params, filename, title="Fisher Matrix"):
    """
    Plot a heatmap of the Fisher matrix.

    Args:
        F (ndarray): (N,N) Fisher matrix
        params (list): parameter labels
        filename (str): save location
        title (str): plot title
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(F, cmap="viridis")
    plt.colorbar(label="Fisher Information")
    plt.xticks(np.arange(len(params)), params, rotation=45)
    plt.yticks(np.arange(len(params)), params)
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_covariance_matrix(C, params, filename, title="Covariance Matrix"):
    """
    Plot a heatmap of the covariance matrix.

    Args:
        C (ndarray): (N,N) covariance matrix
        params (list): parameter labels
        filename (str): save location
        title (str): plot title
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(C, cmap="magma")
    plt.colorbar(label="Covariance")
    plt.xticks(np.arange(len(params)), params, rotation=45)
    plt.yticks(np.arange(len(params)), params)
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=200)
    plt.close()
