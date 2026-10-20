"""Paper figure: initial EMRI eccentricity distribution versus MBH mass."""
import sys, os
sys.path.insert(0, os.path.abspath('./'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm

from plot_style import setup_style, savefig_all, fig_size, COL_W

setup_style()


def emri_parameters(M):
    """
    Returns alpha1, beta1, alpha2, beta2, w for a given MBH mass M.
    """

    lgM = np.log10(M)

    # ----------------------------
    # Classical EMRIs (stable peak ~0.2)
    # ----------------------------
    alpha1 = 4.0 + 0.5 * (lgM - 6)
    beta1  = 10.0 - 0.8 * (lgM - 6)

    # ----------------------------
    # Wide / cliffhanger EMRIs
    # ----------------------------
    alpha2 = 3.0 - 0.3 * (lgM - 6)
    beta2  = 6.5 + 0.5 * (lgM - 6)

    # avoid invalid values

    alpha1 = max(alpha1, 1.0)
    beta1  = max(beta1,  1.0)
    alpha2 = max(alpha2, 1.0)
    beta2  = max(beta2,  1.0)


    # ----------------------------
    # Mixing weight (strong MBH dependence)
    # ----------------------------
    M0 = 1e5
    gamma = 1.2
    w = 1.0 / (1.0 + (M0 / M)**gamma)

    return alpha1, beta1, alpha2, beta2, w

def emri_eccentricity_pdf(e, M):
    """
    p(e | M_BH) for EMRIs (mixture of classical + wide).
    """

    a1, b1, a2, b2, w = emri_parameters(M)

    return (
        w * beta.pdf(e, a1, b1)
        + (1 - w) * beta.pdf(e, a2, b2)
    )

def sample_emri_eccentricity(M, size=1):
    """
    Draw samples of eccentricity at plunge for a given MBH mass.
    """

    a1, b1, a2, b2, w = emri_parameters(M)

    # choose component
    is_classical = np.random.rand(size) < w

    e = np.zeros(size)

    # classical
    n1 = np.sum(is_classical)
    if n1 > 0:
        e[is_classical] = np.random.beta(a1, b1, n1)

    # wide
    n2 = size - n1
    if n2 > 0:
        e[~is_classical] = np.random.beta(a2, b2, n2)

    return e if size > 1 else e[0]


# ----------------------------------
# Masses to plot
# ----------------------------------
M_vals = np.logspace(4, 6.5, 10)
e_vals = np.linspace(1e-4, 1-1e-4, 500)

# ----------------------------------
# Colormap for MBH mass
# ----------------------------------
cmap = plt.cm.magma
norm = LogNorm(vmin=M_vals.min(), vmax=M_vals.max())

fig, ax = plt.subplots(figsize=fig_size(COL_W, 0.8))

for M in M_vals:

    a1, b1, a2, b2, w = emri_parameters(M)

    color = cmap(norm(M))

    # Classical
    ax.plot(
        e_vals,
        beta.pdf(e_vals, a1, b1) / max(emri_eccentricity_pdf(e_vals, M)),  # normalize to max of total
        ls='--',
        lw=0.8,
        color=color,
        alpha=0.9,
    )

    # Wide / cliffhanger
    ax.plot(
        e_vals,
        beta.pdf(e_vals, a2, b2) / max(emri_eccentricity_pdf(e_vals, M)),  # normalize to max of total
        ls=':',
        lw=0.8,
        color=color,
        alpha=0.9,
    )

    ax.plot(
        e_vals,
        emri_eccentricity_pdf(e_vals, M),
        ls='-',
        lw=1.4,
        color=color,
        alpha=0.9)

# ----------------------------------
# Colorbar = MBH mass
# ----------------------------------
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label(r'$M_{\rm BH}\,[M_\odot]$')

# ----------------------------------
# Line-style legend
# ----------------------------------
style_handles = [
    Line2D([0], [0], color='k', lw=1.2, ls='--',
           label='Classical EMRI'),
    Line2D([0], [0], color='k', lw=1.2, ls=':',
           label='Wide / Cliffhanger EMRI'),
    Line2D([0], [0], color='k', lw=1.2, ls='-',
           label='Combined'),
]

style_legend = ax.legend(
    handles=style_handles,
    loc='upper right',
)


mass_handles = [
    Line2D([0], [0], color=cmap(norm(1e4)), lw=1.8,
           label=r'$10^4\,M_\odot$'),
    Line2D([0], [0], color=cmap(norm(1e5)), lw=1.8,
           label=r'$10^5\,M_\odot$'),
    Line2D([0], [0], color=cmap(norm(1e6)), lw=1.8,
           label=r'$10^6\,M_\odot$'),
]

mass_legend = ax.legend(
    handles=mass_handles,
    loc='upper right', bbox_to_anchor=(1.0, 0.62),
    title=r'$M_{\rm BH}$', labelspacing=0.25
)

ax.add_artist(mass_legend)
ax.add_artist(style_legend)

ax.add_artist(style_legend)

# ----------------------------------
# Labels
# ----------------------------------
ax.set_xlabel(r'Eccentricity at plunge, $e_{\rm pl}$')
ax.set_ylabel(r'$p(e_{\rm pl}|M_{\rm BH})$')
# ax.set_title('EMRI eccentricity distributions')
ax.set_xlim(0, 1)
ax.set_ylim(top=4.0)
plt.tight_layout()

savefig_all(fig, 'emri_eccentricity_vs_MBH')
plt.close(fig)