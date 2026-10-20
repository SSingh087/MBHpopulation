"""
Shared style module for all paper plotting scripts.

Every figure is generated at the size at which it is printed in the MNRAS
two-column layout (single column = COL_W, full width = TEXT_W) and the TeX
includes it at width=\\columnwidth / \\textwidth, i.e. with NO rescaling. The
fonts are therefore identical in every figure: same typeface as the paper body
(Times via newtx), axis labels 9 pt, tick labels 8 pt, legends/annotations 8 pt.

Usage:
    setup_style()                                  # once at the top of a script
    fig, ax = plt.subplots(figsize=fig_size(COL_W, 0.75))   # width, aspect (h/w)
    ...
    savefig_all(fig, name)                         # PNG + PDF into PLOTS_DIR
"""

import os
import matplotlib
import matplotlib.pyplot as plt

PLOTS_DIR = "./data/PLOTS"

# MNRAS geometry (inches)
COL_W = 3.375    # \columnwidth
TEXT_W = 6.97    # \textwidth

# Uniform font sizes (pt)
FS_LABEL = 9
FS_TICK = 8
FS_LEG = 8
FS_TEXT = 8

# Scenario names (old M1..M8 -> new naming used in the paper)
SCEN_LABEL = {
    "M1": r"$\star$HU", "M2": r"$\sigma$HU", "M3": r"$\star$NU", "M4": r"$\sigma$NU",
    "M5": r"$\star$HV", "M6": r"$\sigma$HV", "M7": r"$\star$NV", "M8": r"$\sigma$NV",
}


def fig_size(width=COL_W, aspect=0.75):
    """(width, width*aspect) in inches."""
    return (width, width * aspect)


def setup_style(figsize=None, fontsize=FS_LABEL):
    """Configure matplotlib for LaTeX rendering with the paper's Times font and uniform sizes."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{newtxtext,newtxmath}'
    matplotlib.rc('font', family='serif', serif=['Times New Roman', 'Times', 'Nimbus Roman'],
                  size=fontsize)
    matplotlib.rcParams.update({
        'axes.labelsize': FS_LABEL,
        'axes.titlesize': FS_LABEL,
        'xtick.labelsize': FS_TICK,
        'ytick.labelsize': FS_TICK,
        'legend.fontsize': FS_LEG,
        'legend.frameon': False,
        'legend.handlelength': 1.8,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'lines.linewidth': 1.2,
        'figure.figsize': figsize or fig_size(COL_W, 0.75),
    })


def savefig_all(fig, name, dpi=300, tight=False):
    """Save `fig` as both PNG and PDF into PLOTS_DIR under `name` (no extension).

    tight=False keeps the exact figure size (so fonts stay 1:1 in the paper); use
    tight=True only for figures whose artists extend beyond the canvas.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    png_path = os.path.join(PLOTS_DIR, f"{name}.png")
    pdf_path = os.path.join(PLOTS_DIR, f"{name}.pdf")
    kw = dict(bbox_inches='tight', pad_inches=0.02) if tight else {}
    fig.savefig(png_path, dpi=dpi, **kw)
    fig.savefig(pdf_path, **kw)
    print(f"Saved -> {png_path}")
    print(f"Saved -> {pdf_path}")
