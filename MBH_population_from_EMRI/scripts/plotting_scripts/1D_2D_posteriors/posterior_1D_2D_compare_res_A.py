import numpy as np
import seaborn as sns
import matplotlib
import h5py
import corner
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

# --- Matplotlib setup ---
matplotlib.rc('font', family='serif', serif=['Computer Modern'], size=28)
matplotlib.rc('text', usetex=True)

pal = sns.color_palette(palette="colorblind").as_hex()

true_x_A = {
        "log10_M": {"xc": 7.5},
        "log10_mu": {"mu": 1.5, "sigma": 0.5, "alpha": -0.5},
        "a": {"alpha": 12, "beta": 8},
        "e0": {"alpha": 8, "beta": 3},
        }

# Ground truths for vertical lines
truths = [
    true_x_A['log10_M']['xc'], true_x_A['log10_mu']['alpha'],
    true_x_A['a']['alpha'], true_x_A['a']['beta'],
    true_x_A['e0']['alpha'], true_x_A['e0']['beta']
]

labels = ["$x_c$", "$\\gamma_{\\mu}$", "$\\alpha_a$", "$\\beta_a$", "$\\alpha_{e0}$", "$\\beta_{e0}$"]

# --- Load posterior samples ---
def load_samples(base_path):
    with h5py.File(f"{base_path}/result.hdf5", "r") as hf:
        ps = hf["posterior_samples"]
        keys = ['xc_M', 'alpha_mu', 'alpha_a', 'beta_a', 'alpha_e0', 'beta_e0']
        return np.vstack([np.array(ps[k]) for k in keys]).T

base_dir = "/data/wiay/postgrads/shashwat/EMRI_data/INFERENCE_DATA/_pop_A_NO_SF"
samples_all = [
    load_samples(f"{base_dir}/1E2_events/inference"),
    load_samples(f"{base_dir}/1E3_events/inference"),
    load_samples(f"{base_dir}/1E4_events/inference"),
]
event_labels = ['$10^2$', '$10^3$', '$10^4$']
colors = [pal[4], pal[2], pal[0]]

# # --- Corner plot setup ---
# default_kwargs = dict(
#     bins=32,
#     smooth=0.9,
#     truth_color='black',
#     plot_density=False,
#     plot_datapoints=False,
#     fill_contours=False,
#     max_n_ticks=4,
#     show_titles=False,
#     label_kwargs=dict(fontsize=26),
#     labelpad=0.2,
#     title_kwargs={"fontsize": 18}
# )

# # --- Corner plot (overlayed for 3 event sets) ---
# fig_corner = corner.corner(
#     samples_all[0], color=pal[4], truths=truths, labels=labels,
#     hist_kwargs=dict(density=True, color=pal[4], linewidth=2),
#     contour_kwargs=dict(linewidths=2), **default_kwargs
# )
# fig_corner = corner.corner(
#     samples_all[1], fig=fig_corner, color=pal[2], truths=truths, labels=labels,
#     hist_kwargs=dict(density=True, color=pal[2], linewidth=2),
#     contour_kwargs=dict(linewidths=2), **default_kwargs
# )
# fig_corner = corner.corner(
#     samples_all[2], fig=fig_corner, color=pal[0], truths=truths, labels=labels,
#     hist_kwargs=dict(density=True, color=pal[0], linewidth=2),
#     contour_kwargs=dict(linewidths=2), **default_kwargs
# )

# --- Legend in upper right (aligned like MIX plot) ---
E2 = mlines.Line2D([], [], color=pal[4], linewidth=2, label='$10^2$')
E3 = mlines.Line2D([], [], color=pal[2], linewidth=2, label='$10^3$')
E4 = mlines.Line2D([], [], color=pal[0], linewidth=2, label='$10^4$')

# fig_corner.legend(
#     handles=[E2, E3, E4],
#     loc='upper right',
#     fontsize=28,
#     title='Injected Events',
#     title_fontsize=32,
#     frameon=False,
#     bbox_to_anchor=(0.85, 0.93)
# )

# fig_corner.savefig("/data/wiay/postgrads/shashwat/EMRI_data/INFERENCE_DATA/PLOTS/2D-posteriors/pop_A_corner.pdf", dpi=500, bbox_inches='tight')
# fig_corner.savefig("/data/wiay/postgrads/shashwat/EMRI_data/INFERENCE_DATA/PLOTS/2D-posteriors/pop_A_corner.png", dpi=500, bbox_inches='tight')


# --- Combined 1D KDE plot grid (like MIX) ---
nrows, ncols = 3, 2
fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 2.6*nrows))
axes = axes.flatten()

# --- Fill all subplots with KDEs ---
y_positions = [0.95, 0.75, 0.55]  # vertical positions for annotations
x_positions = [0.90, 0.90, 0.90, 0.05, 0.05, 0.05] # horizontal positions for annotations
horizontalalignment_pos = ['right', 'right', 'right', 'left', 'left', 'left']

for idx, label in enumerate(labels):
    ax = axes[idx]
    for samples, color, y_pos in zip(samples_all, colors, y_positions):
        sns.kdeplot(samples[:, idx], fill=True, color=color, linewidth=1.8, ax=ax)
        
        q16, q50, q84 = np.percentile(samples[:, idx], [16, 50, 84])
        err_label = f"{q50:.2f}$_{{-{q50 - q16:.2f}}}^{{+{q84 - q50:.2f}}}$"
        print(x_positions[idx], y_pos)
        ax.text(x_positions[idx], y_pos, err_label, color=color, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment=horizontalalignment_pos[idx], fontsize=14)
    
    # Ground truth vertical line
    ax.axvline(truths[idx], color='black', linestyle='--', linewidth=1.5)
    
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.set_xlabel(label, fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.grid(alpha=0.2)
    ax.minorticks_on()

# --- Global legend (top right) ---
# handles = [plt.Line2D([], [], color=c, lw=2, label=lbl) for c, lbl in zip(colors, event_labels)]
# fig.legend(
#     handles=handles,
#     loc='upper right',
#     fontsize=8,
#     # title='Events',
#     title_fontsize=10,
#     frameon=False,
#     # bbox_to_anchor=(0.30, 0.95)
# )

plt.subplots_adjust(left=0.08, right=0.88, top=0.93, bottom=0.01, wspace=0.35, hspace=0.30)
plt.savefig("/data/wiay/postgrads/shashwat/EMRI_data/INFERENCE_DATA/PLOTS/1D-posteriors/pop_A_combined_1D_posteriors.pdf", dpi=300, bbox_inches="tight")
plt.savefig("/data/wiay/postgrads/shashwat/EMRI_data/INFERENCE_DATA/PLOTS/1D-posteriors/pop_A_combined_1D_posteriors.png", dpi=300, bbox_inches="tight")
# plt.show()
