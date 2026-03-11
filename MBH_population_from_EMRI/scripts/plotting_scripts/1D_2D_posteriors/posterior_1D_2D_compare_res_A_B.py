import numpy as np
import seaborn as sns
import matplotlib
import h5py
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import corner

# --- Matplotlib setup ---
matplotlib.rc('font', family='serif', serif=['Computer Modern'], size=22)
matplotlib.rc('text', usetex=True)

pal = sns.color_palette(palette="colorblind").as_hex()

# --- Ground truths ---
true_x_MIX = {
    "log10_M": {"xc_A": 7.5, "lam_B": -1.5},
    "log10_mu": {"mu_A": 1.5, "sigma_A": 0.5, "alpha_A": -0.5, "lam_B": -3.0},
    "a": {"alpha_A": 12, "beta_A": 8, "mu_B": 0.3, "sigma_B": 0.01},
    "e0": {"alpha_A": 8, "beta_A": 3, "UNIFORM_B": {}},
    "Y0": {},
    "dist": {"lam": 3},
    "qS": {}, "phiS": {}, "qK": {}, "phiK": {},
    "Phi_phi0": {}, "Phi_theta0": {}, "Phi_r0": {}, "T": {}
}

truths = [
    true_x_MIX['log10_M']['xc_A'],
    true_x_MIX['log10_mu']['alpha_A'],
    true_x_MIX['a']['alpha_A'],
    true_x_MIX['a']['beta_A'],
    true_x_MIX['e0']['alpha_A'],
    true_x_MIX['e0']['beta_A'],
    true_x_MIX['log10_M']['lam_B'],
    true_x_MIX['log10_mu']['lam_B'],
    true_x_MIX['a']['mu_B'],
    true_x_MIX['a']['sigma_B'],
    0.7
]

labels = [
    "$x_c^A$", "$\\gamma^A_{\\mu}$", "$\\alpha^A_a$", "$\\beta^A_a$",
    "$\\alpha^A_{e0}$", "$\\beta^A_{e0}$", "$\\Lambda^B_M$", "$\\Lambda^B_{\\mu}$",
    "$\\mu^B_a$", "$\\sigma^B_a$", "$w$"
]

# --- Function to load samples ---
def load_samples(base_path):
    with h5py.File(f"{base_path}/result.hdf5", "r") as hf:
        ps = hf["posterior_samples"]
        keys = [
            'A_xc_M', 'A_alpha_mu', 'A_alpha_a', 'A_beta_a',
            'A_alpha_e0', 'A_beta_e0', 'B_lambda_M', 'B_lambda_mu',
            'B_mu_a', 'B_sigma_a', 'weight'
        ]
        return np.vstack([np.array(ps[k]) for k in keys]).T

# --- Load all sample sets ---
event_counts = ['1E2', '1E3', '1E4']
base_dir = "/home/singh087/UofG/work/Code/test_EMRI_population/HPC_DATA/_pop_MIX_A_B_NO_SF"
samples_all = [
    load_samples(f"{base_dir}/{ev}_events/inference") for ev in event_counts
]
event_labels = ['$10^2$', '$10^3$', '$10^4$']
colors = [pal[4], pal[2], pal[0]]


# ----------- Plot ythe corner plots ------------


default_kwargs = dict(
    bins=32,
    smooth=0.9,
    truth_color='black',
    # quantiles=[0.16, 0.84],
    # levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=False,
    max_n_ticks = 4,
    show_titles=False,
    label_kwargs=dict(fontsize=26),
    labelpad=0.2,
    title_kwargs={"fontsize": 18}
)

fig_corner = corner.corner(
    samples_all[0], color=pal[4], truths=truths, labels=labels, hist_kwargs=dict(density=True,color=pal[4], linewidth=2),contour_kwargs=dict(linewidths=2), **default_kwargs
)

fig_corner = corner.corner(
    samples_all[1], fig=fig_corner, color=pal[2], truths=truths, labels=labels, hist_kwargs=dict(density=True, color=pal[2], linewidth=2), contour_kwargs=dict(linewidths=2), **default_kwargs
)

fig_corner = corner.corner(
    samples_all[2], fig=fig_corner, color=pal[0], truths=truths, labels=labels, hist_kwargs=dict(density=True, color=pal[0], linewidth=2), contour_kwargs=dict(linewidths=2), **default_kwargs
)

E2 = mlines.Line2D([],[], color=pal[4], linewidth=2,label='$10^2$')
E3 = mlines.Line2D([],[], color=pal[2], linewidth=2,label='$10^3$')
E4 = mlines.Line2D([],[], color=pal[0], linewidth=2,label='$10^4$')

fig_corner.legend(handles=[E2, E3, E4], loc='upper right', fontsize=40, title_fontsize=50, frameon=False, title='Injected Events', bbox_to_anchor=(0.85, 0.93))

fig_corner.savefig(f'/data/wiay/postgrads/shashwat/EMRI_data/INFERENCE_DATA/PLOTS/2D-posteriors/pop_MIX_A_B.pdf',dpi=500, bbox_inches='tight')
fig_corner.savefig(f'/data/wiay/postgrads/shashwat/EMRI_data/INFERENCE_DATA/PLOTS/2D-posteriors/pop_MIX_A_B.png',dpi=500, bbox_inches='tight')



# --- Combined KDE Figure ---
nrows, ncols = 3, 4  # total grid (12 subplots)
fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 2.5*nrows))
axes = axes.flatten()

# --- Reserve top-left subplot for text ---
label_index = ncols - 1  # top-right corner
axes[label_index].axis('off')

# --- Fill remaining subplots with KDEs ---
y_positions = [0.95, 0.75, 0.55]  # vertical positions for annotations
plot_indices = [i for i in range(len(axes)) if i != label_index]
for idx, label in enumerate(labels):
    if idx >= len(plot_indices):
        break
    ax = axes[plot_indices[idx]]
    for samples, color, ev_label, y_pos in zip(samples_all, colors, event_labels, y_positions):
        sns.kdeplot(samples[:, idx], fill=True, color=color, linewidth=1.8, ax=ax)
        q16, q50, q84 = np.percentile(samples[:, idx], [16, 50, 84])
        err_label = f"{q50:.2f}$_{{-{q50 - q16:.2f}}}^{{+{q84 - q50:.2f}}}$"
        ax.text(0.05, y_pos, err_label, color=color, transform=ax.transAxes,
                verticalalignment='top', fontsize=12, fontweight='bold')
    ax.axvline(truths[idx], color='black', linestyle='--', linewidth=1.5)
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.set_xlabel(label, fontsize=16)
    ax.tick_params(axis='x', labelsize=18)
    ax.grid(alpha=0.2)
    ax.minorticks_on()


# --- Hide any remaining unused subplots ---
for j in range(len(labels) + 1, len(axes)):
    axes[j].axis('off')

# --- Global legend (top right) ---
handles = [plt.Line2D([], [], color=c, lw=2, label=lbl) for c, lbl in zip(colors, event_labels)]
fig.legend(handles=handles, loc='upper right', fontsize=20, title='Events', title_fontsize=22, frameon=False, bbox_to_anchor=(0.85, 0.93))

plt.subplots_adjust(left=0.08, right=0.88, top=0.93, bottom=0.08, wspace=0.35, hspace=0.50)
plt.savefig("/data/wiay/postgrads/shashwat/EMRI_data/INFERENCE_DATA/PLOTS/1D-posteriors/pop_MIX_A_B_combined_1D_posteriors.pdf", dpi=300, bbox_inches="tight")
plt.savefig("/data/wiay/postgrads/shashwat/EMRI_data/INFERENCE_DATA/PLOTS/1D-posteriors/pop_MIX_A_B_combined_1D_posteriors.png", dpi=300, bbox_inches="tight")
# plt.show()