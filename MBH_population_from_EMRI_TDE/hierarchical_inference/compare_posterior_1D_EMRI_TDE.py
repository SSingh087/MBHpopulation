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

# --- Ground truths ---
true_x_B = {
    "log10_M": {"lam": -1.5},
    "log10_mu": {"lam": -3.0},
    "a": {"mu": 0.6, "sigma": 0.01},
}

truths = [
    true_x_B['log10_M']['lam'],
    true_x_B['log10_mu']['lam'],
    true_x_B['a']['mu'],
    true_x_B['a']['sigma'],
]

labels = ["$\\Lambda_M$", "$\\Lambda_{\\mu}$", "$\\mu_a$", "$\\sigma_a$"]

# --- Load posterior samples ---
def load_samples(base_path):
    with h5py.File(f'{base_path}/result.hdf5', 'r') as hf:
        ps = hf["posterior_samples"]
        lambda_M = np.array(ps['lambda_M'])
        lambda_mu = np.array(ps['lambda_mu'])
        mu_a = np.array(ps['mu_a'])
        sigma_a = np.array(ps['sigma_a'])
    return np.vstack([lambda_M, lambda_mu, mu_a, sigma_a]).T

base_dir = f"/data/wiay/postgrads/shashwat/EMRI_TDE_data/inference_data/{args.GALAXIES}"

samples_all = [
    load_samples(f"{base_dir}/1E1_events/inference/EMRI"),
    load_samples(f"{base_dir}/1E1_events/inference/TDE"),
    load_samples(f"{base_dir}/1E1_events/inference/EMRI_TDE"),
]

colors = [pal[3], pal[2], pal[0]]

def smart_format(q50, q_low, q_high, alpha=0.3):
    # uncertainties
    err_minus = q50 - q_low
    err_plus  = q_high - q50

    # smallest error determines precision
    u = min(abs(err_minus), abs(err_plus))

    # avoid log10(0)
    if u == 0:
        d = 1
    else:
        d = int(round(-np.log10(u) + alpha))
        d = max(d, 0)   # never negative decimal places

    # special rule: uncertainties beginning with "1.xxxxx"
    if 1 <= u < 2:
        d = max(d, 1)

    # round everything to d decimals
    q50_r = round(q50, d)
    em_r  = round(err_minus, d)
    ep_r  = round(err_plus, d)

    # format string
    fmt = f"{{:.{d}f}}"
    return f"${fmt.format(q50_r)}_{{-{fmt.format(em_r)}}}^{{+{fmt.format(ep_r)}}}$"


# --- Combined 1D KDE plot grid (like MIX) ---
nrows, ncols = 2, 2
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
axes = axes.flatten()

# --- Fill all subplots with KDEs ---

labels_post = ["$\\lambda_M$", "$\\lambda_{\\mu}$", "$\\mu_a$", "$\\sigma_a$"]
y_positions = [0.95, 0.75, 0.55]
x_positions = [0.05, 0.05, 0.90, 0.90]
ha_positions = ['left','left','right','right']

for idx, label in enumerate(labels_post):
    ax = axes[idx]

    for samples, color, y_pos in zip(samples_all, colors, y_positions):
        sns.kdeplot(samples[:, idx], fill=True, color=color, linewidth=1.8, ax=ax, alpha=0.1)

        q5, q50, q95 = np.percentile(samples[:, idx], [5, 50, 95])
        err = smart_format(q50, q5, q95)

        ax.text(
            x_positions[idx], y_pos, err,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment=ha_positions[idx],
            color=color,
            fontsize=14
        )

    ax.axvline(truths[idx], color='black', linestyle='--', linewidth=1.5)
    ax.set_yticks([])
    ax.set_ylabel(f'$p$({label}$|\\textit{{\\textbf{{d}}}}$)')
    ax.set_xlabel(label, fontsize=16)
    ax.tick_params(axis='x', labelsize=16)
    # ax.grid(alpha=0.2)
    ax.minorticks_on()

plt.subplots_adjust(left=0.08, right=0.88, top=0.93, bottom=0.08, wspace=0.35, hspace=0.45)
# plt.savefig("/data/wiay/postgrads/shashwat/EMRI_data/INFERENCE_DATA/PLOTS/1D-posteriors/pop_B_combined_1D_posteriors.pdf", dpi=300, bbox_inches="tight")
plt.savefig("pop_B_combined_1D_posteriors_compare_EMRI_TDE.png", dpi=300, bbox_inches="tight")
