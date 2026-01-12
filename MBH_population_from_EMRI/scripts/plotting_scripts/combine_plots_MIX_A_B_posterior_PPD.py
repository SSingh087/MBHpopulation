import numpy as np
import seaborn as sns
import matplotlib
import h5py
import matplotlib.pyplot as plt
import torch
import os, sys
import re
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.abspath('../poplar'))
from poplar.distributions import *

# --- Matplotlib and seaborn config ---
matplotlib.rc('font', family='serif', serif=['Computer Modern'], size=15)
matplotlib.rc('text', usetex=True)
pal = sns.color_palette(palette="colorblind").as_hex()

INJ = ['$10^2$', '$10^3$', '$10^4$']

# --- Truths for vertical lines ---
DIST = {
"log10_M": [FixedLimitSchechterFunction([6, 10]), FixedLimitsPowerLaw([6, 10])],
"log10_mu": [FixedLimitsTruncatedSkewNormal([1, 2]), FixedLimitsPowerLaw([1, 2])],
"a" : [FixedLimitTruncatedBetaDistribution([0.1, 0.7]), FixedLimitsTruncatedGaussian([0.1, 0.7])],
"e0" : [FixedLimitTruncatedBetaDistribution([0.1, 0.7]), UniformDistribution([0.1, 0.7])],
}

true_x_MIX = {
    "log10_M": {"xc_A": 7.5, "lam_B": -1.5},
    "log10_mu": {"mu_A": 1.5, "sigma_A": 0.5, "alpha_A": -0.5, "lam_B": -3.0},
    "a": {"alpha_A": 12, "beta_A": 8, "mu_B": 0.3, "sigma_B": 0.01},
    "e0": {"alpha_A": 8, "beta_A": 3,"UNIFORM_B" : {}},
}

# Ground truths for vertical lines
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

event_labels = ['$10^2$', '$10^3$', '$10^4$']
colors = [pal[4], pal[2], pal[0]]

# --- Load posterior samples ---
def load_samples(path):
    with h5py.File(f'{path}/result.hdf5', 'r') as hf:
        # breakpoint()
        samples = [
            np.array(hf['posterior_samples'][key])
            for key in ['A_xc_M', 'A_alpha_mu', 'A_alpha_a', 'A_beta_a', 'A_alpha_e0', 'A_beta_e0', 'B_lambda_M', 'B_lambda_mu', 'B_mu_a', 'B_sigma_a','weight']
        ]
    return np.vstack(samples).T

def make_true_x(samples):
    return {
            "log10_M": {"xc_A": samples[0], "lam_B": samples[6]},
            
            "log10_mu": {"mu_A": 1.5, "sigma_A": 0.5, "alpha_A": samples[1], "lam_B": samples[7]},
            
            "a": {"alpha_A": samples[2], "beta_A": samples[3], "mu_B": samples[8], "sigma_B": samples[9]},

            "e0": {"alpha_A": samples[4], "beta_A": samples[5], "UNIFORM_B": {}}
        }

# --- Helper to compute histograms ---
bins = 50
def compute_hist(data, bins):
    data = np.array(data)
    data = data[~np.isnan(data)]        # remove NaNs
    data = data[np.isfinite(data)]      # remove inf/-inf
    if len(data) == 0:
        # Avoid empty data
        return np.linspace(0, 1, bins), np.zeros(bins)

    hist, edges = np.histogram(data, bins=bins, density=False)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return centers, hist

class PopulationDistribution:
    def __init__(self, distributions, data) -> None:
        self.distributions = distributions
        self.data = data

    def draw_samples(self, x, weight=1.0, size=500):
        out = {}
        self.weight = weight
        
        for key in self.distributions.keys():
            hyperparams = list(x[key].items())
            cleaned_hyperparams_A = {re.sub(r'_A$', '', k): v for k, v in hyperparams if k.endswith('_A')}
            cleaned_hyperparams_B = {re.sub(r'_B$', '', k): v for k, v in hyperparams if k.endswith('_B')}
            
            if not cleaned_hyperparams_A and not cleaned_hyperparams_B:  
                out[key] = self.distributions[key].draw_samples(**x[key], size=int(size))
            else:
                choices = torch.bernoulli(torch.full((size,), self.weight)).bool()
                
                if cleaned_hyperparams_A == {'UNIFORM': {}}:
                    samples_A = self.distributions[key][0].draw_samples(size=size)
                    samples_B = self.distributions[key][1].draw_samples(**cleaned_hyperparams_B, size=size)
                elif cleaned_hyperparams_B == {'UNIFORM': {}}:
                    samples_A = self.distributions[key][0].draw_samples(**cleaned_hyperparams_A, size=size)
                    samples_B = self.distributions[key][1].draw_samples(size=size)
                else:
                    samples_A = self.distributions[key][0].draw_samples(**cleaned_hyperparams_A, size=size)
                    samples_B = self.distributions[key][1].draw_samples(**cleaned_hyperparams_B, size=size)

                out[key] = torch.where(choices, samples_A, samples_B)

        return out

# --- Paths and load samples ---
base_path = '/data/wiay/postgrads/shashwat/EMRI_data/INFERENCE_DATA/_pop_MIX_A_B_NO_SF'
event_dirs = ['1E2_events', '1E3_events', '1E4_events']
samples_all = [load_samples(f'{base_path}/{event}/inference') for event in event_dirs]


SIZE = 100000

# generate truth catalogue
popdist = PopulationDistribution(distributions=DIST, data=None)
catalogue_true = popdist.draw_samples(true_x_MIX, weight=0.7, size=SIZE)

# compute true histograms
log10_M_centers, y_true_log10_M = compute_hist(catalogue_true['log10_M'], bins)
log10_mu_centers, y_true_log10_mu = compute_hist(catalogue_true['log10_mu'], bins)
a_centers, y_true_a = compute_hist(catalogue_true['a'], bins)
e0_centers, y_true_e0 = compute_hist(catalogue_true['e0'], bins)

fig = plt.figure(figsize=(17, 13))  # wide figure

# ------------------------------------------------
# LEFT BLOCK: 2x2 PPD
# ------------------------------------------------
gs_ppd = gridspec.GridSpec(
    2, 2,
    figure=fig,
    left=0.05, right=0.48,
    bottom=0.32, top=0.95,   # leave bottom space for 1×5
    wspace=0.25, hspace=0.25
)
ppd_axes = np.array([[fig.add_subplot(gs_ppd[r, c]) for c in range(2)] for r in range(2)])
ax = ppd_axes  # alias used by the remainder of your code

# ------------------------------------------------
# RIGHT BLOCK: 3×2 (6 posteriors)
# ------------------------------------------------
gs_post_top = gridspec.GridSpec(
    3, 2,
    figure=fig,
    left=0.52, right=0.97,
    bottom=0.32, top=0.95,
    wspace=0.35, hspace=0.45
)
post_top_axes = [fig.add_subplot(gs_post_top[i, j]) for i in range(3) for j in range(2)]

# ------------------------------------------------
# BOTTOM BLOCK: 1×5 (5 posteriors)
# ------------------------------------------------
gs_post_bottom = gridspec.GridSpec(
    1, 5,
    figure=fig,
    left=0.05, right=0.97,
    bottom=0.05, top=0.25,    # full width
    wspace=0.30
)
post_bottom_axes = [fig.add_subplot(gs_post_bottom[0, k]) for k in range(5)]

# ------------------------------------------------
# COMBINE posterior axes into one list of 11
# ------------------------------------------------
post_axes = post_top_axes + post_bottom_axes   # exactly 11 axes

# ================================
# PPD PLOTS (LEFT)
# ================================
keys = ['log10_M', 'log10_mu', 'a', 'e0']
labels_ppd = [
    '$\\log_{10}(M/M_\\odot)$',
    '$\\log_{10}(\\mu/M_\\odot)$',
    '$a$',
    '$e_0$'
]

for total in range(len(samples_all)):

    LEN = len(samples_all[total][:,0])

    hist_log10_M = []
    hist_log10_mu = []
    hist_a = []
    hist_e0 = []

    for i in range(LEN):
        # if i % 1000 == 0:
        #     print(f"making estimated {i}")
        true_x = make_true_x(samples_all[total][i])
        catalogue_estimated = popdist.draw_samples(true_x, weight=0.7, size=SIZE)

        centers_M, h_M = compute_hist(catalogue_estimated['log10_M'], bins)
        centers_mu, h_mu = compute_hist(catalogue_estimated['log10_mu'], bins)
        centers_a, h_a = compute_hist(catalogue_estimated['a'], bins)
        centers_e0, h_e0 = compute_hist(catalogue_estimated['e0'], bins)

        hist_log10_M.append(h_M)
        hist_log10_mu.append(h_mu)
        hist_a.append(h_a)
        hist_e0.append(h_e0)

    hist_log10_M = np.array(hist_log10_M).T
    hist_log10_mu = np.array(hist_log10_mu).T
    hist_a = np.array(hist_a).T
    hist_e0 = np.array(hist_e0).T

    percentiles_M = np.percentile(hist_log10_M, [16, 84], axis=1)
    percentiles_mu = np.percentile(hist_log10_mu, [16, 84], axis=1)
    percentiles_a = np.percentile(hist_a, [16, 84], axis=1)
    percentiles_e0 = np.percentile(hist_e0, [16, 84], axis=1)

    mean_M = np.mean(hist_log10_M, axis=1)
    mean_mu = np.mean(hist_log10_mu, axis=1)
    mean_a = np.mean(hist_a, axis=1)
    mean_e0 = np.mean(hist_e0, axis=1)

    # plot each PPD panel
    ax[0,0].plot(centers_M, percentiles_M[0], color=colors[total])
    ax[0,0].plot(centers_M, percentiles_M[1], color=colors[total])
    ax[0,0].plot(centers_M, mean_M, linestyle='--', color=colors[total])
    ax[0,0].fill_between(centers_M, percentiles_M[0], percentiles_M[1], alpha=0.3, color=colors[total])

    ax[0,1].plot(centers_mu, percentiles_mu[0], color=colors[total])
    ax[0,1].plot(centers_mu, percentiles_mu[1], color=colors[total])
    ax[0,1].plot(centers_mu, mean_mu, linestyle='--', color=colors[total])
    ax[0,1].fill_between(centers_mu, percentiles_mu[0], percentiles_mu[1], alpha=0.3, color=colors[total])

    ax[1,0].plot(a_centers, percentiles_a[0], color=colors[total], label=INJ[total])
    ax[1,0].plot(a_centers, percentiles_a[1], color=colors[total])
    ax[1,0].plot(a_centers, mean_a, linestyle='--', color=colors[total])
    ax[1,0].fill_between(a_centers, percentiles_a[0], percentiles_a[1], alpha=0.3, color=colors[total])

    ax[1,1].plot(e0_centers, percentiles_e0[0], color=colors[total])
    ax[1,1].plot(e0_centers, percentiles_e0[1], color=colors[total])
    ax[1,1].plot(e0_centers, mean_e0, linestyle='--', color=colors[total])
    ax[1,1].fill_between(e0_centers, percentiles_e0[0], percentiles_e0[1], alpha=0.3, color=colors[total])


# true distributions (left panel)
ax[0,0].hist(catalogue_true['log10_M'], color='black', lw=2, label='True', histtype='step', bins=bins)
ax[0,0].legend(frameon=False, fontsize=18)
ax[0,1].hist(catalogue_true['log10_mu'], color='black', lw=2, histtype='step', bins=bins)
ax[1,0].hist(catalogue_true['a'],         color='black', lw=2, histtype='step', bins=bins)
ax[1,1].hist(catalogue_true['e0'],        color='black', lw=2, histtype='step', bins=bins)

delta = 1e-6
ax[0,0].set_xlim(6-delta, 10+delta)
ax[0,1].set_xlim(1-delta, 2+delta)
ax[1,0].set_xlim(0.1, 0.7)
ax[1,1].set_xlim(0.1, 0.7)

for pos, lab in enumerate(labels_ppd):
    r, c = divmod(pos, 2)
    ax[r, c].set_xlabel(lab, fontsize=18)
    ax[r, c].set_ylabel(f'd$N$/d{lab}', fontsize=18)
    ax[r, c].tick_params(axis='both', labelsize=16)
    ax[r, c].set_yticks([])

ax[1,0].legend(frameon=False, title='Injections', title_fontsize=18, fontsize=16)

# ================================
# Posterior KDE panel (RIGHT)
# ================================

labels_post = [
    "$x_c^{\\mathrm{A}}$", "$\\gamma^{\\mathrm{A}}_{\\mu}$", "$\\alpha^{\\mathrm{A}}_a$", "$\\beta^{\\mathrm{A}}_a$",
    "$\\alpha^{\\mathrm{A}}_{e0}$", "$\\beta^{\\mathrm{A}}_{e0}$", "$\\lambda^{\\mathrm{B}}_M$", "$\\lambda^{\\mathrm{B}}_{\\mu}$",
    "$\\mu^{\\mathrm{B}}_a$", "$\\sigma^{\\mathrm{B}}_a$", "$w$"
]
y_positions = [0.95, 0.75, 0.55]
x_positions = [0.05, 0.90, 0.90, 0.05, 0.05, 0.90, 0.05, 0.05, 0.05, 0.90, 0.05]
ha_positions = ['left','right','right','left','left','right', 'left','left','left', 'right', 'left']

for idx, label in enumerate(labels_post):
    axp = post_axes[idx]

    for samples, color, y_pos in zip(samples_all, colors, y_positions):
        sns.kdeplot(samples[:, idx], fill=True, color=color, linewidth=1.8, ax=axp)

        q16, q50, q84 = np.percentile(samples[:, idx], [16,50,84])
        err = "${:.2f}_{{- {:.4f}}}^{{+ {:.4f}}}$".format(q50, q50 - q16, q84 - q50)
        axp.text(
            x_positions[idx], y_pos, err,
            transform=axp.transAxes,
            verticalalignment='top',
            horizontalalignment=ha_positions[idx],
            color=color,
            fontsize=14
        )

    axp.axvline(truths[idx], color='black', linestyle='--', linewidth=1.5)
    axp.set_yticks([])
    axp.set_ylabel(f'$p$({label}$|\\textit{{\\textbf{{d}}}}$)')
    axp.set_xlabel(label, fontsize=16)
    axp.tick_params(axis='x', labelsize=16)
    # axp.grid(alpha=0.2)
    axp.minorticks_on()

plt.tight_layout()

plt.savefig("/data/wiay/postgrads/shashwat/EMRI_data/INFERENCE_DATA/PLOTS/combine_plot_MIX_A_B_posterior1D_PPD.png", dpi=300, bbox_inches="tight")
plt.savefig("/data/wiay/postgrads/shashwat/EMRI_data/INFERENCE_DATA/PLOTS/combine_plot_MIX_A_B_posterior1D_PPD.pdf", dpi=300, bbox_inches="tight")