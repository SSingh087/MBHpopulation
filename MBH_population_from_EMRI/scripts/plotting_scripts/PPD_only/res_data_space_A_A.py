import numpy as np
import seaborn as sns
import matplotlib
import h5py
import matplotlib.pyplot as plt
import torch
import os, sys
import re

sys.path.insert(0, os.path.abspath('../poplar'))
from poplar.distributions import *

# --- Matplotlib and seaborn config ---
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'],'size':15})
matplotlib.rc('text', usetex=True)
pal = sns.color_palette(palette="colorblind").as_hex()

# --- Truths for vertical lines ---

DIST = {
    "log10_M": [FixedLimitSchechterFunction([6, 10]), FixedLimitSchechterFunction([6, 10])],
    "log10_mu": [FixedLimitsTruncatedSkewNormal([1, 2]), FixedLimitsTruncatedSkewNormal([1, 2])],
    "a" : [FixedLimitTruncatedBetaDistribution([0.1, 0.7]), FixedLimitTruncatedBetaDistribution([0.1, 0.7])],
    "e0" : [FixedLimitTruncatedBetaDistribution([0.1, 0.7]), FixedLimitTruncatedBetaDistribution([0.1, 0.7])],
}

true_x_MIX = {
    "log10_M": {"xc_A": 7.0, "xc_B": 8.0},
    "log10_mu": {"mu_A": 1.5, "sigma_A": 0.5, "alpha_A": -0.5, "mu_B": 1.5, "sigma_B": 0.5, "alpha_B": 0.5},
    "a": {"alpha_A": 12, "beta_A": 8, "alpha_B": 8, "beta_B": 12},
    "e0": {"alpha_A": 8, "beta_A": 3, "alpha_B": 3, "beta_B": 8},
    }

event_labels = ['$10^2$', '$10^3$', '$10^4$']
colors = [pal[4], pal[2], pal[0]]

# --- Load posterior samples ---
def load_samples(path):
    with h5py.File(f'{path}/result.hdf5', 'r') as hf:
        # breakpoint()
        samples = [
            np.array(hf['posterior_samples'][key])
            for key in ['A_xc_M', 'A_alpha_mu', 'A_alpha_a', 'A_beta_a', 'A_alpha_e0', 'A_beta_e0', 'B_xc_M', 'B_alpha_mu', 'B_alpha_a', 'B_beta_a', 'B_alpha_e0', 'B_beta_e0', 'weight']
        ]
    return np.vstack(samples).T

def make_true_x(samples):
    return {
            "log10_M": {"xc_A": samples[0], "xc_B": samples[6]},
            "log10_mu": {"mu_A": 1.5, "sigma_A": 0.5, "alpha_A": samples[1], "mu_B": 1.5, "sigma_B": 0.5, "alpha_B": samples[7]},
            "a": {"alpha_A": samples[2], "beta_A": samples[3], "alpha_B": samples[8], "beta_B": samples[9]},
            "e0": {"alpha_A": samples[4], "beta_A": samples[5], "alpha_B": samples[10], "beta_B": samples[11],}
        }

class PopulationDistribution:
    def __init__(self, distributions, data) -> None:
        self.distributions = distributions
        self.data = data

    def draw_samples(self, x, weight=1.0, size=500):
        out = {}

        self.weight = weight
        
        for key in self.distributions.keys():

            hyperparams = list(x[key].items())
            cleaned_hyperparams_A = {re.sub(r'_A$', '', k): v for k, v in hyperparams if k.endswith('_A')} # k.replace
            cleaned_hyperparams_B = {re.sub(r'_B$', '', k): v for k, v in hyperparams if k.endswith('_B')}
            
            if not cleaned_hyperparams_A and not cleaned_hyperparams_B:  
                # If no '_A' or '_B' suffixes exist, use params directly
                out[key] = self.distributions[key].draw_samples(**x[key], size=int(size))
            
            else:

                choices = torch.bernoulli(torch.full((size,), self.weight)).bool()
                
                if cleaned_hyperparams_A == {'UNIFORM': {}}:
                    samples_A = self.distributions[key][0].draw_samples(size=size)
                    samples_B = self.distributions[key][1].draw_samples(**cleaned_hyperparams_B, size=size)

                elif cleaned_hyperparams_B == {'UNIFORM': {}}:
                    samples_A = self.distributions[key][0].draw_samples(**cleaned_hyperparams_A, size=size)
                    samples_B = self.distributions[key][1].draw_samples(size=size)

                else :
                    samples_A = self.distributions[key][0].draw_samples(**cleaned_hyperparams_A, size=size)
                    samples_B = self.distributions[key][1].draw_samples(**cleaned_hyperparams_B, size=size)

                # Select from A or B based on choices
                out[key] = torch.where(choices, samples_A, samples_B)

        return out


base_path = '/data/wiay/postgrads/shashwat/EMRI_data/INFERENCE_DATA/_pop_MIX_A_A_NO_SF'
event_dirs = ['1E2_events', '1E3_events']#, '1E4_events']
samples_all = [load_samples(f'{base_path}/{event}/inference') for event in event_dirs]


SIZE = 1000

fig, ax = matplotlib.pyplot.subplots(2, 2, figsize=(8, 8))

keys = ['log10_M', 'log10_mu', 'a', 'e0']
labels = ['$\\log_{10}(M/M_\odot)$', '$\\log_{10}(\\mu/M_\odot)$', '$a$', '$e_0$']
INJ = ['$10^2$', '$10^3$', '$10^4$']

popdist = PopulationDistribution(distributions=DIST, data=None)
catalogue_true = popdist.draw_samples(true_x_MIX, weight=0.7, size=SIZE)

# --- Helper to compute histograms ---
bins = 100  
def compute_hist(data, bins):
    data = np.array(data)
    data = data[~np.isnan(data)]        # remove NaNs
    data = data[np.isfinite(data)]      # remove inf/-inf
    if len(data) == 0:
        # Avoid empty data
        return np.linspace(0, 1, bins), np.zeros(bins)

    hist, edges = np.histogram(data, bins=bins, density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return centers, hist

# True distributions as histograms
log10_M_centers, y_true_log10_M = compute_hist(catalogue_true['log10_M'], bins)
log10_mu_centers, y_true_log10_mu = compute_hist(catalogue_true['log10_mu'], bins)
a_centers, y_true_a = compute_hist(catalogue_true['a'], bins)
e0_centers, y_true_e0 = compute_hist(catalogue_true['e0'], bins)

# --- Loop over samples ---
for total in range(len(samples_all)):
    
    LEN = len(samples_all[total][:,0])
    print(LEN)

    hist_log10_M = []
    hist_log10_mu = []
    hist_a = []
    hist_e0 = []

    for i in range(LEN):
        if i % 100 == 0:
            print(f"making estimated {i}")
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

    # Convert to arrays and compute mean and percentiles
    hist_log10_M = np.array(hist_log10_M)
    hist_log10_mu = np.array(hist_log10_mu)
    hist_a = np.array(hist_a)
    hist_e0 = np.array(hist_e0)

    hist_log10_M_T = hist_log10_M.T
    hist_log10_mu_T = hist_log10_mu.T
    hist_a_T = hist_a.T
    hist_e0_T = hist_e0.T

    percentiles_log10_M = np.percentile(hist_log10_M_T, [16, 84], axis=1)
    mean_log10_M = np.mean(hist_log10_M_T, axis=1)

    percentiles_log10_mu = np.percentile(hist_log10_mu_T, [16, 84], axis=1)
    mean_log10_mu = np.mean(hist_log10_mu_T, axis=1)

    percentiles_a = np.percentile(hist_a_T, [16, 84], axis=1)
    mean_a = np.mean(hist_a_T, axis=1)

    percentiles_e0 = np.percentile(hist_e0_T, [16, 84], axis=1)
    mean_e0 = np.mean(hist_e0_T, axis=1)

    # --- Plotting ---
    ax[0,0].plot(centers_M, percentiles_log10_M[0], color=colors[total])
    ax[0,0].plot(centers_M, percentiles_log10_M[1], color=colors[total])
    ax[0,0].plot(centers_M, mean_log10_M, color=colors[total], linestyle='--')
    ax[0,0].fill_between(centers_M, percentiles_log10_M[0], percentiles_log10_M[1], alpha=0.3, color=colors[total])

    ax[0,1].plot(centers_mu, percentiles_log10_mu[0], color=colors[total])
    ax[0,1].plot(centers_mu, percentiles_log10_mu[1], color=colors[total])
    ax[0,1].plot(centers_mu, mean_log10_mu, color=colors[total], linestyle='--')
    ax[0,1].fill_between(centers_mu, percentiles_log10_mu[0], percentiles_log10_mu[1], alpha=0.3, color=colors[total])

    ax[1,0].plot(a_centers, percentiles_a[0], color=colors[total])
    ax[1,0].plot(a_centers, percentiles_a[1], color=colors[total])
    ax[1,0].plot(a_centers, mean_a, color=colors[total], linestyle='--')
    ax[1,0].fill_between(a_centers, percentiles_a[0], percentiles_a[1], alpha=0.3, color=colors[total], label=INJ[total])
    ax[1,0].legend(frameon=False, title='Events')

    ax[1,1].plot(e0_centers, percentiles_e0[0], color=colors[total])
    ax[1,1].plot(e0_centers, percentiles_e0[1], color=colors[total])
    ax[1,1].plot(e0_centers, mean_e0, color=colors[total], linestyle='--')
    ax[1,1].fill_between(e0_centers, percentiles_e0[0], percentiles_e0[1], alpha=0.3, color=colors[total])

    for pos, key in enumerate(keys):
        row, col = divmod(pos, 2)
        ax[row, col].set_ylabel(f'dN/d{labels[pos]}')
        ax[row, col].set_xlabel(labels[pos])

# --- Plot true distributions ---
ax[0,0].plot(log10_M_centers, y_true_log10_M, color='black', lw=2, label='True')
ax[0,0].legend(frameon=False)
ax[0,1].plot(log10_mu_centers, y_true_log10_mu, color='black', lw=2)
ax[1,0].plot(a_centers, y_true_a, color='black', lw=2)
ax[1,1].plot(e0_centers, y_true_e0, color='black', lw=2)

plt.tight_layout()
plt.savefig('PPD_MIX_A_A_hist.png', dpi=300)
plt.savefig('PPD_MIX_A_A_hist.pdf', dpi=300)