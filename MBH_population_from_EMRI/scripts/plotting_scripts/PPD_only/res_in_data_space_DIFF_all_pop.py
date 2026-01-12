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

device = 'cpu'

# --- Matplotlib and seaborn config ---
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'],'size':15})
matplotlib.rc('text', usetex=True)
pal = sns.color_palette(palette="colorblind").as_hex()

# --- Truths and distributions ---

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

def get_dist(pop):
    DIST = {}

    if pop == "A":
        DIST = {
        "log10_M": FixedLimitSchechterFunction([6, 10], device=device),
        "log10_mu": FixedLimitsTruncatedSkewNormal([1, 2], device=device),
        "a" : FixedLimitTruncatedBetaDistribution([0.1, 0.7], device=device),
        "e0" : FixedLimitTruncatedBetaDistribution([0.1, 0.7], device=device),
        }

    if pop =="B":
    # this population is derived from https://doi.org/10.1093/mnras/stad1397
        DIST = {
        "log10_M": FixedLimitsPowerLaw([6, 10], device=device),
        "log10_mu": FixedLimitsPowerLaw([1, 2], device=device),
        "a" : FixedLimitsTruncatedGaussian([0.1, 0.7], device=device),
        "e0" : UniformDistribution([0.1, 0.7], device=device),
        }

    if pop=="A_B":
        DIST = {
            "log10_M": [FixedLimitSchechterFunction([6, 10], device=device), FixedLimitsPowerLaw([6, 10], device=device)],
            "log10_mu": [FixedLimitsTruncatedSkewNormal([1, 2], device=device), FixedLimitsPowerLaw([1, 2], device=device)],
            "a" : [FixedLimitTruncatedBetaDistribution([0.1, 0.7], device=device), FixedLimitsTruncatedGaussian([0.1, 0.7], device=device)],
            "e0" : [FixedLimitTruncatedBetaDistribution([0.1, 0.7], device=device), UniformDistribution([0.1, 0.7], device=device)],
        }

    return DIST

def get_true_x(pop):

    true_x = {}
    if pop == "A":
        true_x = {
            "log10_M": {"xc": 6.8},
            "log10_mu": {"mu": 1.5, "sigma": 0.5, "alpha": -1},
            "a": {"alpha": 12, "beta": 8},
            "e0": {"alpha": 8, "beta": 3},
            }

    if pop == "B":
        true_x = {
                "log10_M": {"lam": -1.5},
                "log10_mu": {"lam": -1.1},
                "a": {"mu": 0.6, "sigma": 0.01},
                "e0" : {}
                }

    if pop == "A_B":
        true_x = {
                "log10_M": {"xc_A": 7.5, "lam_B": -1.5},
                "log10_mu": {"mu_A": 1.5, "sigma_A": 0.5, "alpha_A": -0.5, "lam_B": -3.0},
                "a": {"alpha_A": 12, "beta_A": 8, "mu_B": 0.3, "sigma_B": 0.01},
                "e0": {"alpha_A": 8, "beta_A": 3,"UNIFORM_B" : {}},
                }
    return true_x

def make_true_x(pop, samples):
    if pop == "A":
        return {
            "log10_M": {"xc": samples[0]},
            "log10_mu": {"mu": 1.5, "sigma": 0.5, "alpha": samples[1]},
            "a": {"alpha": samples[2], "beta": samples[3]},
            "e0": {"alpha": samples[4], "beta": samples[5]}
        }

    elif pop == "B":
        return {
            "log10_M": {"lam": samples[0]},
            "log10_mu": {"lam": samples[1]},
            "a": {"mu": samples[2], "sigma": samples[3]},
            "e0": {},
        }

    elif pop in ["A_B"]:  # treat A_B as the mixed population
        return {
                "log10_M": {"xc_A": samples[0], "lam_B": samples[6]},
                "log10_mu": {"mu_A": 1.5, "sigma_A": 0.5, "alpha_A": samples[1], "lam_B": samples[7]},
                "a": {"alpha_A": samples[2], "beta_A": samples[3], "mu_B": samples[8], "sigma_B": samples[9]},
                "e0": {"alpha_A": samples[4], "beta_A": samples[5], "UNIFORM_B": {}}
            }

def get_weight(pop):
    if pop == "A" or pop=="B":
        return 1.0
    else :
        return 0.7

keys = ['log10_M', 'log10_mu', 'a', 'e0']
labels = ['$\\log_{10}(M/M_\odot)$', '$\\log_{10}(\\mu/M_\odot)$', '$a$', '$e_0$']

# --- Keys expected inside your HDF5 'posterior_samples' group ---

keys_A = ['xc_M', 'alpha_mu', 'alpha_a', 'beta_a', 'alpha_e0', 'beta_e0']
keys_B = ['lambda_M', 'lambda_mu', 'mu_a', 'sigma_a']
keys_A_B = ['A_xc_M', 'A_alpha_mu', 'A_alpha_a', 'A_beta_a', 'A_alpha_e0', 'A_beta_e0', 'B_lambda_M', 'B_lambda_mu', 'B_mu_a', 'B_sigma_a', 'weight']

event_counts = ['1E2', '1E3', '1E4']
event_labels = ['$10^2$', '$10^3$', '$10^4$']
colors = [pal[4], pal[2], pal[0]]

SIZE = 100000

bins = 50

pop_to_keys = {
    'A': keys_A,
    'B': keys_B,
    'A_B': keys_A_B,   # your script used 'A_B' as the mixed population label
    'MIX': keys_A_B
}

# --- Load posterior samples ---
def load_samples(base_path, pop_key='A'):
    """
    Open base_path/inference/result.hdf5, look for group 'posterior_samples' and extract arrays
    in the order defined by pop_to_keys[pop_key]. Return (n_samples, n_params) array.
    """
    h5file = os.path.join(base_path, "inference", "result.hdf5")

    if not os.path.exists(h5file):
        raise FileNotFoundError(f"No file found: {h5file}")

    keys = pop_to_keys.get(pop_key)
    if keys is None:
        raise ValueError(f"Unknown population key '{pop_key}'. Valid: {list(pop_to_keys.keys())}")
    
    with h5py.File(h5file, "r") as hf:
        print("In", pop_key)
        if "posterior_samples" not in hf:
            raise KeyError(f"'posterior_samples' group not found in {h5file}")

        ps = hf["posterior_samples"]
        arrays = []
        for k in keys:
            arr = np.array(ps[k])
            arrays.append(arr)
        # Each arrays[i] is shape (n_samples,)
        stacked = np.vstack(arrays).T  # shape (n_samples, n_params)
        return stacked

# --- Helper to compute histograms ---
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

# Outer loop: injected population
# for POP in ['A', 'B', 'A_B']:
for POP in ['A_B']:
    print("POP", POP)
    base_dir = f"/data/wiay/postgrads/shashwat/EMRI_data/INFERENCE_DATA/_pop_{POP}_NO_SF_DIFF_POP"

    # Inner loop: which (different) population was used to create the data
    for DIFF_POP in ['A', 'B', 'A_B']:

        # Plot the true population 
        fig, ax = matplotlib.pyplot.subplots(2, 2, figsize=(8, 10))
        popdist_true = PopulationDistribution(distributions=get_dist(POP), data=None)
        catalogue_true = popdist_true.draw_samples(get_true_x(POP), weight=get_weight(POP), size=SIZE)

        # True distributions as histograms
        log10_M_centers, y_true_log10_M = compute_hist(catalogue_true['log10_M'], bins)
        log10_mu_centers, y_true_log10_mu = compute_hist(catalogue_true['log10_mu'], bins)
        a_centers, y_true_a = compute_hist(catalogue_true['a'], bins)
        e0_centers, y_true_e0 = compute_hist(catalogue_true['e0'], bins)
        
        print("DIFF_POP", DIFF_POP)
        # load sample arrays for each event count; if any file missing, skip that combination
        samples_all = []
        successful_counts = []
        for ev in event_counts:
            path = os.path.join(base_dir, f"{ev}_events", DIFF_POP)
            h5_path = os.path.join(f"{path}/inference/", "result.hdf5")
            if not os.path.exists(h5_path):
                print(f"Skipping: no file found for {h5_path}")
                continue
            else :
                print(f"Found file for {h5_path}")
            try:
                samples = load_samples(path, pop_key=DIFF_POP)
                samples_all.append(samples)
                successful_counts.append(ev)
                print(f"Success: POP={POP}, DIFF_POP={DIFF_POP}, events={ev}")
            except Exception as e:
                print(f"Warning: failed to load samples from {h5_path} — {e}")
                continue
            
        # --- If no samples were successfully loaded, skip this combination ---
        if len(samples_all) == 0:
            print(f"Skipping: no valid samples for POP={POP}, DIFF_POP={DIFF_POP}")
            continue


        # --- Loop over samples ---
        for total in range(len(samples_all)):
            
            LEN = len(samples_all[total][:,0])
            print(LEN)

            hist_log10_M = []
            hist_log10_mu = []
            hist_a = []
            hist_e0 = []

            for i in range(LEN):
                # if i % 1000 == 0:
                #     print(f"making estimated {i}")
                true_x = make_true_x(DIFF_POP, samples_all[total][i])
                popdist_estimated = PopulationDistribution(distributions=get_dist(DIFF_POP), data=None)
                catalogue_estimated = popdist_estimated.draw_samples(true_x, weight=get_weight(DIFF_POP), size=SIZE)

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
            ax[1,0].plot(a_centers, percentiles_a[1], color=colors[total], label=event_labels[total])
            ax[1,0].plot(a_centers, mean_a, color=colors[total], linestyle='--')
            ax[1,0].fill_between(a_centers, percentiles_a[0], percentiles_a[1], alpha=0.3, color=colors[total])
            ax[1,0].legend(frameon=False, title='Injections', loc='upper right', title_fontsize=18, fontsize=16)

            ax[1,1].plot(e0_centers, percentiles_e0[0], color=colors[total])
            ax[1,1].plot(e0_centers, percentiles_e0[1], color=colors[total])
            ax[1,1].plot(e0_centers, mean_e0, color=colors[total], linestyle='--')
            ax[1,1].fill_between(e0_centers, percentiles_e0[0], percentiles_e0[1], alpha=0.3, color=colors[total])

            for pos, key in enumerate(keys):
                row, col = divmod(pos, 2)
                ax[row, col].set_ylabel(f'd$N$/d{labels[pos]}')
                ax[row, col].set_xlabel(labels[pos])
                ax[row, col].set_yticks([])


        # --- Plot true distributions ---
        ax[0,0].hist(catalogue_true['log10_M'], color='black', lw=2, label='True', histtype='step', bins=bins)
        ax[0,0].legend(frameon=False, fontsize=18)
        ax[0,1].hist(catalogue_true['log10_mu'], color='black', lw=2, histtype='step', bins=bins)
        ax[1,0].hist(catalogue_true['a'], color='black', lw=2, histtype='step', bins=bins)
        ax[1,1].hist(catalogue_true['e0'], color='black', lw=2, histtype='step', bins=bins)

        delta = 1e-6
        ax[0,0].set_xlim(6-delta, 10+delta)
        ax[0,1].set_xlim(1-delta, 2+delta)
        ax[1,0].set_xlim(0.1, 0.7)
        ax[1,1].set_xlim(0.1, 0.7)

        plt.tight_layout()
        plt.savefig(f'/data/wiay/postgrads/shashwat/EMRI_data/INFERENCE_DATA/PLOTS/PPD/PPD_{POP}_DIFF_POP_{DIFF_POP}_hist.png', dpi=300)
        plt.savefig(f'/data/wiay/postgrads/shashwat/EMRI_data/INFERENCE_DATA/PLOTS/PPD/PPD_{POP}_DIFF_POP_{DIFF_POP}_hist.pdf', dpi=300)