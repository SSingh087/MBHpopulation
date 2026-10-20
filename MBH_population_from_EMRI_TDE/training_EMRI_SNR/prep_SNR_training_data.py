#!/usr/bin/env python
"""Combine and rescale the EMRI parameters and SNRs into the SNR-network training set."""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import glob, h5py

parser = argparse.ArgumentParser(description="Generate data for training.")
parser.add_argument("--events", type=int, required=True, help="total events")

args = parser.parse_args()

events_for_dir = f"{args.events:.0E}".replace("+0", "").replace(".0", "")
output_dir = f'./data/EMRI/PRE_TRAIN_DATA/{events_for_dir}_events/'

# === remove the phases from the SNR data =====
# Rapid determination of LISA sensitivity to extreme mass ratio inspirals with machine learning
# arXiv:2212.06166v1 [astro-ph.HE] 12 Dec 2022

inj_files_T_lt_T_obs = h5py.File(f'{output_dir}/new_injection_params_T_lt_T_obs.h5', 'r')
inj_data_T_lt_T_obs = np.array(inj_files_T_lt_T_obs.get('params'))

inj_data_T_gt_T_obs = np.load(f'{output_dir}/injection_params_T_gt_T_obs.npy')

inj_data_COMBINED = np.vstack((inj_data_T_lt_T_obs, inj_data_T_gt_T_obs))

inj_data_phases_removed_COMBINED = np.delete(inj_data_COMBINED, [-2, -3, -4], axis=1)

inj_data_phases_dist_removed = np.delete(inj_data_phases_removed_COMBINED, 6, axis=1)

np.save(f'{output_dir}/injection_params_COMBINED.npy', inj_data_phases_removed_COMBINED)

np.save(f'{output_dir}/injection_params_dist_REMOVED_COMBINED.npy', inj_data_phases_dist_removed)

# COMBINE SNRS as well

def minmax_scale(x, min_val, max_val):
    """Scale x to [0, 1] given its bounds."""
    return (x - min_val) / (max_val - min_val)


SNRs_T_lt_T_obs = np.array(np.load(f'{output_dir}/snrs_T_lt_T_obs.npy'))
SNRs_T_gt_T_obs = np.array(np.load(f'{output_dir}/snrs_T_gt_T_obs.npy'))

SNRs_COMBINED = np.hstack((SNRs_T_lt_T_obs, SNRs_T_gt_T_obs))

distances = inj_data_COMBINED[:, 6]

SNRs_dist_scaled_COMBINED = SNRs_COMBINED / distances

SNRs_dist_log_scaled_COMBINED = np.log10(SNRs_COMBINED / distances)

SNRs_log_minmax_rescaled_COMBINED = minmax_scale(np.log10(SNRs_COMBINED), np.min(np.log10(SNRs_COMBINED)), np.max(np.log10(SNRs_COMBINED)))

SNRs_dist_log_minmax_rescaled_COMBINED = minmax_scale(SNRs_dist_log_scaled_COMBINED, np.min(SNRs_dist_log_scaled_COMBINED), np.max(SNRs_dist_log_scaled_COMBINED))


import matplotlib.pyplot as plt

plt.hist(SNRs_COMBINED, bins=500)
plt.savefig('SNRs_unrescaled.png')
plt.close()

plt.hist(np.log10(SNRs_COMBINED), bins=500)
plt.savefig('SNRs_log_scaled.png')
plt.close()

plt.hist(SNRs_dist_scaled_COMBINED, bins=500)
plt.savefig('SNRs_dist_scaled.png')
plt.close()

plt.hist(SNRs_dist_log_scaled_COMBINED, bins=500)
plt.savefig('SNRs_dist_log_scaled.png')
plt.close()

plt.hist(SNRs_dist_log_minmax_rescaled_COMBINED, bins=500)
plt.savefig('SNRs_dist_log_minmax_rescaled.png')

np.save(f'{output_dir}/snrs_COMBINED.npy', SNRs_COMBINED)
np.save(f'{output_dir}/snrs_log_scaled_COMBINED.npy', np.log10(SNRs_COMBINED))
np.save(f'{output_dir}/snrs_dist_scaled_COMBINED.npy', SNRs_dist_scaled_COMBINED)
np.save(f'{output_dir}/snrs_dist_log_scaled_COMBINED.npy', SNRs_dist_log_scaled_COMBINED)
np.save(f'{output_dir}/snrs_log_minmax_rescaled_COMBINED.npy', SNRs_log_minmax_rescaled_COMBINED)
np.save(f'{output_dir}/snrs_dist_log_minmax_rescaled_COMBINED.npy', SNRs_dist_log_minmax_rescaled_COMBINED)

# bounds needed to invert the network output back to SNR (see data_generation/calculate_SNR_EMRI.py)
with open(f'{output_dir}/snr_scaling.json', 'w') as f:
    json.dump({"y_min": float(np.min(SNRs_dist_log_scaled_COMBINED)),
               "y_max": float(np.max(SNRs_dist_log_scaled_COMBINED))}, f)
