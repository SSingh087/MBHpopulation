#!/usr/bin/env python

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

from numpy import load, concatenate, linspace
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Generate data for training.")
parser.add_argument("--events", type=int, required=True, help="total events")

args = parser.parse_args()

events_for_dir = f"{args.events:.0E}".replace("+0", "").replace(".0", "")
output_dir = f'/data/wiay/postgrads/shashwat/EMRI_data/PRE_TRAIN_DATA/{events_for_dir}_events'

data_NEW_INJ_PARAMS = load(f'{output_dir}/new_injection_params_COMBINED.npy')
SNRs = load(f'{output_dir}/snrs_COMBINED.npy')


p0 = data_NEW_INJ_PARAMS[:, 3] # This is for _T_MAX
e0 = data_NEW_INJ_PARAMS[:, 4] # This is for _T_MAX
Y0 = data_NEW_INJ_PARAMS[:, 5] # This is for _T_MAX

Phi_phi0 = data_NEW_INJ_PARAMS[:, 11] # This is for _T_MAX
Phi_theta0 = data_NEW_INJ_PARAMS[:, 12] # This is for _T_MAX
Phi_r0 = data_NEW_INJ_PARAMS[:, 13] # This is for _T_MAX

T = data_NEW_INJ_PARAMS[:, 14]

variables = {
    "p0": p0,
    "e0": e0,
    "Y0": Y0,
    "Phi_phi0": Phi_phi0,
    "Phi_theta0": Phi_theta0,
    "Phi_r0": Phi_r0
}

plt.figure(figsize=(12, 8))  # Set figure size

for i, (label, values) in enumerate(variables.items(), 1):
    plt.subplot(2, 3, i)  # Create a 2x3 grid of subplots
    scatter = plt.scatter(T[::10], values[::10], c=SNRs[::10], cmap='viridis')
    plt.colorbar(scatter, label="SNR")  # Add colorbar to each plot
    plt.xlabel("T")
    plt.ylabel(label)
    plt.title(f"T vs {label}")

plt.tight_layout()  # Adjust layout for better spacing
plt.savefig('traj_SNR_T_evol.png', dpi=200)

