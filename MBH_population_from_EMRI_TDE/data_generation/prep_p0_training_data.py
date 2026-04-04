#!/usr/bin/env python

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

from numpy import load, save, vstack, concatenate

parser = argparse.ArgumentParser(description="Generate data for training.")
parser.add_argument("--events", type=int, required=True, help="total events")

args = parser.parse_args()

events_for_dir = f"{args.events:.0E}".replace("+0", "").replace(".0", "")
output_dir = f'/data/wiay/postgrads/shashwat/EMRI_data/PRE_TRAIN_DATA/{events_for_dir}_events/'

data_inj_params_COMBINED = load(f'{output_dir}/injection_params_COMBINED.npy')


def minmax_scale(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


columns_to_extract = [0, 1, 2, 4, 5, -1] # Indices for M, mu, a, e0, Y0, T_INJECT

params_for_p0_T_SIGNAL = data_inj_params_COMBINED[:, columns_to_extract]

save(f'{output_dir}/params_for_p0_ALL_COMBINED.npy', params_for_p0_T_SIGNAL)

p0 = data_inj_params_COMBINED[:, 3]

p0_minmax_reascaled = minmax_scale(p0, min(p0), max(p0))

save(f'{output_dir}/p0_ALL_COMBINED.npy', p0)
save(f'{output_dir}/p0_minmax_rescaled_ALL_COMBINED.npy', p0_minmax_reascaled)