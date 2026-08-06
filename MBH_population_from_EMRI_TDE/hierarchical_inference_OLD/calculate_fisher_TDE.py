#!/usr/bin/env python

import os, sys, argparse
sys.path.insert(0, os.path.abspath('../../poplar'))

from population import *
from poplar.selection import matched_filter_snr_from_optimal_snr
import torch, pickle, h5py, cupy
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description="Generate data for training.")
parser.add_argument("--events", type=int, required=True, help="total events")
parser.add_argument("--work_dir", type=str, required=True, help="work_dir")

args = parser.parse_args()

# trained_model_path = './data/EMRI/trained_models'
work_dir = args.work_dir 

if not os.path.exists(work_dir):
    os.makedirs(work_dir, exist_ok=True)


events = int(args.events)

N_samples_theta = 1000


if __name__ == "__main__":
    
    catalogue = popdist_B.draw_samples(x=true_x_B, size=events)

    kept_catalogue = {
        n: np.random.normal(
            loc=d[:, None] + np.random.normal(loc=0, scale=1e-1, size=(len(d), 1)),  
            scale=1e-1,
            size=(len(d), N_samples_theta)
        )  
        for n, d in catalogue.items() if n not in ['T']
    }

    popdist_B.data = kept_catalogue
    print("DATA SIZE IS (EVENTS, POSTERIOR_SAMPLES): ", kept_catalogue['log10_M'].shape)
    hf = h5py.File(f'{work_dir}/true_pop_data_TDE.h5', 'w')

    for key in popdist_B.data:
        hf.create_dataset(key, data=popdist_B.data[key])

    # # # PLOT THE HISTOGRAM PF THE POPULATION 

    # for key in popdist_B.data:
    #     plt.hist(popdist_B.data[key].mean(axis=1), bins='auto', histtype='step', lw=3, alpha=0.8, color='black')
    #     plt.xlabel(f'{key}')
    #     plt.savefig(f'true_{key}.png', dpi=200)
    #     plt.close()
