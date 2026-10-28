"""Plot helper for the EMRI SNR training scripts."""
import numpy as np
import matplotlib.pyplot as plt

output_dir = './data/PRE_TRAIN_DATA/{events}_events/'.format(events='1E6')

injection_params_lt = np.load(f'{output_dir}/injection_params_T_lt_T_obs.npy')
injection_params_gt = np.load(f'{output_dir}/injection_params_T_gt_T_obs.npy')

M = np.concatenate([10**injection_params_lt[:, 0], 10**injection_params_gt[:, 0]])
a = np.concatenate([injection_params_lt[:, 1], injection_params_gt[:, 1]])

p0 = np.concatenate([injection_params_lt[:, 2], injection_params_gt[:, 2]])
e0 = np.concatenate([injection_params_lt[:, 3], injection_params_gt[:, 3]])
Y0 = np.concatenate([injection_params_lt[:, 4], injection_params_gt[:, 4]])

params_to_plot = ['lgMBH', 'a', 'p0', 'e0', 'Y0']
params_to_plot_data = [M, a, p0, e0, Y0]

for i, params in enumerate(params_to_plot):
    plt.hist(params_to_plot_data[i], bins=50, alpha=0.7)
    plt.title(params)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{params}.png')
    plt.close()