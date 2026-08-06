#!/usr/bin/env python

import os, sys, argparse
sys.path.insert(0, os.path.abspath('../../poplar'))

from population import *
import torch, pickle, h5py, cupy
import numpy as np

from few.trajectory.inspiral import EMRIInspiral
from stableemrifisher.fisher import StableEMRIFisher
from few.waveform import GenerateEMRIWaveform
from scipy.stats import multivariate_normal

parser = argparse.ArgumentParser(description="Generate data for training.")
parser.add_argument("--work_dir", type=str, required=True, help="work_dir")
args = parser.parse_args()


if not os.path.exists(f'{args.work_dir}/noisy_fisher_data/'):
    os.makedirs(f'{args.work_dir}/noisy_fisher_data/')

use_gpu = True

inspiral_kwargs = {
    "DENSE_STEPPING": 0,
    "max_init_len": int(1e5),
    "err": 1e-10,
    "use_rk4": True,
}

# # keyword arguments for summation generator
sum_kwargs = {
    # "use_gpu": use_gpu,
    "pad_output": True,
}

## fixed waveform parameters
dt = 50.0  # seconds 
traj = EMRIInspiral(func="PN5")

N_samples_theta = 1000

wave_params = np.load(f'{args.work_dir}/true_population_params.npy')

M = np.float64(10**wave_params[:, 0])
mu = np.float64(10**wave_params[:, 1])
a = np.float64(wave_params[:, 2])
p0 = np.float64(wave_params[:, 3])
e0 = np.float64(wave_params[:, 4])
Y0 = np.float64(wave_params[:, 5])
dist = np.float64(wave_params[:, 6])
qS = np.float64(wave_params[:, 7])
phiS = np.float64(wave_params[:, 8])
qK = np.float64(wave_params[:, 9])
phiK = np.float64(wave_params[:, 10])
Phi_phi0 = np.float64(wave_params[:, 11])
Phi_theta0 = np.float64(wave_params[:, 12])
Phi_r0 = np.float64(wave_params[:, 13])
T = np.float64(wave_params[:, 14])

# ==============================================================================
# Calculate posteriors for the events which have noise added

waveform_model = GenerateEMRIWaveform('Pn5AAKWaveform', return_list=True, inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs)

# varied parameters
# https://github.com/cchapmanbird/StableEMRIFisher/blob/e1063087979e9baeb39e62a9f3597895c6f9f7c9/stableemrifisher/fisher/fisher.py#L53
param_names = ['M', 'mu', 'a', 'p0', 'e0', 'Y0', 'dist', 'qS', 'phiS',  'qK', 'phiK', 'Phi_phi0', 'Phi_theta0', 'Phi_r0']

minmax = {
    'Phi_phi0': [0.1, 2 * np.pi * 0.99],
    'Phi_r0': [0.1, 2 * np.pi * 0.999],
    'Phi_theta0': [0.1, 2 * np.pi * 0.99],
    'qS': [0.1, np.pi * 0.99],
    'qK': [0.1, np.pi * 0.99],
    'phiS': [0.1, 2 * np.pi * 0.99],
    'phiK': [0.1, 2 * np.pi * 0.99]
}

# Initialize variables outside the loop
M_noise, mu_noise, a_noise, e0_noise, p0_noise = [None] * 5
Y0_noise, qS_noise, phiS_noise, dist_noise, qK_noise, phiK_noise = [None] * 6
Phi_phi0_noise, Phi_theta0_noise, Phi_r0_noise = [None] * 3


kept_catalogue_indices = np.load(f'{args.work_dir}/events_detected_indices.npy')

noisy_data = np.empty((len(kept_catalogue_indices), 13, N_samples_theta))

for num, num_event in enumerate(kept_catalogue_indices):
    print("DOING FOR EVENT", num_event)
    # TO DO !!
    # ADD A CHECK SUCH THAT IF THE FISHER MATRIX HAS BEEN CALCULATED OR
    # IT HAS FAILED 
    cov_mat_true = np.load(f'{args.work_dir}/fisher_data/true_cov_mat_{num_event}.npy')
    
    # Draw a point from a distribution since true parameters are never detected.
    i = 0
    while i < 100000:
        i+=1
        if i % 10000 == 0:
            print(f"checking for {i}")
        # Sample from the multivariate normal distribution
        (M_noise, mu_noise, a_noise, p0_noise, e0_noise,
         Y0_noise, dist_noise, qS_noise, phiS_noise, qK_noise, phiK_noise,
         Phi_phi0_noise, Phi_theta0_noise, Phi_r0_noise) = \
        multivariate_normal(mean=(M[num_event], mu[num_event], a[num_event], p0[num_event], e0[num_event], Y0[num_event],
                                 dist[num_event], qS[num_event], phiS[num_event], qK[num_event], phiK[num_event],
                                 Phi_phi0[num_event], Phi_theta0[num_event], Phi_r0[num_event]),
                            cov=cov_mat_true, allow_singular=True).rvs()

        # Check if the conditions are met
        # and because it has to be within the blob of the 14D space.
        if (minmax['Phi_phi0'][0] <= Phi_phi0_noise <= minmax['Phi_phi0'][1] and
            minmax['Phi_r0'][0] <= Phi_r0_noise <= minmax['Phi_r0'][1] and
            minmax['Phi_theta0'][0] <= Phi_theta0_noise <= minmax['Phi_theta0'][1] and
            minmax['qS'][0] <= qS_noise <= minmax['qS'][1] and
            minmax['qK'][0] <= qK_noise <= minmax['qK'][1] and
            minmax['phiS'][0] <= phiS_noise <= minmax['phiS'][1] and
            minmax['phiK'][0] <= phiK_noise <= minmax['phiK'][1]):
            break  # Exit the loop if all conditions are satisfied

    new_means = np.array([M_noise, mu_noise, a_noise, p0_noise, e0_noise,
                            Y0_noise, dist_noise, qS_noise, phiS_noise, qK_noise, phiK_noise,
                            Phi_phi0_noise, Phi_theta0_noise, Phi_r0_noise])

    np.save(f'{args.work_dir}/noisy_fisher_data/noisy_mean_{num_event}.npy', new_means)

    # again calculate Fisher matrix
    if T[num_event] > 4 :
        print("T = ", T[num_event])

        fish_noise = StableEMRIFisher(M_noise, mu_noise, a_noise, p0_noise, e0_noise,
                                        Y0_noise, dist_noise, qS_noise, phiS_noise, qK_noise, phiK_noise,
                                        Phi_phi0_noise, Phi_theta0_noise, Phi_r0_noise,
                                        dt=dt, T=4.0, EMRI_waveform_gen=waveform_model,
                                        param_names=param_names, Ndelta=8, stats_for_nerds=False, use_gpu=True,
                                        filename=f'{args.work_dir}/noisy_fisher_data/{num_event}',
                                        CovEllipse=True, live_dangerously=False)
        
        _, cov_mat_noise = fish_noise()
    
    if T[num_event] < 4 or T[num_event] == 4 :
        print("T =", T[num_event])
        
        fish_noise = StableEMRIFisher(M_noise, mu_noise, a_noise, p0_noise, e0_noise,
                                        Y0_noise, dist_noise, qS_noise, phiS_noise, qK_noise, phiK_noise,
                                        Phi_phi0_noise, Phi_theta0_noise, Phi_r0_noise,
                                        dt=dt, T=T[num_event], EMRI_waveform_gen=waveform_model,
                                        param_names=param_names, Ndelta=8, stats_for_nerds=False, use_gpu=True,
                                        filename=f'{args.work_dir}/noisy_fisher_data/{num_event}',
                                        CovEllipse=True, live_dangerously=False)
        
        _, cov_mat_noise = fish_noise()
    
    # save true Cov Matrix
    np.save(f'{args.work_dir}/noisy_fisher_data/noise_cov_mat_{num_event}', cov_mat_noise)

    
    # Sample from the multivariate normal distribution
    noisy_data_samples = multivariate_normal(mean=(M_noise, mu_noise, a_noise, p0_noise, e0_noise, Y0_noise,
                                                    dist_noise, qS_noise, phiS_noise, qK_noise, phiK_noise,
                                                    Phi_phi0_noise, Phi_theta0_noise, Phi_r0_noise),
                                            cov=cov_mat_noise, allow_singular=True).rvs(size=N_samples_theta).T
    # p0 is excluded from here
    # for this sequence ['M', 'mu', 'a', --'p0'--, 'e0', 'Y0', 'dist', 'qS', 'phiS',  'qK', 'phiK', 'Phi_phi0', 'Phi_theta0', 'Phi_r0']
    
    noisy_data[num] = np.vstack([np.log10(noisy_data_samples[0:2]), noisy_data_samples[2], noisy_data_samples[4:]])
    cupy.get_default_memory_pool().free_all_blocks()
    cupy.cuda.Device().synchronize()

kept_catalogue = {}
filtered_distributions = {k: v for k, v in popdist_B.distributions.items() if k != "T"}

for i, key in enumerate(filtered_distributions.keys()):
    print(f"saving true data for {key}")
    kept_catalogue[key] = noisy_data[:, i, :]

popdist_B.data = kept_catalogue

hf = h5py.File(f'{args.work_dir}/noisy_pop_data.h5', 'w')
for key in popdist_B.data:
    hf.create_dataset(key, data=popdist_B.data[key])

hf.close()