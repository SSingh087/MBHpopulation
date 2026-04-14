import os, sys, argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import corner


parser = argparse.ArgumentParser(description="Generate data for training.")
parser.add_argument("--OBSERVING_WINDOW", type=float, required=True, help="Observing window in days")
parser.add_argument("--PLOT_FISHER", type=bool, default=False)
parser.add_argument("--PLOT_CORNER", type=bool, default=False)
parser.add_argument("--PLOT_COVARIANCE", type=bool, default=False)
parser.add_argument("--PLOT_HISTOGRAMS", type=bool, default=False)

args = parser.parse_args()

N_samples_theta = 1000

# WE will need to add EMRI SNRs here for now we assume all EMRIs will be detected
# with h5py.File(f'../data_generation/DATA/all_galaxies_EMRI_SNR_results.h5', 'r') as hf:
#     galaxy_to_events = {g: hf[g]["event_index"][:] for g in hf}
with h5py.File(f'../data_generation/DATA/all_galaxies_EMRI_events.h5', 'r') as hf:
    all_galaxies = {g: {k: hf[g][k][()] for k in hf[g]} for g in hf}

hf.close()


true_data = []
noisy_data = []

with h5py.File(f'./true_data_EMRI.h5', 'w') as hf_true_data, \
     h5py.File(f'./noisy_data_EMRI.h5', 'w') as hf_noisy_data:

    for gal in all_galaxies:
        num_events = len(all_galaxies[gal]['e0'])
        print(f"\n[GALAXY] {gal} - {num_events} events")
        
        theta0 = {}

        for k, v in all_galaxies[gal].items():
            if isinstance(v, (np.ndarray, list)):  # For array-like variables
                theta0[k] = v[:num_events]  # Direct slicing instead of per-indexing
            else:  # Scalar variables
                theta0[k] = np.full((num_events,), v)  # Replicate scalar for all events if needed
        
        catalogue_true_all = {
            param: np.random.normal(
                loc=theta0[param][:, None],   # shape (num_events, 1)
                scale=1e-3,
                size=(num_events, N_samples_theta)
            )
            for param in theta0
        }

        for event_idx in range(num_events):
            catalogue_true = {param: catalogue_true_all[param][event_idx] for param in theta0}
            true_data.append(catalogue_true)

    for key in all_galaxies[gal].keys():
        print(f"[SAVING] '{key}' to HDF5 dataset...")
        true_data_array = np.array([entry[key] for entry in true_data])
        hf_true_data.create_dataset(key, data=true_data_array, compression="gzip")


# breakpoint()
# plot_population_histograms(np.array([list(c.values()) for c in true_data]), PARAMS, f'./plots/true_data_histograms_EMRI.png', bins=30)


