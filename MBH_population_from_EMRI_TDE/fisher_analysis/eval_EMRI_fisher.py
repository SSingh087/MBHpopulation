import argparse
import os
import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES", type=int, required=True, help="Number of galaxies")
parser.add_argument("--OBSERVING_WINDOW", type=float, required=True)
parser.add_argument("--BANDS", nargs="+", default=["ztfg", "ztfr", "ztfi"])
parser.add_argument("--SURVEY", type=str, default="ztf")
parser.add_argument("--is_clustering", type=int, default=0)
parser.add_argument("--NPROC", type=int, default=8)
parser.add_argument("--PLOT_FISHER", action="store_true")
parser.add_argument("--PLOT_CORNER", action="store_true")
parser.add_argument("--PLOT_COVARIANCE", action="store_true")
parser.add_argument("--PLOT_HISTOGRAMS", action="store_true")
args = parser.parse_args()

loc = f'./data/fisher_data/{args.GALAXIES}'
if not os.path.exists(loc):
    os.makedirs(loc)

N_SAMPLES_PER_THETA = 2000
OBSERVING_WINDOW = args.OBSERVING_WINDOW
BANDS = args.BANDS

PLOT_FISHER = args.PLOT_FISHER
PLOT_CORNER = args.PLOT_CORNER
PLOT_COVARIANCE = args.PLOT_COVARIANCE
PLOT_HISTOGRAMS = args.PLOT_HISTOGRAMS

print("Plotting options:    ")
print(f"  - Fisher: {PLOT_FISHER}")
print(f"  - Corner: {PLOT_CORNER}")
print(f"  - Covariance: {PLOT_COVARIANCE}")
print(f"  - Histograms: {PLOT_HISTOGRAMS}")

# WE will need to add EMRI SNRs here for now we assume all EMRIs will be detected
# with h5py.File(f'../data_generation/DATA/all_galaxies_EMRI_SNR_results.h5', 'r') as hf:
#     galaxy_to_events = {g: hf[g]["event_index"][:] for g in hf}
with h5py.File(f'./data/astrophysical_data/{args.GALAXIES}/all_galaxies_EMRI_events_{ "clustering" if args.is_clustering else "no_clustering" }.h5', 'r') as hf:
    all_galaxies = {g: {k: hf[g][k][()] for k in hf[g]} for g in hf}

true_data = []
noisy_data = []

if not os.path.exists(loc):
    os.makedirs(loc)

with h5py.File(f'{loc}/true_data_EMRI_{ "clustering" if args.is_clustering else "no_clustering" }.h5', 'w') as hf_true_data, \
     h5py.File(f'{loc}/noisy_data_EMRI_{ "clustering" if args.is_clustering else "no_clustering" }.h5', 'w') as hf_noisy_data:

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
                size=(num_events, N_SAMPLES_PER_THETA)
            )
            for param in theta0
        }
        
        # Add CO_mass as a separate parameter with fixed true value of 10
        # sampled with small scatter around that value
        catalogue_true_all["CO_mass"] = np.random.normal(
            loc=10.0,  # True CO mass value
            scale=1e-3,  # Same small scatter as other parameters
            size=(num_events, N_SAMPLES_PER_THETA)
        )

        for event_idx in range(num_events):
            catalogue_true = {param: catalogue_true_all[param][event_idx] for param in catalogue_true_all}
            true_data.append(catalogue_true)

    # Save all parameters (including CO_mass which was added)
    all_params = set()
    for entry in true_data:
        all_params.update(entry.keys())

    breakpoint()
    
    for key in all_params:
        true_data_array = np.array([entry[key] for entry in true_data])
        hf_true_data.create_dataset(key, data=true_data_array, compression="gzip")
        print(f"[SAVING] '{key}' to HDF5 dataset of size {true_data_array.shape}...")


