import argparse
import h5py
import numpy as np
from multiprocessing import Pool, set_start_method
from fish_TDE import *
from utils_fisher import *
import os

# ----------------------------
# Arguments
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--OBSERVING_WINDOW", type=float, required=True)
parser.add_argument("--BANDS", nargs="+", default=["ztfg", "ztfr", "ztfi"])
parser.add_argument("--SURVEY", type=str, default="ztf")
parser.add_argument("--NPROC", type=int, default=8)
parser.add_argument("--PLOT_FISHER", type=bool, default=False)
parser.add_argument("--PLOT_CORNER", type=bool, default=False)
parser.add_argument("--PLOT_COVARIANCE", type=bool, default=False)
parser.add_argument("--PLOT_HISTOGRAMS", type=bool, default=False)
args = parser.parse_args()

OBSERVING_WINDOW = args.OBSERVING_WINDOW
BANDS = args.BANDS

PLOT_FISHER = args.PLOT_FISHER
PLOT_CORNER = args.PLOT_CORNER
PLOT_COVARIANCE = args.PLOT_COVARIANCE
PLOT_HISTOGRAMS = args.PLOT_HISTOGRAMS

def plotting_enabled():
    return PLOT_FISHER or PLOT_CORNER or PLOT_COVARIANCE or PLOT_HISTOGRAMS

SIM_FN, SURVEY_STR, t0_mjd = choose_simulator(args.SURVEY)
PARAMS = list(PARAM_INFO.keys())

with h5py.File('../data_generation/DATA/all_galaxies_TDE_events.h5', 'r') as hf:
    all_galaxies = {g: {k: hf[g][k][()] for k in hf[g]} for g in hf}

with h5py.File(f'../data_generation/DATA/all_galaxies_TDE_SNR_results_{args.SURVEY.upper()}.h5', 'r') as hf:
    galaxy_to_events = {g: hf[g]["event_index"][:] for g in hf}

EVENT_TASKS = []
for gal, events in galaxy_to_events.items():
    for eidx in events:
        EVENT_TASKS.append((gal, int(eidx)))

def process_event(task):
    gal, eidx = task
    data_gal = all_galaxies[gal]

    print(f"[WORKER] Galaxy {gal} event {eidx}")

    theta0 = {
        "redshift": float(data_gal["z_gal"]),
        "mbh_6": float(data_gal["lgMBH_mass"] - 6),
        "stellar_mass": float(data_gal["star_mass"]),
        "eta": 0.1,
        "alpha": 0.1,
        "beta": 1.001,
        "t0_mjd_transient": t0_mjd,
        "t0": t0_mjd,
        "ra": float(data_gal["ra"]),
        "dec": float(data_gal["dec"]),
    }

    # simulate
    df = run_telescope_simulator(theta0, SIM_FN, SURVEY_STR, OBSERVING_WINDOW)
    data_by_band = extract_observed_data_by_band(df, BANDS)
    times_by_band, bands = extract_times_by_band(data_by_band)

    # Fisher 1
    F1, C1, derivs1 = compute_fisher(theta0, times_by_band, bands)
    truths1 = [theta0[p] for p in PARAMS]
    true_samples = np.random.multivariate_normal(truths1, C1, size=2000)

    # Fisher 2
    idx = np.random.choice(true_samples.shape[0])
    theta_new = {p: true_samples[idx, i] for i, p in enumerate(PARAMS)}
    F2, C2, derivs2 = compute_fisher(theta_new, times_by_band, bands)
    truths2 = [theta_new[p] for p in PARAMS]
    noisy_samples = np.random.multivariate_normal(truths2, C2, size=2000)

    return (gal, eidx, F1, C1, truths1, F2, C2, truths2, true_samples, noisy_samples)


if __name__ == "__main__":

    set_start_method("spawn", force=True)

    hf_fisher = h5py.File(f'./fisher_results_TDE_{args.SURVEY.upper()}.h5', 'w')
    hf_true   = h5py.File(f'./true_data_TDE_{args.SURVEY.upper()}.h5', 'w')
    hf_noisy  = h5py.File(f'./noisy_data_TDE_{args.SURVEY.upper()}.h5', 'w')

    true_data = {p: [] for p in PARAMS}
    noisy_data = {p: [] for p in PARAMS}

    # Parallel processing over events since events >> number of galaxies, and each event is independent.
    with Pool(args.NPROC) as pool:
        for result in pool.imap_unordered(process_event, EVENT_TASKS):
            (
                gal, eidx,
                F1, C1, truths1,
                F2, C2, truths2,
                true_samples, noisy_samples
            ) = result

            if gal not in hf_fisher:
                gal_group = hf_fisher.create_group(gal)
            else:
                gal_group = hf_fisher[gal]

            egrp = gal_group.create_group(f"event_{eidx}")
            egrp.create_dataset("fisher_matrix", data=F1)
            egrp.create_dataset("covariance_matrix", data=C1)
            egrp.create_dataset("errors", data=np.sqrt(np.diag(C1)))
            egrp.create_dataset("truths", data=truths1)
            egrp.create_dataset("fisher_matrix_new", data=F2)
            egrp.create_dataset("covariance_matrix_new", data=C2)
            egrp.create_dataset("errors_new", data=np.sqrt(np.diag(C2)))
            egrp.create_dataset("truths_new", data=truths2)

            for i, p in enumerate(PARAMS):
                true_data[p].append(true_samples[:, i])
                noisy_data[p].append(noisy_samples[:, i])

            print(f"[SAVED] Galaxy {gal} event {eidx}")

    # Save global sample datasets EXACTLY like your original script

    for p in PARAMS:
        true_data_array = np.vstack(true_data[p])   # shape = (N_events, 2000)
        noisy_data_array = np.vstack(noisy_data[p]) # shape = (N_events, 2000)

        hf_true.create_dataset(p, data=true_data_array, compression="gzip")
        hf_noisy.create_dataset(p, data=noisy_data_array, compression="gzip")


    hf_fisher.close()
    hf_true.close()
    hf_noisy.close()

    print("\nFISHER DONE FOR {} \n".format(args.SURVEY.upper()))