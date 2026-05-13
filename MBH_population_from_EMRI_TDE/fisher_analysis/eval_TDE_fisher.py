import argparse
import h5py
import numpy as np
from multiprocessing import Pool, set_start_method
from fish_TDE import *
from utils_fisher import *

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

SIM_FN, SURVEY_STR, t0_mjd = choose_simulator(args.SURVEY)
PARAMS = list(PARAM_INFO.keys())

with h5py.File(f'./data/astrophysical_data/{args.GALAXIES}/all_galaxies_TDE_events_{ "clustering" if args.is_clustering else "no_clustering" }.h5', 'r') as hf:
    all_galaxies = {g: {k: hf[g][k][()] for k in hf[g]} for g in hf}

################ THIS NEEDS TO BE CHANGED TO MATCH IS_CLUSTERING FLAG IN THE ABOVE FILE ################
# with h5py.File(f'./data/astrophysical_data/{args.GALAXIES}/all_galaxies_TDE_SNR_results_{args.SURVEY.upper()}_{ "clustering" if args.is_clustering else "no_clustering" }.h5', 'r') as hf:
####################################################################################################

with h5py.File(f'./data/astrophysical_data/{args.GALAXIES}/all_galaxies_TDE_SNR_results_{args.SURVEY.upper()}.h5', 'r') as hf:
    galaxy_to_events = {g: hf[g]["event_index"][:] for g in hf}

EVENT_TASKS = []
for gal, events in galaxy_to_events.items():
    for eidx in events:
        EVENT_TASKS.append((gal, int(eidx)))


def sample_truncated_gaussian(mean, cov, n_target, n_draw=50000):
    samples = np.random.multivariate_normal(mean, cov, size=n_draw)
    
    # print(f"{PARAMS[0]}:", samples[:, 0].min(), samples[:, 0].max())
    # print(f"{PARAMS[1]}:", samples[:, 1].min(), samples[:, 1].max())
    # print(f"{PARAMS[2]}:", samples[:, 2].min(), samples[:, 2].max())
    # print(f"{PARAMS[3]}:", samples[:, 3].min(), samples[:, 3].max())
    # print(f"{PARAMS[4]}:", samples[:, 4].min(), samples[:, 4].max())
    # print(f"{PARAMS[5]}:", samples[:, 5].min(), samples[:, 5].max())

    mask = (
        (0.001 <= samples[:, 0]) & (samples[:, 0] <= 10) &
        (-2 <= samples[:, 1])    & (samples[:, 1] <= 4) &
        (0.001 <= samples[:, 2]) & (samples[:, 2] <= 5) &
        (0.001 <= samples[:, 3]) & (samples[:, 3] <= 0.1) &
        (0.01 <= samples[:, 4])  & (samples[:, 4] <= 0.1) &
        (1.001 <= samples[:, 5]) & (samples[:, 5] <= 30)
    )

    valid = samples[mask]

    if len(valid) == 0:
        raise RuntimeError("No samples inside prior")

    if len(valid) >= n_target:
        return valid[:n_target]

    # Resample with replacement
    idx = np.random.choice(len(valid), size=n_target, replace=True)
    return valid[idx]


def process_event(task):
    gal, eidx = task
    try:
        data_gal = all_galaxies[gal]
        galaxy_events = galaxy_to_events[gal]

        local_idx = np.where(galaxy_events == eidx)[0][0]

        print(f"[WORKER] Galaxy {gal} event {eidx}")

        theta0 = {
            "redshift": float(data_gal["z_gal"]),
            "mbh_6": float(data_gal["lgMBH_mass"] - 6),
            "stellar_mass": float(data_gal["star_mass"]),
            
            "eta": float(data_gal["eta"][local_idx]),
            "alpha": float(data_gal["alpha"][local_idx]),
            "beta": float(data_gal["beta"][local_idx]),

            "t0_mjd_transient": float(data_gal["t0_mjd_transient_" + args.SURVEY.upper()][local_idx]),
            "t0": float(data_gal["t0_mjd_transient_" + args.SURVEY.upper()][local_idx]),
            "ra": float(data_gal["ra_rad"]),
            "dec": float(data_gal["dec_rad"]),
        }

        # print(f"[WORKER] Simulating for Galaxy {gal} event {eidx} with parameters: {theta0}")
        # simulate
        df = run_telescope_simulator(theta0, SIM_FN, SURVEY_STR, OBSERVING_WINDOW)
        data_by_band = extract_observed_data_by_band(df, BANDS)
        times_by_band, bands = extract_times_by_band(data_by_band)
        
        # Fisher 1
        F1, C1, derivs1 = compute_fisher(theta0, times_by_band, bands)
        truths1 = [theta0[p] for p in PARAMS]

        true_samples = sample_truncated_gaussian(mean=np.array(truths1), cov=C1, n_target=N_SAMPLES_PER_THETA)

        # Fisher 2 - with retry logic and fallback
        F2, C2, truths2, noisy_samples = None, None, None, None
        max_attempts = 10
        
        for attempt in range(max_attempts):
            try:
                # sample a *small* batch from F1 to get a valid theta_new
                theta_candidates = sample_truncated_gaussian(mean=np.array(truths1), cov=C1, n_target=50,n_draw=10000)

                theta_new_vec = theta_candidates[0]
                theta_new = {p: theta_new_vec[i] for i, p in enumerate(PARAMS)}


                # Try to compute Fisher matrix
                F2, C2, derivs2 = compute_fisher(theta_new, times_by_band, bands)
                truths2 = [theta_new[p] for p in PARAMS]
                
                # Sample from F2 covariance with bounds
                noisy_samples = sample_truncated_gaussian(mean=np.array(truths2), cov=C2, n_target=N_SAMPLES_PER_THETA)
                print(f"[SUCCESS] Galaxy {gal} event {eidx}: F2 computed on attempt {attempt+1}")
                break  
                
            except (ValueError, np.linalg.LinAlgError, TypeError) as e:
                print(f"[ATTEMPT {attempt+1} FAILED] Galaxy {gal} event {eidx}: {type(e).__name__}")
                if attempt == max_attempts - 1:
                    # All attempts exhausted - use F1 as fallback
                    print(f"[FALLBACK] Galaxy {gal} event {eidx}: Using F1/C1 (all F2 attempts failed)")
                    F2, C2 = F1, C1
                    truths2 = truths1
                    
                    noisy_samples = true_samples.copy()
                    truths2 = truths1.copy()

                continue
        
        return (gal, eidx, F1, C1, truths1, F2, C2, truths2, true_samples, noisy_samples)
    
    except Exception as e:
        print(f"[CRITICAL ERROR] Galaxy {gal} event {eidx}: {type(e).__name__}: {e}")
        return None


if __name__ == "__main__":

    set_start_method("spawn", force=True)

    corner_dir = f"./data/fisher_data/{args.GALAXIES}/corners_{args.SURVEY.upper()}_{ 'clustering' if args.is_clustering else 'no_clustering' }"
    if PLOT_CORNER and not os.path.exists(corner_dir):
        os.makedirs(corner_dir, exist_ok=True)

    hf_fisher = h5py.File(f'./data/fisher_data/{args.GALAXIES}/fisher_results_TDE_{args.SURVEY.upper()}_{ "clustering" if args.is_clustering else "no_clustering" }.h5', 'w')
    hf_true   = h5py.File(f'./data/fisher_data/{args.GALAXIES}/true_data_TDE_{args.SURVEY.upper()}_{ "clustering" if args.is_clustering else "no_clustering" }.h5', 'w')
    hf_noisy  = h5py.File(f'./data/fisher_data/{args.GALAXIES}/noisy_data_TDE_{args.SURVEY.upper()}_{ "clustering" if args.is_clustering else "no_clustering" }.h5', 'w')

    true_samples_list = []
    noisy_samples_list = []

    event_results = []

    # check with for loop first to debug and ensure it works for a few events before parallelizing
    # for i, task in enumerate(EVENT_TASKS):
    #     result = process_event(task)
    #     if result is None:
    #         print(f"[SKIPPED] Event {i} returned None (critical failure)")
    #         continue
        
    #     (
    #         gal, eidx,
    #         F1, C1, truths1,
    #         F2, C2, truths2,
    #         true_samples, noisy_samples
    #     ) = result

    # Parallel processing over events since events >> number of galaxies, and each event is independent.
    with Pool(args.NPROC) as pool:
        for result in pool.imap_unordered(process_event, EVENT_TASKS):
            # Skip failed events (returns None)
            if result is None:
                print("[SKIPPED] Event returned None (critical failure)")
                continue
            
            (
                gal, eidx,
                F1, C1, truths1,
                F2, C2, truths2,
                true_samples, noisy_samples
            ) = result

            true_samples_list.append(true_samples)
            noisy_samples_list.append(noisy_samples)

            event_results.append(( gal, eidx, truths1, truths2, true_samples, noisy_samples))

            if gal not in hf_fisher:
                gal_group = hf_fisher.create_group(gal)

                gal_group.attrs["z_gal"] = float(all_galaxies[gal]["z_gal"])
                gal_group.attrs["stellar_mass"] = float(all_galaxies[gal]["star_mass"])
                gal_group.attrs["ra_rad"] = float(all_galaxies[gal]["ra_rad"])
                gal_group.attrs["dec_rad"] = float(all_galaxies[gal]["dec_rad"])
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

    
    for i, key in enumerate(PARAMS):    
        true_data_array = np.vstack([arr[:, i] for arr in true_samples_list])
        noisy_data_array = np.vstack([arr[:, i] for arr in noisy_samples_list])

        hf_true.create_dataset(key, data=true_data_array, compression="gzip")
        hf_noisy.create_dataset(key, data=noisy_data_array, compression="gzip")
        print(f"[SAVED] Samples for parameter '{key}' with shape {true_data_array.shape} (true) and {noisy_data_array.shape} (noisy)")

    hf_fisher.close()
    hf_true.close()
    hf_noisy.close()

    print("\nFISHER DONE FOR {} \n".format(args.SURVEY.upper()))
    
    # ---------- PLOTS ----------
    if PLOT_CORNER:
        for gal, eidx, truths1, truths2, true_samples, noisy_samples in event_results:
            print(f"[PLOTTING] Generating corner plot for {gal} event {eidx}...")
            plot_corner(
                samples=true_samples,
                truths=truths1,
                params=PARAMS,
                filename=f"{corner_dir}/corner_true_{gal}_event_{eidx}.png",
                title=f"True Fisher – {gal}, event {eidx}",
            )

            plot_corner(
                samples=noisy_samples,
                truths=truths2,
                params=PARAMS,
                filename=f"{corner_dir}/corner_noisy_{gal}_event_{eidx}.png",
                title=f"Noisy Fisher – {gal}, event {eidx}",
            )

    if PLOT_FISHER:
        for gal, eidx, truths1, truths2, true_samples, noisy_samples in event_results:
            print(f"[PLOTTING] Generating Fisher plot for {gal} event {eidx}...")
            plot_fisher_matrices(
                F1=np.linalg.inv(C1),
                F2=np.linalg.inv(C2),
                params=PARAMS,
                filename=f"{loc}/fisher_{gal}_event_{eidx}.png",
                title=f"Fisher Matrices – {gal}, event {eidx}",
            )

    if PLOT_COVARIANCE:
        for gal, eidx, truths1, truths2, true_samples, noisy_samples in event_results:
            print(f"[PLOTTING] Generating covariance plot for {gal} event {eidx}...")
            plot_covariance_matrices(
                C1=C1,
                C2=C2,
                params=PARAMS,
                filename=f"{loc}/covariance_{gal}_event_{eidx}.png",
                title=f"Covariance Matrices – {gal}, event {eidx}",
            )

    if PLOT_HISTOGRAMS:
        for gal, eidx, truths1, truths2, true_samples, noisy_samples in event_results:
            print(f"[PLOTTING] Generating histograms for {gal} event {eidx}...")
            plot_parameter_histograms(
                samples=true_samples,
                params=PARAMS,
                filename=f"{loc}/histograms_true_{gal}_event_{eidx}.png",
                title=f"True Samples – {gal}, event {eidx}",
            )

            plot_parameter_histograms(
                samples=noisy_samples,
                params=PARAMS,
                filename=f"{loc}/histograms_noisy_{gal}_event_{eidx}.png",
                title=f"Noisy Samples – {gal}, event {eidx}",
            )