import argparse
import h5py
from fish_TDE import *

parser = argparse.ArgumentParser()
parser.add_argument("--OBSERVING_WINDOW", type=float, required=True)
parser.add_argument("--BANDS", nargs="+", default=["ztfg", "ztfr", "ztfi"])
parser.add_argument("--SURVEY", type=str, default="ztf")
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

# choose simulator
SIM_FN, SURVEY_STR, t0_mjd = choose_simulator(args.SURVEY)


with h5py.File('../data_generation/DATA/all_galaxies_TDE_events.h5', 'r') as hf:
    all_galaxies = {g: {k: hf[g][k][()] for k in hf[g]} for g in hf}


with h5py.File(f'../data_generation/DATA/all_galaxies_TDE_SNR_results_{args.SURVEY.upper()}.h5', 'r') as hf:
    galaxy_to_events = {g: hf[g]["event_index"][:] for g in hf}


PARAMS = list(PARAM_INFO.keys())

true_data = []
noisy_data = []

with h5py.File(f'./fisher_results_{args.SURVEY.upper()}.h5', 'w') as hf_injection, \
    h5py.File(f'./true_data_{args.SURVEY.upper()}.h5', 'w') as hf_true_data, \
     h5py.File(f'./noisy_data_{args.SURVEY.upper()}.h5', 'w') as hf_noisy_data:

    for gal, events in galaxy_to_events.items():
        
        galaxy_group = hf_injection.create_group(gal)
      
        data_gal = all_galaxies[gal]

        for eidx in events:
            
            catalogue_true, catalogue_noisy = {}, {}

            print(f"\n[EVENT] {gal} event {eidx}")

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

            # FIRST FISHER
            fisher_matrix, covariance_matrix, derivatives = compute_fisher(theta0, times_by_band, bands)
            errors_on_params = np.sqrt(np.diag(covariance_matrix))

            print("\n[FISHER ERRORS RUN 1]")
            for p, e in zip(PARAMS, errors_on_params):
                print(f"  {p:15s} = {e:.4g}")

            truths = [theta0[p] for p in PARAMS]
            true_samples = np.random.multivariate_normal(truths, covariance_matrix, size=2000)

            # SECOND FISHER (from samples)

            idx = np.random.choice(true_samples.shape[0])
            theta0_new = {p: true_samples[idx, i] for i, p in enumerate(PARAMS)}

            fisher_matrix_new, covariance_matrix_new, derivatives_new = compute_fisher(theta0_new, times_by_band, bands)
            errors_on_params_new = np.sqrt(np.diag(covariance_matrix_new))

            print("\n[FISHER ERRORS RUN 2 (from samples)]")
            for p, e in zip(PARAMS, errors_on_params_new):
                print(f"  {p:15s} = {e:.4g}")
            
            new_truths = [theta0_new[p] for p in PARAMS]
            noisy_samples = np.random.multivariate_normal(new_truths, covariance_matrix_new, size=2000)


            # SAVE TRUE AND NOISY SAMPLES
            for i, key in enumerate(PARAMS):
                catalogue_true[key] = true_samples[:, i]
                catalogue_noisy[key] = noisy_samples[:, i]
            true_data.append(catalogue_true)
            noisy_data.append(catalogue_noisy)


            event_group = galaxy_group.create_group(f"event_{eidx}")
            event_group.create_dataset("fisher_matrix", data=fisher_matrix)
            event_group.create_dataset("covariance_matrix", data=covariance_matrix)
            event_group.create_dataset("errors", data=errors_on_params)
            event_group.create_dataset("truths", data=truths)
            event_group.create_dataset("fisher_matrix_new", data=fisher_matrix_new)
            event_group.create_dataset("covariance_matrix_new", data=covariance_matrix_new)
            event_group.create_dataset("errors_new", data=errors_on_params_new)
            event_group.create_dataset("truths_new", data=new_truths)
            # for key in derivatives:
            #     event_group.create_dataset(f"derivatives_{key}", data=derivatives[key])
            #     event_group.create_dataset(f"derivatives_new_{key}", data=derivatives_new[key])
                
    
    for i, key in enumerate(PARAMS):
        print(f"[SAVING] '{key}' to HDF5 dataset...")
        true_data_array = np.array([entry[key] for entry in true_data])
        noisy_data_array = np.array([entry[key] for entry in noisy_data])
        hf_true_data.create_dataset(key, data=true_data_array, compression="gzip")
        hf_noisy_data.create_dataset(key, data=noisy_data_array, compression="gzip")


if plotting_enabled():
    pass
    # plot_corner(true_samples, truths, PARAMS, f"./fisher_TDE_corner_plots/corner_{gal}_event_{eidx}_{args.SURVEY.upper()}.pdf")
    # plot_fisher_matrix(fisher_matrix, PARAMS, f"./fisher_TDE_corner_plots/fisher_{gal}_event_{eidx}_{args.SURVEY.upper()}.pdf")
    # plot_covariance_matrix(covariance_matrix, PARAMS, f"./fisher_TDE_corner_plots/covariance_{gal}_event_{eidx}_{args.SURVEY.upper()}.pdf")
    # plot_parameter_histograms(true_samples, PARAMS, f"./fisher_TDE_corner_plots/histograms_{gal}_event_{eidx}_{args.SURVEY.upper()}.pdf")