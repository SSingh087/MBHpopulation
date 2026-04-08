import logging
logging.getLogger('bilby').disabled = True
logging.getLogger('redback').disabled = True

import argparse
import numpy as np
import h5py
from multiprocessing import Pool
import matplotlib.pyplot as plt
import corner
from scipy.stats import multivariate_normal

from redback.simulate_transients import SimulateGenericTransient, SimulateOpticalTransient

parser = argparse.ArgumentParser(description="Calculate SNR for TDE events.")
parser.add_argument("--OBSERVING_WINDOW", type=float, required=True)
parser.add_argument("--BANDS", nargs="+", default=["ztfg", "ztfr", "ztfi"], help="Bands to check for detections (e.g., 'ztfg ztfr ztfi' for ZTF)")
parser.add_argument("--SURVEY", type=str, default="ztf", help="Survey to simulate (e.g., 'ztf' or 'lsst')")

args = parser.parse_args()

OBSERVING_WINDOW = args.OBSERVING_WINDOW

BANDS = args.BANDS
SURVEY = args.SURVEY

if SURVEY == "ZTF":
    SIMULATE_FN = SimulateOpticalTransient.simulate_transient_in_ztf
    SURVEY = "ztf"
    t0_mjd_transient = 58288 # MJD for ZTF (can be adjusted)
elif SURVEY == "LSST":
    SIMULATE_FN = SimulateOpticalTransient.simulate_transient_in_rubin
    SURVEY = "Rubin_10yr_baseline"
    t0_mjd_transient = 60000 # MJD for Rubin (can be adjusted)
else:
    raise ValueError(f"Unsupported survey: {SURVEY}")


MODEL_NAME = "cooling_envelope"
# these are the range of stable deltas from testing, but we can adjust if needed (e.g., if derivatives are too noisy or zero)

PARAM_INFO = {
    "redshift":     {"delta": 1e-7, "type": "linear"},
    "mbh_6":        {"delta": 1e-7, "type": "linear"},
    "stellar_mass": {"delta": 1e-7, "type": "linear"},
    "eta":          {"delta": 5e-6, "type": "log10"},
    "alpha":        {"delta": 1e-3, "type": "log10"},
    "beta":         {"delta": 1e-1, "type": "linear"},
}

def run_telescope_simulator(theta):
    """Simulate ZTF light curve for given parameters."""
    sim = SIMULATE_FN(
        model=MODEL_NAME,
        survey=SURVEY,
        parameters=theta,
        model_kwargs={},
        end_transient_time=OBSERVING_WINDOW,
        snr_threshold=0.01,
        add_source_noise=False
    )
    cols = ['time (days)', 'magnitude', 'e_magnitude', 'band', 'detected']
    return sim.observations[cols]


def extract_observed_data_by_band(df):
    """Extract time/magnitude arrays per band."""
    grouped = df.groupby("band")
    return {
        band: {
            "time": g["time (days)"].to_numpy(float),
            "magnitude": g["magnitude"].to_numpy(float),
            "err": g["e_magnitude"].to_numpy(float),
        }
        for band, g in grouped
        if band in BANDS
    }


def extract_times_by_band(data_by_band):
    """Return timestamps per band (NO detection filtering)."""
    times_by_band = {b: d["time"] for b, d in data_by_band.items()}
    bands = list(times_by_band.keys())
    return times_by_band, bands

def run_generic_for_band(theta, band, times):
    """Compute model magnitudes for a single band at given times."""
    obs = SimulateGenericTransient(
        model=MODEL_NAME,
        parameters=theta,
        times=times,
        data_points=len(times),
        model_kwargs={'bands': (band,), 'output_format': 'magnitude'},
        multiwavelength_transient=True,
        noise_term=0.0,
    )
    df = obs.data
    df = df[df["band"] == band].sort_values("time")
    return df["true_output"].to_numpy(float)


def evaluate_model(theta, times_by_band, bands):
    return {b: run_generic_for_band(theta, b, times_by_band[b]) for b in bands}


def perturb_value(base, delta, kind):
    if kind == "linear":
        return base + delta, base - delta
    if kind == "log10":
        return 10**(np.log10(base) + delta), 10**(np.log10(base) - delta)
    raise ValueError


def numerical_derivative(param, theta_base, times_by_band, bands):
    info = PARAM_INFO[param]
    delta = info["delta"]
    kind = info["type"]

    val_p, val_m = perturb_value(theta_base[param], delta, kind)
    theta_p = {**theta_base, param: val_p}
    theta_m = {**theta_base, param: val_m}

    m_p = evaluate_model(theta_p, times_by_band, bands)
    m_m = evaluate_model(theta_m, times_by_band, bands)

    deriv = {}
    if kind == "linear":
        for b in bands:
            deriv[b] = (m_p[b] - m_m[b]) / (2 * delta)
    else:  # log10 case
        factor = 1.0 / (theta_base[param] * np.log(10.0))
        for b in bands:
            dm_du = (m_p[b] - m_m[b]) / (2 * delta)
            deriv[b] = dm_du * factor

    return deriv

def _deriv_worker(args):
    p, theta0, times_by_band, bands = args
    return p, numerical_derivative(p, theta0, times_by_band, bands)

def plot_corner(samples, truths, params, filename):
    fig = corner.corner(samples, labels=params, truths=truths, quantiles=[0.16, 0.5, 0.84],
                        truth_color='red', show_titles=True, title_fmt=".4g",
                        title_kwargs={"fontsize": 12}, color="blue",
                        hist_kwargs={"density": True, "color": "black"})
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"[SAVED] corner plot: {filename}\n")

if __name__ == "__main__":

    print("===========================================================")
    print("[INFO] Loading detected events for " + args.SURVEY.upper() + "...")
    print("===========================================================\n")

    with h5py.File(f'../data_generation/DATA/all_galaxies_TDE_SNR_results_{args.SURVEY.upper()}.h5', 'r') as hf:
        galaxy_to_events = {
            gal: hf[gal]["event_index"][:] for gal in hf.keys()
        }

    print("[INFO] Loading galaxy parameters...")
    with h5py.File('../data_generation/DATA/all_galaxies_TDE_events.h5', 'r') as hf:
        galaxies = {
            gal: {k: np.array(hf[gal][k]) for k in hf[gal].keys()}
            for gal in hf.keys()
        }

    # Filter only galaxies with detected events
    filtered = {gal: galaxies[gal] for gal in galaxy_to_events}

    print("[INFO] Loading Fisher covariances...")
    with h5py.File(f'./corner_samples_from_injection_{args.SURVEY.upper()}.h5', 'r') as hf_samples_injection:



    hf_samples_injection.close()
    
    PARAMS = list(PARAM_INFO.keys())

    with h5py.File(f'./fisher_results_from_samples_{args.SURVEY.upper()}.h5', 'w') as hf_out, \
        h5py.File(f'./corner_samples_from_samples_{args.SURVEY.upper()}.h5', 'w') as hf_samples:

        data = []

        for gal, dat in filtered.items():
        # gal = 'galaxy_36' # since this is common for LSST and ZTF
        # dat = filtered[gal]
        # event_list = galaxy_to_events[gal]

            event_list = galaxy_to_events[gal]
            print(f"\n[GALAXY] {gal} → {len(event_list)} detected events")

            galaxy_group = hf_out.create_group(gal)

            z_gal = float(dat["z_gal"])
            ra = float(dat["ra"])
            dec = float(dat["dec"])
            mbh6 = float(dat["lgMBH_mass"] - 6)
            stellar_mass = float(dat["star_mass"])

            alpha = dat["alpha"]
            beta = dat["beta"]
            eta = dat["eta"]
            
            print(f"Parameters: z={z_gal:.4g}, mbh6={mbh6:.4g}, stellar_mass={stellar_mass:.4g}, alpha")

            for eidx in event_list:
                catalogue = {}
                print(covariances[gal][eidx])
                z_gal_new, mbh6_new, stellar_mass_new, eta_new, alpha_new, beta_new = multivariate_normal(mean=[z_gal, mbh6, stellar_mass, 0.1, 0.1, 0.9], cov=covariances[gal][eidx], allow_singular=True).rvs()
                
                print(f"\n[EVENT] Galaxy {gal} event {eidx}: computing Fisher matrix")

                theta0_new = {
                    "redshift": z_gal_new,
                    "mbh_6": mbh6_new,
                    "stellar_mass": stellar_mass_new,
                    "eta": 0.1, #float(eta[i]),
                    "alpha": 0.1, #float(alpha[i]),
                    "beta": 0.9, #float(beta[i]),
                    "t0_mjd_transient": t0_mjd_transient,
                    "t0": t0_mjd_transient,
                    "ra": ra,
                    "dec": dec,
                }

                df_ZTF_new = run_telescope_simulator(theta0_new)
                data_by_band = extract_observed_data_by_band(df_ZTF_new)
                
                times_by_band, bands_to_pass = extract_times_by_band(data_by_band)

                NEW_TIME_GRID = np.linspace(0.2, OBSERVING_WINDOW, 400)   # or any resolution
                NEW_TIME_GRID_BY_BAND = {b: NEW_TIME_GRID for b in bands_to_pass}

                print("[INFO] Using " + args.SURVEY.upper() + " bands:", bands_to_pass)

                tasks = [(p, theta0_new, NEW_TIME_GRID_BY_BAND, bands_to_pass) 
                        for p in PARAM_INFO.keys()]

                print("[INFO] Computing partial derivatives...")
                with Pool() as pool:
                    result_list = pool.map(_deriv_worker, tasks)

                # result_list = []
                # for task in tasks:
                #     print(f"  Computing derivative for parameter '{task[0]}'...")
                #     result_list.append(_deriv_worker(task))


                derivs = {p: d for p, d in result_list}

                for p in PARAMS:
                    for b in bands_to_pass:
                        arr = derivs[p][b]
                        if not np.all(np.isfinite(arr)):
                            print(f"[WARN] Non-finite derivative for {p} in band {b}")
                        if np.allclose(arr, 0.0):
                            print(f"[WARN] Zero derivative for {p} in band {b}")


                P = len(PARAMS)
                F = np.zeros((P, P))

                for b in bands_to_pass:
                    J = np.vstack([derivs[p][b] for p in PARAMS])  # shape (P, N)
                    sigma = np.full(J.shape[1], 0.05)
                    W = 1.0 / sigma**2
                    JW = J * W
                    F += JW @ J.T

                F = 0.5 * (F + F.T)
                cov = np.linalg.inv(F)
                errs = np.sqrt(np.diag(cov))

                print("\n[FISHER ERRORS]")
                for p, e in zip(PARAMS, errs):
                    print(f"  {p:15s} = {e:.4g}")

                truths = [theta0_new[p] for p in PARAMS]
                samples = np.random.multivariate_normal(truths, cov, size=1000)
                
                # plot_corner(samples, truths, PARAMS, f"./fisher_TDE_corner_plots/corner_{gal}_event_{eidx}.pdf")

                for i, key in enumerate(PARAMS):
                    catalogue[key] = samples[:, i]
                    print(f"[CATALOGUE] Parameter '{key}': mean={catalogue[key].mean():.4g}, std={catalogue[key].std():.4g}")

                data.append(catalogue)

                event_group = galaxy_group.create_group(f"event_{eidx}")
                event_group.create_dataset("fisher_matrix", data=F)
                event_group.create_dataset("covariance_matrix", data=cov)
                event_group.create_dataset("errors", data=errs)
                event_group.create_dataset("truths", data=truths)

        for i, key in enumerate(PARAMS):
            print(f"[SAVING] '{key}' to HDF5 dataset...")
            data_array = np.array([entry[key] for entry in data])
            hf_samples.create_dataset(key, data=data_array, compression="gzip")
        

    print("\n[DONE] Fisher matrices computed for all detected events.")
    