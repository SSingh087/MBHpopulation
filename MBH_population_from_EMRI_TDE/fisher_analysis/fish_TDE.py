import numpy as np
from multiprocessing import Pool
import logging
import matplotlib.pyplot as plt
import corner
import os

logging.getLogger('bilby').disabled = True
logging.getLogger('redback').disabled = True

from redback.simulate_transients import SimulateGenericTransient, SimulateOpticalTransient



PARAM_INFO = {
    "redshift":     {"delta": 1e-7, "type": "linear"},
    "mbh_6":        {"delta": 1e-7, "type": "linear"},
    "stellar_mass": {"delta": 1e-7, "type": "linear"},
    "eta":          {"delta": 5e-6, "type": "log10"},
    "alpha":        {"delta": 1e-3, "type": "log10"},
    "beta":         {"delta": 1e-1, "type": "linear"},
}


def choose_simulator(survey):
    """Return survey-specific simulator & reference MJD."""
    survey_up = survey.upper()
    if survey_up == "ZTF":
        return SimulateOpticalTransient.simulate_transient_in_ztf, "ztf", 58288
    elif survey_up == "LSST":
        return SimulateOpticalTransient.simulate_transient_in_rubin, "Rubin_10yr_baseline", 60000
    else:
        raise ValueError(f"Unsupported survey: {survey}")


def run_telescope_simulator(theta, simulate_fn, survey, observing_window):
    """Simulate telescope light curve."""
    sim = simulate_fn(
        model="cooling_envelope",
        survey=survey,
        parameters=theta,
        model_kwargs={},
        end_transient_time=observing_window,
        snr_threshold=5.0,
        add_source_noise=False
    )
    cols = ['time (days)', 'magnitude', 'e_magnitude', 'band', 'detected']
    return sim.observations[cols]


def extract_observed_data_by_band(df, bands):
    grouped = df.groupby("band")
    return {
        band: {
            "time": g["time (days)"].to_numpy(float),
            "magnitude": g["magnitude"].to_numpy(float),
            "err": g["e_magnitude"].to_numpy(float),
        }
        for band, g in grouped if band in bands
    }


def extract_times_by_band(data_by_band):
    return {b: d["time"] for b, d in data_by_band.items()}, list(data_by_band.keys())


def run_generic_for_band(theta, band, times):
    obs = SimulateGenericTransient(
        model="cooling_envelope",
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
    print(f"Parameter '{param}': val_p={val_p:.4g}, val_m={val_m:.4g}")
    # breakpoint()
    
    deriv = {}
    if kind == "linear":
        for b in bands:
            deriv[b] = (m_p[b] - m_m[b]) / (2 * delta)
    else:
        factor = 1.0 / (theta_base[param] * np.log(10.0))
        for b in bands:
            dm_du = (m_p[b] - m_m[b]) / (2 * delta)
            deriv[b] = dm_du * factor

    return deriv


def _deriv_worker(args):
    p, theta0, times_by_band, bands = args
    return p, numerical_derivative(p, theta0, times_by_band, bands)


def compute_fisher(theta0, times_by_band, bands):
    params = list(PARAM_INFO.keys())
    tasks = [(p, theta0, times_by_band, bands) for p in params]

    # with Pool() as pool:
    #     result_list = pool.map(_deriv_worker, tasks)

    result_list = []
    for task in tasks:
        print(f"  Computing derivative for parameter '{task[0]}'...")
        result_list.append(_deriv_worker(task))


    derivs = {p: d for p, d in result_list}

    P = len(params)
    F = np.zeros((P, P))

    for b in bands:
        J = np.vstack([derivs[p][b] for p in params])
        sigma = np.full(J.shape[1], 0.05)
        W = 1.0 / sigma**2
        JW = J * W
        F += JW @ J.T

    F = 0.5 * (F + F.T)
    cov = np.linalg.inv(F)

    return F, cov, derivs


def plot_corner(samples, truths, params, filename, title=None):
    """
    Generate a corner plot of Fisher-sampled parameters.

    Args:
        samples (ndarray): shape (Nsamples, Nparams)
        truths (list): true parameter values (length Nparams)
        params (list): parameter names (length Nparams)
        filename (str): where to save the PDF/PNG
        title (str): optional figure title
    """
    fig = corner.corner(
        samples,
        labels=params,
        truths=truths,
        color="navy",
        truth_color="red",
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".4g",
        hist_kwargs={"density": True, "color": "black"},
    )

    if title:
        fig.suptitle(title, fontsize=14)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_fisher_matrix(F, params, filename, title="Fisher Matrix"):
    """
    Plot a heatmap of the Fisher matrix.

    Args:
        F (ndarray): (N,N) Fisher matrix
        params (list): parameter labels
        filename (str): save location
        title (str): plot title
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(F, cmap="viridis")
    plt.colorbar(label="Fisher Information")
    plt.xticks(np.arange(len(params)), params, rotation=45)
    plt.yticks(np.arange(len(params)), params)
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_covariance_matrix(C, params, filename, title="Covariance Matrix"):
    """
    Plot a heatmap of the covariance matrix.

    Args:
        C (ndarray): (N,N) covariance matrix
        params (list): parameter labels
        filename (str): save location
        title (str): plot title
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(C, cmap="magma")
    plt.colorbar(label="Covariance")
    plt.xticks(np.arange(len(params)), params, rotation=45)
    plt.yticks(np.arange(len(params)), params)
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_parameter_histograms(samples, params, filename, bins=30):
    """
    Quick 1D histogram for each parameter.

    Args:
        samples (ndarray): (N, P) sample array
        params (list): length-P param names
        filename (str): where to save
        bins (int): histogram bins
    """
    P = samples.shape[1]
    fig, axes = plt.subplots(P, 1, figsize=(6, 2*P), sharex=False)

    if P == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.hist(samples[:, i], bins=bins, color="gray", alpha=0.8)
        ax.set_title(params[i])
        ax.set_ylabel("Count")

    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=200)
    plt.close()