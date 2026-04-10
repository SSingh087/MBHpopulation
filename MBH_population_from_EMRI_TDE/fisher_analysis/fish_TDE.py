import numpy as np
import logging
logging.getLogger('bilby').disabled = True
logging.getLogger('redback').disabled = True

from redback.simulate_transients import SimulateGenericTransient, SimulateOpticalTransient

# Parameter info
PARAM_INFO = {
    "redshift":     {"delta": 1e-7, "type": "linear"},
    "mbh_6":        {"delta": 1e-7, "type": "linear"},
    "stellar_mass": {"delta": 1e-7, "type": "linear"},
    "eta":          {"delta": 5e-6, "type": "log10"},
    "alpha":        {"delta": 1e-3, "type": "log10"},
    "beta":         {"delta": 1e-1, "type": "linear"},
}

def choose_simulator(survey):
    su = survey.upper()
    if su == "ZTF":
        return SimulateOpticalTransient.simulate_transient_in_ztf, "ztf", 58288
    elif su == "LSST":
        return SimulateOpticalTransient.simulate_transient_in_rubin, "Rubin_10yr_baseline", 60000
    else:
        raise ValueError(f"Unsupported survey {survey}")

def run_telescope_simulator(theta, simulate_fn, survey, observing_window):
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
        b: {
            "time": g["time (days)"].to_numpy(float),
            "magnitude": g["magnitude"].to_numpy(float),
            "err": g["e_magnitude"].to_numpy(float),
        }
        for b, g in grouped if b in bands
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

def compute_fisher(theta0, times_by_band, bands):
    params = list(PARAM_INFO.keys())
    derivs = {}

    # sequential derivatives (stable)
    for p in params:
        # print(f"  Derivative: {p}")
        derivs[p] = numerical_derivative(p, theta0, times_by_band, bands)

    P = len(params)
    F = np.zeros((P, P))

    for b in bands:
        J = np.vstack([derivs[p][b] for p in params])
        sigma = np.full(J.shape[1], 0.05)
        W = 1.0 / sigma**2
        JW = J * W
        F += JW @ J.T

    F = 0.5 * (F + F.T)
    cov = np.linalg.inv(F + 1e-12 * np.eye(P))  # small reg

    return F, cov, derivs