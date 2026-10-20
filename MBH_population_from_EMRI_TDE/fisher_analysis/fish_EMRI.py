"""Fisher-matrix helpers for EMRIs: LISA response (FEW AAK + fastlisaresponse), numerical derivatives and the Fisher matrix."""
import numpy as np

# Waveform arguments in the order expected by the response wrapper (M, mu, a, p0, e0, Y0, dist, ...).
# `delta` is the relative step of the central difference; `scale` maps the catalogue column to the argument.
PARAM_INFO = {
    "lgMBH_mass":  {"delta": 1e-6, "kind": "log10"},    # M = 10**lgMBH_mass
    "CO_mass":     {"delta": 1e-6, "kind": "linear"},   # mu [Msun]
    "MBHspin":     {"delta": 1e-6, "kind": "linear"},   # a
    "p0":          {"delta": 1e-6, "kind": "linear"},
    "e0":          {"delta": 1e-6, "kind": "linear"},
    "Y0":          {"delta": 1e-6, "kind": "linear"},
    "distance_Gpc": {"delta": 1e-6, "kind": "linear"},
    "qS":          {"delta": 1e-6, "kind": "linear"},
    "phiS":        {"delta": 1e-6, "kind": "linear"},
    "qK":          {"delta": 1e-6, "kind": "linear"},
    "phiK":        {"delta": 1e-6, "kind": "linear"},
    "Phi_phi0":    {"delta": 1e-6, "kind": "linear"},
    "Phi_theta0":  {"delta": 1e-6, "kind": "linear"},
    "Phi_r0":      {"delta": 1e-6, "kind": "linear"},
}
PARAMS = list(PARAM_INFO.keys())

# Prior bounds used to truncate the Gaussian posteriors (same ranges as the injections)
BOUNDS = {
    "lgMBH_mass": (4.0, 10.0), "CO_mass": (1.0, 100.0), "MBHspin": (0.0, 0.99),
    "p0": (0.0, np.inf), "e0": (0.0, 0.8), "Y0": (-1.0, 1.0), "distance_Gpc": (0.0, np.inf),
    "qS": (0.0, np.pi), "phiS": (0.0, 2 * np.pi), "qK": (0.0, np.pi), "phiK": (0.0, 2 * np.pi),
    "Phi_phi0": (0.0, 2 * np.pi), "Phi_theta0": (0.0, 2 * np.pi), "Phi_r0": (0.0, 2 * np.pi),
}


def build_response(observing_window, dt=10.0, use_gpu=True):
    """LISA TDI (A, E) response for the 5PN AAK waveform."""
    from few.waveform import GenerateEMRIWaveform
    from fastlisaresponse import ResponseWrapper

    gen_wave = GenerateEMRIWaveform(
        "Pn5AAKWaveform",
        sum_kwargs={"use_gpu": use_gpu, "pad_output": True},
        inspiral_kwargs={"max_init_len": int(1e7)},
    )
    tdi_kwargs = dict(orbit_kwargs=dict(orbit_file="./data/orbit_files/esa-trailing-orbits.h5"),
                      order=25, tdi="1st generation", tdi_chan="AET")
    return ResponseWrapper(gen_wave, observing_window, dt, 8, 7, t0=10000.0, flip_hx=True,
                           remove_sky_coords=False, is_ecliptic_latitude=False,
                           remove_garbage=True, use_gpu=use_gpu, **tdi_kwargs)


def to_waveform_args(theta):
    """Catalogue-style dict -> positional waveform arguments."""
    return (10 ** theta["lgMBH_mass"], theta["CO_mass"], theta["MBHspin"], theta["p0"], theta["e0"],
            theta["Y0"], theta["distance_Gpc"], theta["qS"], theta["phiS"], theta["qK"], theta["phiK"],
            theta["Phi_phi0"], theta["Phi_theta0"], theta["Phi_r0"])


def to_frequency_domain(response, theta, dt):
    """(A, E) frequency-domain strain of the injection, DC bin removed."""
    import cupy as cp
    aet = response(*to_waveform_args(theta))
    ae = cp.asarray(aet[0:2])
    return cp.fft.rfft(ae, axis=1)[:, 1:] * dt


def inner_product(a_fd, b_fd, psd, df):
    """Noise-weighted inner product (a|b) summed over the A and E channels."""
    return float(4 * df * (a_fd.conj() * b_fd / psd).real.sum())


def numerical_derivative(response, theta, name, dt):
    """Central-difference derivative of the frequency-domain waveform wrt one parameter."""
    info = PARAM_INFO[name]
    step = info["delta"] * max(abs(theta[name]), 1.0)
    up, dn = dict(theta), dict(theta)
    up[name] += step
    dn[name] -= step
    return (to_frequency_domain(response, up, dt) - to_frequency_domain(response, dn, dt)) / (2 * step)


def compute_fisher(response, theta, psd, dt):
    """Fisher matrix over PARAMS at theta."""
    derivs = [numerical_derivative(response, theta, p, dt) for p in PARAMS]
    if derivs[0].shape[1] != psd.shape[-1]:
        raise ValueError(f"PSD has {psd.shape[-1]} frequency bins but the waveform has {derivs[0].shape[1]}: "
                         f"generate the PSD for the same observing window and dt (training_EMRI_SNR/calculate_PSD.py)")
    df = 1.0 / (derivs[0].shape[1] * 2 * dt)
    n = len(PARAMS)
    F = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            F[i, j] = F[j, i] = inner_product(derivs[i], derivs[j], psd, df)
    return F


def covariance(F, rcond=1e-12):
    """Covariance from the Fisher matrix (pseudo-inverse for near-singular directions)."""
    return np.linalg.pinv(F, rcond=rcond)


def sample_truncated_gaussian(mean, cov, n_target, rng, n_draw=20000):
    """Draw from N(mean, cov) truncated to BOUNDS."""
    lo = np.array([BOUNDS[p][0] for p in PARAMS])
    hi = np.array([BOUNDS[p][1] for p in PARAMS])
    draws = rng.multivariate_normal(mean, cov, size=n_draw)
    ok = draws[((draws >= lo) & (draws <= hi)).all(axis=1)]
    if len(ok) == 0:
        raise RuntimeError("no posterior samples inside the prior bounds")
    return ok[rng.choice(len(ok), size=n_target, replace=len(ok) < n_target)]


def plunge_eccentricity(traj, samples, T_years, dt):
    """Eccentricity at plunge for each (lgMBH_mass, CO_mass, MBHspin, p0, e0, Y0) sample."""
    e_pl = np.empty(len(samples))
    for k, s in enumerate(samples):
        th = dict(zip(PARAMS, s))
        _, _, e, *_ = traj(10 ** th["lgMBH_mass"], th["CO_mass"], th["MBHspin"], th["p0"], th["e0"],
                           th["Y0"], Phi_phi0=0.0, Phi_theta0=0.0, Phi_r0=0.0, dt=dt, T=T_years)
        e_pl[k] = e[-1]
    return e_pl
