"""Simulate survey observations of TDEs with redback and label each as detected or not."""
import logging
import os

logging.getLogger('bilby').disabled = True
logging.getLogger('redback').disabled = True

import argparse
import numpy as np
import h5py
from multiprocessing import Pool
from redback.simulate_transients import SimulateOpticalTransient

# bpl_cooling_envelope and gaussianrise_cooling_envelope are excluded: both
# crash deterministically (KeyError: 'bands') under redback's
# SimulateOpticalTransient (confirmed still broken in redback 1.18.0).
MODEL_CHOICES = [
    "cooling_envelope",
    "stream_stream_tde",
    "tde_analytical",
    "tde_fallback",
]

# Same per-model dispatch tables as calculate_SNR_TDE_PARALLEL.py -- keep in sync.
SHARED_PARAM_MAP = {
    "cooling_envelope":              {"mbh_key": "mbh_6", "mass_key": "stellar_mass", "eta": True,  "alpha_beta": True},
    "stream_stream_tde":             {"mbh_key": "mbh_6", "mass_key": "mstar",        "eta": False, "alpha_beta": False},
    "tde_analytical":                {"mbh_key": None,    "mass_key": None,           "eta": False, "alpha_beta": False},
    "tde_fallback":                  {"mbh_key": "mbh6",  "mass_key": "mstar",        "eta": True,  "alpha_beta": False},
}

EXTRA_PARAM_KEYS = {
    "cooling_envelope":              [],
    "stream_stream_tde":             ["peak_time", "sigma_t", "c1", "del_omega", "f", "h_r", "inc_tcool"],
    "tde_analytical":                ["mej", "vej", "kappa", "kappa_gamma", "temperature_floor", "l0", "t_0_turn"],
    "tde_fallback":                  ["tvisc", "bb", "rph0", "lphoto", "leddlimit"],
}

STELLAR_MASS_FIXED = 1.0  # Msun, fixed for every model that has stellar_mass/mstar

parser = argparse.ArgumentParser(description="Calculate SNR for TDE events.")

parser.add_argument("--events", type=int, required=True, help="total events")
parser.add_argument("--LENGTH_OF_TDE", type=float, required=True)
parser.add_argument("--MIN_DETECTIONS", type=int, required=True)
parser.add_argument("--BANDS", nargs="+", default=["ztfg", "ztfr", "ztfi"], help="Bands to check for detections (e.g., 'ztfg ztfr ztfi' for ZTF)")
parser.add_argument("--SURVEY", type=str, default="ztf", help="Survey to simulate (e.g., 'ztf' or 'lsst')")
parser.add_argument("--Z_MIN", type=float, default=0.0, help="Minimum redshift")
parser.add_argument("--Z_MAX", type=float, default=2.0, help="Maximum redshift")
parser.add_argument("--model", type=str, required=True, choices=MODEL_CHOICES)

args = parser.parse_args()

LENGTH_OF_TDE = args.LENGTH_OF_TDE
MIN_DETECTIONS = args.MIN_DETECTIONS

SURVEY = args.SURVEY
BANDS = args.BANDS
MODEL_NAME = args.model

output_dir = f'./data/PRE_TRAIN_DATA_TDE/1E{np.log10(args.events):.0f}_events/'

if SURVEY == "ZTF":
    SIMULATE_FN = SimulateOpticalTransient.simulate_transient_in_ztf
    SURVEY = "ztf"
elif SURVEY == "LSST":
    SIMULATE_FN = SimulateOpticalTransient.simulate_transient_in_rubin
    SURVEY = "Rubin_10yr_baseline"
else:
    raise ValueError(f"Unsupported survey: {SURVEY}")

SHARED_KEYS = ["lgMBH", "z_gal", "ra_rad", "dec_rad", "eta", "alpha", "beta", "t0_mjd_transient"]
EXTRA_KEYS = EXTRA_PARAM_KEYS[MODEL_NAME]
MAP = SHARED_PARAM_MAP[MODEL_NAME]

def run_telescope_simulator(theta):

    sim = SIMULATE_FN(
        model=MODEL_NAME,
        survey=SURVEY,
        parameters=theta,
        model_kwargs={},
        end_transient_time=LENGTH_OF_TDE,
        snr_threshold=5.0,
        add_source_noise=False
    )

    cols = ['time (days)', 'magnitude', 'e_magnitude', 'band', 'detected']
    return sim.observations[cols]

def extract_observed_data_by_band(df):
    """Vectorised band-group extraction."""
    grouped = df.groupby("band")

    return {
        band: {
            "time": g["time (days)"].to_numpy(float),
            "magnitude": g["magnitude"].to_numpy(float),
            "e_magnitude": g["e_magnitude"].to_numpy(float),
            "detected": g["detected"].to_numpy(bool),
        }
        for band, g in grouped
        if band in BANDS
    }

def if_detected_per_band(data_by_band):
    """Check detection threshold per band."""
    detected_bands = [
        b for b, d in data_by_band.items()
        if d["detected"].sum() >= MIN_DETECTIONS
    ]

    times_by_band = {b: data_by_band[b]["time"] for b in detected_bands}
    return times_by_band, detected_bands

def build_theta(theta_dict, i):
    """Build the per-event parameter dict for MODEL_NAME using only the
    kwargs that model's redback function signature actually accepts.
    Keep in sync with calculate_SNR_TDE_PARALLEL.py's build_theta()."""
    lgMBH = float(theta_dict["lgMBH"][i])
    theta = {
        "redshift":          float(theta_dict["z_gal"][i]),
        "t0_mjd_transient":  float(theta_dict["t0_mjd_transient"][i]),
        "t0":                float(theta_dict["t0_mjd_transient"][i]),
        "ra":                float(theta_dict["ra_rad"][i]),
        "dec":               float(theta_dict["dec_rad"][i]),
    }
    # mbh_6/mbh6 is the BH mass in units of 1e6 Msun; lgMBH is log10(M_BH/Msun),
    # so the correct unit conversion is 10**(lgMBH-6). The previous
    # "mbh_6 = lgMBH - 6" (a log-difference) produced NEGATIVE/near-zero BH
    # masses for most of the sampled lgMBH range [4,7) -- confirmed to
    # generate NaN/complex garbage inside stream_stream_tde's internal
    # physics (sqrt of a negative quantity), which is what crashed the SNR
    # calc for that model. Fixed here for every model that takes mbh_6/mbh6,
    # including tde_fallback's mbh6.
    if MAP["mbh_key"] is not None:
        theta[MAP["mbh_key"]] = 10 ** (lgMBH - 6)
    if MAP["mass_key"] is not None:
        theta[MAP["mass_key"]] = STELLAR_MASS_FIXED
    if MAP["eta"]:
        theta["eta"] = float(theta_dict["eta"][i])
    if MAP["alpha_beta"]:
        theta["alpha"] = float(theta_dict["alpha"][i])
        theta["beta"] = float(theta_dict["beta"][i])
    for key in EXTRA_KEYS:
        theta[key] = float(theta_dict[key][i])
    return theta

def process_events(theta_dict):
    """
    Process all events and return binary detectability (0 or 1) for each event.
    """
    n_events = len(theta_dict["z_gal"])

    thetas = [build_theta(theta_dict, i) for i in range(n_events)]

    print(f"[INFO] Running {len(thetas)} transient simulations for model '{MODEL_NAME}' (batch processing)...")

    # Process in batches to reduce memory usage
    batch_size = 100
    sims = []
    for batch_start in range(0, len(thetas), batch_size):
        batch_end = min(batch_start + batch_size, len(thetas))
        batch = thetas[batch_start:batch_end]

        print(f"[INFO] Processing batch {batch_start//batch_size + 1}/{(len(thetas) + batch_size - 1)//batch_size} ({batch_start}-{batch_end})")

        with Pool(processes=12) as pool:
            batch_sims = pool.map(run_telescope_simulator, batch)

        sims.extend(batch_sims)

        # Explicit cleanup
        del batch_sims, batch
        import gc
        gc.collect()

    # Determine detectability (0 or 1) for each event
    detectability = np.zeros(n_events, dtype=int)
    for i, df in enumerate(sims):
        if len(df) == 0:
            detectability[i] = 0
            continue

        data_by_band = extract_observed_data_by_band(df)
        times_by_band, detected_bands = if_detected_per_band(data_by_band)

        # Event is detected if it has detected bands
        if len(detected_bands) > 0:
            detectability[i] = 1
        else:
            detectability[i] = 0

    return detectability


if __name__ == "__main__":

    print("[INFO] Loading HDF5 file...")
    h5_path = f'{output_dir}/TDE_params_1E{np.log10(args.events):.0f}_{args.SURVEY}_{MODEL_NAME}.h5'
    with h5py.File(h5_path, 'r') as hf:
        theta = {k: hf[k][:] for k in SHARED_KEYS + EXTRA_KEYS}

    lgMBH = theta["lgMBH"]
    z_gal = theta["z_gal"]

    n_total_events = len(lgMBH)
    print(f"[IMPORTANT SURVEY INFO] Total events: {n_total_events}")
    print(f"[IMPORTANT SURVEY INFO] Model: {MODEL_NAME}")
    print(f"[IMPORTANT SURVEY INFO] Checking detectability with {args.SURVEY.upper()}...")

    # Apply redshift filter
    mask = (z_gal >= args.Z_MIN) & (z_gal <= args.Z_MAX)
    n_filtered_events = mask.sum()
    print(f"[INFO] {n_filtered_events} out of {n_total_events} events are within the redshift range [{args.Z_MIN}, {args.Z_MAX}].")

    theta_filtered = {k: v[mask] for k, v in theta.items()}

    # Process filtered events and get binary detectability
    detectability_filtered = process_events(theta_filtered)

    # Create full detectability array with zeros for filtered-out events
    detectability_full = np.zeros(n_total_events, dtype=int)
    detectability_full[mask] = detectability_filtered

    print(f"[SUMMARY] {detectability_full.sum()} out of {n_total_events} events are detectable with {args.SURVEY.upper()} (model={MODEL_NAME}).")

    # Save output
    output_file = f'{output_dir}/TDE_detectability_1E{np.log10(args.events):.0f}_{args.SURVEY}_{MODEL_NAME}.h5'
    print(f"[INFO] Saving detectability labels to {output_file}...")
    with h5py.File(output_file, 'w') as hf:
        hf.create_dataset("detectability", data=detectability_full)

    np.save(f"{output_dir}/TDE_detectability_1E{np.log10(args.events):.0f}_{args.SURVEY}_{MODEL_NAME}.npy", detectability_full)

    print("[DONE] All events processed and saved.")


    # plotting the distribution of detectable vs non-detectable events (shared params only)
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    det_idx = detectability_full == 1

    params = [
        ('lgMBH',           axs[0,0]),
        ('z_gal',           axs[0,1]),
        ('ra_rad',          axs[0,2]),
        ('dec_rad',         axs[1,0]),
        ('eta',             axs[1,1]),
        ('alpha',           axs[1,2]),
        ('beta',            axs[2,0]),
        ('t0_mjd_transient',axs[2,1]),
    ]

    for key, ax in params:
        ax.hist(theta[key], bins='auto', density=True,
                histtype='step', color='blue', linestyle='-',  linewidth=2, label='All events')
        ax.hist(theta[key][det_idx], bins='auto', density=True,
                histtype='step', color='red',  linestyle='--', linewidth=2, label='Detectable events')
        ax.set_title(key)
        ax.set_ylabel('PDF')

    axs[0,0].legend()
    axs[2,2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,
        f"TDE_params_detectability_1E{int(np.log10(args.events))}_{args.SURVEY}_{MODEL_NAME}.png"))
    plt.close()
