import logging

logging.getLogger('bilby').disabled = True
logging.getLogger('redback').disabled = True

import argparse
import numpy as np
import h5py
from multiprocessing import Pool
from redback.simulate_transients import SimulateOpticalTransient

parser = argparse.ArgumentParser(description="Calculate SNR for TDE events.")

parser.add_argument("--events", type=int, required=True, help="total events")
parser.add_argument("--OBSERVING_WINDOW", type=float, required=True)
parser.add_argument("--MIN_DETECTIONS", type=int, required=True)
parser.add_argument("--BANDS", nargs="+", default=["ztfg", "ztfr", "ztfi"], help="Bands to check for detections (e.g., 'ztfg ztfr ztfi' for ZTF)")
parser.add_argument("--SURVEY", type=str, default="ztf", help="Survey to simulate (e.g., 'ztf' or 'lsst')")
parser.add_argument("--Z_MIN", type=float, default=0.0, help="Minimum redshift")
parser.add_argument("--Z_MAX", type=float, default=2.0, help="Maximum redshift")

args = parser.parse_args()

OBSERVING_WINDOW = args.OBSERVING_WINDOW
MIN_DETECTIONS = args.MIN_DETECTIONS

SURVEY = args.SURVEY
BANDS = args.BANDS


output_dir = f'./data/PRE_TRAIN_DATA_TDE/1E{np.log10(args.events):.0f}_events/'

if SURVEY == "ZTF":
    SIMULATE_FN = SimulateOpticalTransient.simulate_transient_in_ztf
    SURVEY = "ztf"
elif SURVEY == "LSST":
    SIMULATE_FN = SimulateOpticalTransient.simulate_transient_in_rubin
    SURVEY = "Rubin_10yr_baseline"
else:
    raise ValueError(f"Unsupported survey: {SURVEY}")

MODEL_NAME = "cooling_envelope"

def run_telescope_simulator(theta):

    sim = SIMULATE_FN(
        model=MODEL_NAME,
        survey=SURVEY,
        parameters=theta,
        model_kwargs={},
        end_transient_time=OBSERVING_WINDOW,
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

def process_events(theta_dict):
    """
    Process all events and return binary detectability (0 or 1) for each event.
    """
    n_events = len(theta_dict["z_gal"])
    
    # Build parameter dictionaries for each event
    thetas = []
    
    if SURVEY == "ztf":
        # ZTF-safe window
        t0_mjd_transient = np.random.uniform(58200, 59550, size=n_events)
    elif SURVEY == "Rubin_10yr_baseline":
        # Rubin 10yr baseline, with buffers
        t0_mjd_transient = np.random.uniform(60100, 63500, size=n_events)

    for i in range(n_events):
        z_gal = float(theta_dict["z_gal"][i])
        ra_deg = float(theta_dict["ra_deg"][i])
        dec_deg = float(theta_dict["dec_deg"][i])
        lgMBH = float(theta_dict["lgMBH"][i])
        mbh_6 = lgMBH - 6
        eta = float(theta_dict["eta"][i])
        alpha = float(theta_dict["alpha"][i])
        beta = float(theta_dict["beta"][i])

        thetas.append({
            "redshift": z_gal,
            "mbh_6": mbh_6,
            "stellar_mass": 1.0,  # Fixed stellar mass
            "eta": eta,
            "alpha": alpha,
            "beta": beta,
            "t0_mjd_transient": t0_mjd_transient[i],
            "t0": t0_mjd_transient[i],
            "ra": ra_deg,
            "dec": dec_deg,
        })
    
    print(f"[INFO] Running {len(thetas)} transient simulations...")
    with Pool(processes=16) as pool:
        sims = pool.map(run_telescope_simulator, thetas)
    
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
    with h5py.File(f'{output_dir}/TDE_params_1E{np.log10(args.events):.0f}_{args.SURVEY}.h5', 'r') as hf:
        lgMBH = hf['lgMBH'][:]
        MBHspin = hf['MBHspin'][:]
        z_gal = hf['z_gal'][:]
        ra_deg = hf['ra_deg'][:]
        dec_deg = hf['dec_deg'][:]
        eta = hf['eta'][:]
        alpha = hf['alpha'][:]
        beta = hf['beta'][:]

    theta = {
        "lgMBH": lgMBH,
        "MBHspin": MBHspin,
        "z_gal": z_gal,
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "eta": eta,
        "alpha": alpha,
        "beta": beta
    }

    n_total_events = len(lgMBH)
    print(f"[IMPORTANT SURVEY INFO] Total events: {n_total_events}")
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
    
    print(f"[SUMMARY] {detectability_full.sum()} out of {n_total_events} events are detectable with {args.SURVEY.upper()}.")
    
    # Save output
    output_file = f'{output_dir}/TDE_detectability_1E{np.log10(args.events):.0f}_{args.SURVEY}.h5'
    print(f"[INFO] Saving detectability labels to {output_file}...")
    with h5py.File(output_file, 'w') as hf:
        hf.create_dataset("detectability", data=detectability_full)

    np.save(f"{output_dir}/TDE_detectability_1E{np.log10(args.events):.0f}_{args.SURVEY}.npy", detectability_full)
    
    print("[DONE] All events processed and saved.")