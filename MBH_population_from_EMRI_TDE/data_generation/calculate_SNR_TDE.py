import logging

logging.getLogger('bilby').disabled = True
logging.getLogger('redback').disabled = True

import argparse
import numpy as np
import h5py
from multiprocessing import Pool
from redback.simulate_transients import SimulateOpticalTransient

parser = argparse.ArgumentParser(description="Calculate SNR for TDE events.")
parser.add_argument("--OBSERVING_WINDOW", type=float, required=True)
parser.add_argument("--MIN_DETECTIONS", type=int, required=True)
parser.add_argument("--BANDS", nargs="+", default=["ztfg", "ztfr", "ztfi"], help="Bands to check for detections (e.g., 'ztfg ztfr ztfi' for ZTF)")
parser.add_argument("--SURVEY", type=str, default="ztf", help="Survey to simulate (e.g., 'ztf' or 'lsst')")

args = parser.parse_args()

OBSERVING_WINDOW = args.OBSERVING_WINDOW
MIN_DETECTIONS = args.MIN_DETECTIONS

SURVEY = args.SURVEY
BANDS = args.BANDS

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

def run_telescope_simulator(theta):

    sim = SIMULATE_FN(
        model=MODEL_NAME,
        survey=SURVEY,
        parameters=theta,
        model_kwargs={},
        end_transient_time=OBSERVING_WINDOW,
        snr_threshold=10.0,
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

def process_one_galaxy(gal, dat):
    print(f"\n[GALAXY] Processing {gal} at z={dat['z_gal']} with lgMBH={dat['lgMBH_mass']} and star_mass={dat['star_mass']}")

    z_gal = float(dat["z_gal"])
    ra = float(dat["ra"])
    dec = float(dat["dec"])
    mbh6 = float(dat["lgMBH_mass"] - 6)
    stellar_mass = float(dat["star_mass"])

    alpha = dat["alpha"]
    beta = dat["beta"]
    eta = dat["eta"]

    thetas = [{
        "redshift": z_gal,
        "mbh_6": mbh6,
        "stellar_mass": stellar_mass,
        "eta": 0.1, #float(eta[i]),
        "alpha": 0.1, #float(alpha[i]),
        "beta": 0.9, #float(beta[i]),
        "t0_mjd_transient": t0_mjd_transient,
        "t0": t0_mjd_transient,
        "ra": ra,
        "dec": dec,
    } for i in range(len(alpha))]

    print(f"[INFO] Running {len(thetas)} transient simulations...")
    with Pool() as pool:
        sims = pool.map(run_telescope_simulator, thetas)

    results = []
    for i, df in enumerate(sims):
        data_by_band = extract_observed_data_by_band(df)
        times_by_band, detected_bands = if_detected_per_band(data_by_band)

        results.append({
            "event_index": i,
            "detected_bands": detected_bands,
            "times_by_band": times_by_band
        })
    return results


if __name__ == "__main__":
    
    print("[INFO] Loading HDF5 file...")
    with h5py.File('./DATA/all_galaxies_TDE_events.h5', 'r') as hf:
        data = {
            gal: {k: np.array(hf[gal][k]) for k in hf[gal].keys()}
            for gal in hf.keys()
        }

    hf.close()

    all_results = {}
    count = 0
    print(f"[IMPORTANT SURVEY INFO] Checking detectability with {args.SURVEY.upper()} for {len(data)} galaxies...")
    with h5py.File(f'./DATA/all_galaxies_TDE_SNR_results_{args.SURVEY.upper()}.h5', 'w') as hf_out:

        for gal, dat in data.items():
            if data[gal]['z_gal'] > 2.0:
                print(f"[WARNING] Skipping {gal} due to high redshift (z={data[gal]['z_gal']})")
                continue

            print(f"\n[INFO] Starting processing for {gal} with z={dat['z_gal']}")
            results = process_one_galaxy(gal, dat)
            detected_events = [r["event_index"] for r in results if r["detected_bands"]]
            count += len(detected_events)

            if len(detected_events) == 0:
                print(f"[{gal}] No detections in any event.\n")
                continue
            
            gal_group = hf_out.create_group(gal)
            gal_group.create_dataset("event_index", data=np.array(detected_events, dtype=int))

            print(f"[{gal}] Saved detected event indices: {detected_events}")

    print("\n[DONE] All galaxies processed.")
    print(f"[SUMMARY] Total detected events for {SURVEY.upper()}: {count}")