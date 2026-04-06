import logging

logging.getLogger('bilby').disabled = True
logging.getLogger('redback').disabled = True

import argparse
import numpy as np
import pandas as pd
import h5py
from multiprocessing import Pool
import redback
from redback.simulate_transients import SimulateOpticalTransient

parser = argparse.ArgumentParser(description="Calculate SNR for TDE events.")
parser.add_argument("--OBSERVING_WINDOW", type=float, required=True)
parser.add_argument("--MIN_DETECTIONS", type=int, required=True)
args = parser.parse_args()

OBSERVING_WINDOW = args.OBSERVING_WINDOW
MIN_DETECTIONS = args.MIN_DETECTIONS

MODEL_NAME = "cooling_envelope"
BANDS = ("ztfg", "ztfr", "ztfi")

def run_telescope_simulator(theta):
    """Wrapper for ZTF transient simulation."""
    sim = SimulateOpticalTransient.simulate_transient_in_ztf(
        model=MODEL_NAME,
        survey="ztf",
        parameters=theta,
        model_kwargs={},
        end_transient_time=OBSERVING_WINDOW,
        snr_threshold=5,
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
        "t0_mjd_transient": 58288,
        "t0": 58288,
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

    with h5py.File('./DATA/all_galaxies_TDE_SNR_results.h5', 'w') as hf_out:
        for gal, dat in data.items():
            all_results[gal] = process_one_galaxy(gal, dat)
            results = all_results[gal]
            if any(r['detected_bands'] for r in results):
                group = hf_out.create_group(gal)
                for r in results:
                    event_index = r["event_index"]
                    detected_bands = r["detected_bands"]
                    times_by_band = r["times_by_band"]

                    event_group = group.create_group(f"event_{event_index}")
                    event_group.attrs["detected_bands"] = detected_bands
                    for band in detected_bands:
                        event_group.create_dataset(f"{band}_times", data=times_by_band[band])
                print(f"[{gal}] Detected {len([r for r in results if r['detected_bands']])} events in bands: {', '.join(set(b for r in results for b in r['detected_bands']))}")
            else:
                print(f"[{gal}] No detections in any band.")

    print("\n[DONE] All galaxies processed.")
