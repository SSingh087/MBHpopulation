import logging
import os

logging.getLogger('bilby').disabled = True
logging.getLogger('redback').disabled = True

import argparse
import numpy as np
import h5py
from multiprocessing import Pool
from redback.simulate_transients import SimulateOpticalTransient

# ── Argument parsing ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Calculate SNR for TDE events.")
parser.add_argument("--events",           type=int,   required=True)
parser.add_argument("--OBSERVING_WINDOW", type=float, required=True)
parser.add_argument("--MIN_DETECTIONS",   type=int,   required=True)
parser.add_argument("--BANDS", nargs="+", default=["ztfg", "ztfr", "ztfi"])
parser.add_argument("--SURVEY",           type=str,   default="ztf")
parser.add_argument("--Z_MIN",            type=float, default=0.0)
parser.add_argument("--Z_MAX",            type=float, default=2.0)
parser.add_argument("--chunk_id",         type=int,   default=0,  help="0-indexed chunk ID")
parser.add_argument("--n_chunks",         type=int,   default=1,  help="Total number of chunks")
args = parser.parse_args()

OBSERVING_WINDOW = args.OBSERVING_WINDOW
MIN_DETECTIONS   = args.MIN_DETECTIONS
BANDS            = args.BANDS
SURVEY           = args.SURVEY   # keep original for output filenames

output_dir = (f"./data/"
              f"PRE_TRAIN_DATA_TDE/1E{np.log10(args.events):.0f}_events/")
os.makedirs(output_dir, exist_ok=True)

# ── Survey setup ───────────────────────────────────────────────────────────
if SURVEY == "ZTF":
    SIMULATE_FN  = SimulateOpticalTransient.simulate_transient_in_ztf
    SURVEY_STR   = "ztf"
elif SURVEY == "LSST":
    SIMULATE_FN  = SimulateOpticalTransient.simulate_transient_in_rubin
    SURVEY_STR   = "Rubin_10yr_baseline"
else:
    raise ValueError(f"Unsupported survey: {SURVEY}")

MODEL_NAME = "cooling_envelope"

# ── Core simulation functions ──────────────────────────────────────────────
def run_telescope_simulator(theta):
    sim = SIMULATE_FN(
        model=MODEL_NAME,
        survey=SURVEY_STR,
        parameters=theta,
        model_kwargs={},
        end_transient_time=OBSERVING_WINDOW,
        snr_threshold=5.0,
        add_source_noise=False
    )
    cols = ['time (days)', 'magnitude', 'e_magnitude', 'band', 'detected']
    return sim.observations[cols]


def extract_observed_data_by_band(df):
    grouped = df.groupby("band")
    return {
        band: {
            "time":        g["time (days)"].to_numpy(float),
            "magnitude":   g["magnitude"].to_numpy(float),
            "e_magnitude": g["e_magnitude"].to_numpy(float),
            "detected":    g["detected"].to_numpy(bool),
        }
        for band, g in grouped
        if band in BANDS
    }


def if_detected_per_band(data_by_band):
    detected_bands = [
        b for b, d in data_by_band.items()
        if d["detected"].sum() >= MIN_DETECTIONS
    ]
    times_by_band = {b: data_by_band[b]["time"] for b in detected_bands}
    return times_by_band, detected_bands


def process_events(theta_dict):
    n_events = len(theta_dict["z_gal"])

    # Build flat list of per-event parameter dicts
    thetas = []
    for i in range(n_events):
        lgMBH = float(theta_dict["lgMBH"][i])
        thetas.append({
            "redshift":          float(theta_dict["z_gal"][i]),
            "mbh_6":             lgMBH - 6,
            "stellar_mass":      1.0,
            "eta":               float(theta_dict["eta"][i]),
            "alpha":             float(theta_dict["alpha"][i]),
            "beta":              float(theta_dict["beta"][i]),
            "t0_mjd_transient":  float(theta_dict["t0_mjd_transient"][i]),
            "t0":                float(theta_dict["t0_mjd_transient"][i]),
            "ra":                float(theta_dict["ra_rad"][i]),
            "dec":               float(theta_dict["dec_rad"][i]),
        })

    print(f"[INFO] Running {n_events} transient simulations...")

    # Batch over Pool to control memory
    batch_size = 100
    sims = []
    n_batches = int(np.ceil(n_events / batch_size))
    for b in range(n_batches):
        s, e = b * batch_size, min((b + 1) * batch_size, n_events)
        print(f"[INFO] Batch {b+1}/{n_batches}  (events {s}–{e})")
        with Pool(processes=12) as pool:
            batch_sims = pool.map(run_telescope_simulator, thetas[s:e])
        sims.extend(batch_sims)
        del batch_sims
        import gc; gc.collect()

    # Binary detectability
    detectability = np.zeros(n_events, dtype=int)
    for i, df in enumerate(sims):
        if len(df) == 0:
            continue
        data_by_band = extract_observed_data_by_band(df)
        _, detected_bands = if_detected_per_band(data_by_band)
        if len(detected_bands) > 0:
            detectability[i] = 1

    return detectability


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── 1. Load full parameter file ────────────────────────────────────────
    h5_path = (f"{output_dir}/"
               f"TDE_params_1E{np.log10(args.events):.0f}_{args.SURVEY}.h5")
    print(f"[INFO] Loading HDF5: {h5_path}")

    with h5py.File(h5_path, "r") as hf:
        theta = {k: hf[k][:] for k in
                 ["lgMBH", "z_gal", "ra_rad", "dec_rad",
                  "eta", "alpha", "beta", "t0_mjd_transient"]}

    n_total = len(theta["z_gal"])
    print(f"[INFO] Total events in file: {n_total}")

    # ── 2. Redshift mask ───────────────────────────────────────────────────
    mask = (theta["z_gal"] >= args.Z_MIN) & (theta["z_gal"] <= args.Z_MAX)
    filtered_indices = np.where(mask)[0]          # global indices that pass filter
    n_filtered = len(filtered_indices)
    print(f"[INFO] {n_filtered}/{n_total} events pass redshift filter "
          f"[{args.Z_MIN}, {args.Z_MAX}]")

    # ── 3. Chunk slicing ───────────────────────────────────────────────────
    chunk_size       = int(np.ceil(n_filtered / args.n_chunks))
    start            = args.chunk_id * chunk_size
    end              = min(start + chunk_size, n_filtered)
    chunk_global_idx = filtered_indices[start:end]   # global indices for this chunk
    theta_chunk      = {k: v[chunk_global_idx] for k, v in theta.items()}

    print(f"[INFO] Chunk {args.chunk_id}/{args.n_chunks}: "
          f"{len(chunk_global_idx)} events  (filtered positions {start}–{end})")

    if len(chunk_global_idx) == 0:
        print("[WARN] Empty chunk — nothing to do.")
        # Still write an empty npz so merge can account for it
        chunk_file = (f"{output_dir}/TDE_detectability_chunk{args.chunk_id:04d}_"
                      f"1E{np.log10(args.events):.0f}_{args.SURVEY}.npz")
        np.savez(chunk_file, detectability=np.array([], dtype=int),
                 global_indices=np.array([], dtype=int))
        raise SystemExit(0)

    # ── 4. Run simulations ─────────────────────────────────────────────────
    detectability_chunk = process_events(theta_chunk)

    n_det = detectability_chunk.sum()
    print(f"[SUMMARY] Chunk {args.chunk_id}: "
          f"{n_det}/{len(chunk_global_idx)} events detectable.")

    # ── 5. Save chunk result ───────────────────────────────────────────────
    chunk_file = (f"{output_dir}/TDE_detectability_chunk{args.chunk_id:04d}_"
                  f"1E{np.log10(args.events):.0f}_{args.SURVEY}.npz")
    np.savez(chunk_file,
             detectability=detectability_chunk,
             global_indices=chunk_global_idx)
    print(f"[DONE] Saved → {chunk_file}")