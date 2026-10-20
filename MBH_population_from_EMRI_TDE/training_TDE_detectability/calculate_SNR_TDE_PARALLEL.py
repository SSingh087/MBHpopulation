"""Chunked version of calculate_SNR_TDE.py for parallel (condor) runs."""
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

# ── Per-model parameter dispatch ────────────────────────────────────────────
# Which shared base params each model's function signature accepts, and the
# key name it expects them under (mbh_6 vs mbh6, stellar_mass vs mstar).
# "NOT" cases (e.g. tde_analytical has no BH mass/eta/alpha/beta) are simply
# omitted from that model's mapping below.
SHARED_PARAM_MAP = {
    "cooling_envelope":              {"mbh_key": "mbh_6", "mass_key": "stellar_mass", "eta": True,  "alpha_beta": True},
    "stream_stream_tde":             {"mbh_key": "mbh_6", "mass_key": "mstar",        "eta": False, "alpha_beta": False},
    "tde_analytical":                {"mbh_key": None,    "mass_key": None,           "eta": False, "alpha_beta": False},
    "tde_fallback":                  {"mbh_key": "mbh6",  "mass_key": "mstar",        "eta": True,  "alpha_beta": False},
}

# Extra per-model parameter dataset keys (must match gen_TDE_params.py's
# extra_data keys / redback's function-signature kwarg names exactly).
EXTRA_PARAM_KEYS = {
    "cooling_envelope":              [],
    "stream_stream_tde":             ["peak_time", "sigma_t", "c1", "del_omega", "f", "h_r", "inc_tcool"],
    "tde_analytical":                ["mej", "vej", "kappa", "kappa_gamma", "temperature_floor", "l0", "t_0_turn"],
    "tde_fallback":                  ["tvisc", "bb", "rph0", "lphoto", "leddlimit"],
}

STELLAR_MASS_FIXED = 1.0  # Msun, fixed for every model that has stellar_mass/mstar

# ── Argument parsing ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Calculate SNR for TDE events.")
parser.add_argument("--events",           type=int,   required=True)
parser.add_argument("--LENGTH_OF_TDE",    type=float, required=True)
parser.add_argument("--MIN_DETECTIONS",   type=int,   required=True)
parser.add_argument("--BANDS", nargs="+", default=["ztfg", "ztfr", "ztfi"])
parser.add_argument("--SURVEY",           type=str,   default="ztf")
parser.add_argument("--Z_MIN",            type=float, default=0.0)
parser.add_argument("--Z_MAX",            type=float, default=2.0)
parser.add_argument("--model",            type=str,   required=True, choices=MODEL_CHOICES)
parser.add_argument("--chunk_id",         type=int,   default=0,  help="0-indexed chunk ID")
parser.add_argument("--n_chunks",         type=int,   default=1,  help="Total number of chunks")
args = parser.parse_args()

LENGTH_OF_TDE = args.LENGTH_OF_TDE
MIN_DETECTIONS   = args.MIN_DETECTIONS
BANDS            = args.BANDS
SURVEY           = args.SURVEY   # keep original for output filenames
MODEL_NAME       = args.model

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

SHARED_KEYS = ["lgMBH", "z_gal", "ra_rad", "dec_rad", "eta", "alpha", "beta", "t0_mjd_transient"]
EXTRA_KEYS = EXTRA_PARAM_KEYS[MODEL_NAME]
MAP = SHARED_PARAM_MAP[MODEL_NAME]

# ── Core simulation functions ──────────────────────────────────────────────
def run_telescope_simulator(theta):
    sim = SIMULATE_FN(
        model=MODEL_NAME,
        survey=SURVEY_STR,
        parameters=theta,
        model_kwargs={},
        end_transient_time=LENGTH_OF_TDE,
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


def build_theta(theta_dict, i):
    """Build the per-event parameter dict for MODEL_NAME using only the
    kwargs that model's redback function signature actually accepts."""
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
    n_events = len(theta_dict["z_gal"])

    thetas = [build_theta(theta_dict, i) for i in range(n_events)]

    print(f"[INFO] Running {n_events} transient simulations for model '{MODEL_NAME}'...")

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
               f"TDE_params_1E{np.log10(args.events):.0f}_{args.SURVEY}_{MODEL_NAME}.h5")
    print(f"[INFO] Loading HDF5: {h5_path}")

    with h5py.File(h5_path, "r") as hf:
        theta = {k: hf[k][:] for k in SHARED_KEYS + EXTRA_KEYS}

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

    chunk_file = (f"{output_dir}/TDE_detectability_chunk{args.chunk_id:04d}_"
                  f"1E{np.log10(args.events):.0f}_{args.SURVEY}_{MODEL_NAME}.npz")

    if len(chunk_global_idx) == 0:
        print("[WARN] Empty chunk — nothing to do.")
        # Still write an empty npz so merge can account for it
        np.savez(chunk_file, detectability=np.array([], dtype=int),
                 global_indices=np.array([], dtype=int))
        raise SystemExit(0)

    # ── 4. Run simulations ─────────────────────────────────────────────────
    detectability_chunk = process_events(theta_chunk)

    n_det = detectability_chunk.sum()
    print(f"[SUMMARY] Chunk {args.chunk_id} ({MODEL_NAME}): "
          f"{n_det}/{len(chunk_global_idx)} events detectable.")

    # ── 5. Save chunk result ───────────────────────────────────────────────
    np.savez(chunk_file,
             detectability=detectability_chunk,
             global_indices=chunk_global_idx)
    print(f"[DONE] Saved → {chunk_file}")
