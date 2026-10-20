#!/usr/bin/env python
"""
Processes a contiguous slice [START_IDX, END_IDX) of the detection-passed
TDE catalog (output of the SNR/classifier step) and writes one Fisher
result file per batch. Designed to be called by an HTCondor DAGMan node,
one node per batch of BATCH_SIZE events.

Input catalog: TDE_SNRs_{SURVEY}_{suffix}.h5
    -- already flat (one row = one event), already filtered to
       detection_prob > threshold. No galaxy grouping, no event_index
       mapping needed -- START_IDX/END_IDX index directly into these
       arrays.
    -- t0_mjd_transient is a single generic field (survey window cut was
       already applied upstream in doc 3); no more t0_mjd_transient_ZTF /
       t0_mjd_transient_LSST split.

Optional plotting (--PLOT_CORNER / --PLOT_FISHER / --PLOT_COVARIANCE):
    per event, saves corner/heatmap PNGs for BOTH the true (F1/C1) and
    noisy (F2/C2) draws under {loc}/plots/batch_{BATCH_ID}/event_{i}_*.png.
    Off by default -- a full production run over thousands of events
    should not silently try to render thousands of corner plots.
"""

import os, sys, argparse
import numpy as np
import h5py

sys.path.insert(0, os.path.abspath('./'))
from utils import ASTROPHYSICAL_DATA_DIR, FISHER_DATA_DIR
from fish_TDE import *
from utils_fisher import *

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES",       type=int, required=True)
parser.add_argument("--SURVEY",         type=str, default="ztf")
parser.add_argument("--BANDS",          nargs="+", default=["ztfg", "ztfr", "ztfi"])
parser.add_argument("--OBSERVING_WINDOW", type=float, required=True)
parser.add_argument("--START_IDX",      type=int, required=True, help="inclusive start row in detection-passed catalog")
parser.add_argument("--END_IDX",        type=int, required=True, help="exclusive end row")
parser.add_argument("--BATCH_ID",       type=int, required=True)
parser.add_argument("--PLOT_CORNER",     type=int, default=0, choices=[0, 1], help="1=save true+noisy corner plots per event, 0=skip")
parser.add_argument("--PLOT_FISHER",     type=int, default=0, choices=[0, 1], help="1=save true+noisy Fisher-matrix heatmaps per event, 0=skip")
parser.add_argument("--PLOT_COVARIANCE", type=int, default=0, choices=[0, 1], help="1=save true+noisy covariance-matrix heatmaps per event, 0=skip")
parser.add_argument("--mbh_mass_method", type=str, required=False, default="M_sigma", choices=["use_sigma", "use_lgMgal"])
parser.add_argument("--nucleation_method", type=str, required=False, default="hannah", choices=["hannah", "neumayer"])
args = parser.parse_args()
args.PLOT_CORNER     = bool(args.PLOT_CORNER)
args.PLOT_FISHER     = bool(args.PLOT_FISHER)
args.PLOT_COVARIANCE = bool(args.PLOT_COVARIANCE)

loc      = f"{FISHER_DATA_DIR}/{args.GALAXIES}"
os.makedirs(f"{loc}/batches", exist_ok=True)

ANY_PLOTTING = args.PLOT_CORNER or args.PLOT_FISHER or args.PLOT_COVARIANCE
if ANY_PLOTTING:
    plot_dir = f"{loc}/plots/batch_{args.BATCH_ID:05d}"
    os.makedirs(plot_dir, exist_ok=True)

SNR_PATH = f"{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}/TDE_SNRs_{args.SURVEY.upper()}_{args.mbh_mass_method}_{args.nucleation_method}.h5"

N_SAMPLES_PER_THETA = 2000
SIM_FN, SURVEY_STR = choose_simulator(args.SURVEY)
PARAMS = list(PARAM_INFO.keys())

with h5py.File(SNR_PATH, "r") as f:
    n_total = f["z_gal"].shape[0]
    end = min(args.END_IDX, n_total)
    sl = slice(args.START_IDX, end)

    catalog = {
        "z_gal":            f["z_gal"][sl],
        "lgMBH_mass":       f["lgMBH_mass"][sl],
        "star_mass":        f["star_mass"][sl],
        "eta":              f["eta"][sl],
        "alpha":            f["alpha"][sl],
        "beta":             f["beta"][sl],
        "ra_rad":           f["ra_rad"][sl],
        "dec_rad":          f["dec_rad"][sl],
        "t0_mjd_transient": f["t0_mjd_transient"][sl],
    }

n_events = len(catalog["z_gal"])
print(f"[BATCH {args.BATCH_ID}] rows [{args.START_IDX}:{end}) -> {n_events} events")

if n_events == 0:
    print(f"[BATCH {args.BATCH_ID}] empty slice, nothing to do -- exiting cleanly.")
    sys.exit(0)


def sample_truncated_gaussian(mean, cov, n_target, n_draw=50000):
    """Sample a multivariate Gaussian truncated to the prior bounds."""
    samples = np.random.multivariate_normal(mean, cov, size=n_draw)
    mask = (
        (0.001 <= samples[:, 0]) & (samples[:, 0] <= 8) &
        (-2 <= samples[:, 1])    & (samples[:, 1] <= 1) &
        (0.001 <= samples[:, 2]) & (samples[:, 2] <= 5) &
        (0.001 <= samples[:, 3]) & (samples[:, 3] <= 0.1) &
        (0.01 <= samples[:, 4])  & (samples[:, 4] <= 0.1) &
        (1.001 <= samples[:, 5]) & (samples[:, 5] <= 30)
    )
    valid = samples[mask]
    if len(valid) == 0:
        raise RuntimeError("No samples inside prior")
    if len(valid) >= n_target:
        return valid[:n_target]
    idx = np.random.choice(len(valid), size=n_target, replace=True)
    return valid[idx]


def save_event_plots(global_i, truths1, C1, true_samples, truths2, C2, noisy_samples):
    """Saves corner/Fisher/covariance PNGs for both the true and noisy draws.
    Wrapped in try/except per-plot so a plotting failure (e.g. a singular
    covariance slipping through) never takes down the whole batch -- the
    Fisher results themselves are already written to the .h5 regardless."""
    tag = f"event_{global_i}"

    if args.PLOT_CORNER:
        for label, samples, truths in [("true", true_samples, truths1), ("noisy", noisy_samples, truths2)]:
            try:
                plot_corner(
                    samples=samples, truths=truths, params=PARAMS,
                    filename=f"{plot_dir}/{tag}_corner_{label}.png",
                    title=f"{label.capitalize()} Fisher -- event {global_i}",
                )
            except Exception as e:
                print(f"[BATCH {args.BATCH_ID}] event {global_i} corner-plot ({label}) failed: {type(e).__name__}: {e}")

    if args.PLOT_FISHER:
        for label, C in [("true", C1), ("noisy", C2)]:
            try:
                F = np.linalg.inv(C)
                plot_fisher_matrix(
                    F=F, params=PARAMS,
                    filename=f"{plot_dir}/{tag}_fisher_{label}.png",
                    title=f"{label.capitalize()} Fisher matrix -- event {global_i}",
                )
            except Exception as e:
                print(f"[BATCH {args.BATCH_ID}] event {global_i} fisher-plot ({label}) failed: {type(e).__name__}: {e}")

    if args.PLOT_COVARIANCE:
        for label, C in [("true", C1), ("noisy", C2)]:
            try:
                plot_covariance_matrix(
                    C=C, params=PARAMS,
                    filename=f"{plot_dir}/{tag}_covariance_{label}.png",
                    title=f"{label.capitalize()} covariance matrix -- event {global_i}",
                )
            except Exception as e:
                print(f"[BATCH {args.BATCH_ID}] event {global_i} covariance-plot ({label}) failed: {type(e).__name__}: {e}")


def process_event(local_i, global_i):
    """Fisher evaluation for a single event."""
    try:
        theta0 = {
            "redshift":     float(catalog["z_gal"][local_i]),
            "mbh_6":        float(catalog["lgMBH_mass"][local_i] - 6),
            "stellar_mass": float(catalog["star_mass"][local_i]),
            "eta":          float(catalog["eta"][local_i]),
            "alpha":        float(catalog["alpha"][local_i]),
            "beta":         float(catalog["beta"][local_i]),
            "t0_mjd_transient": float(catalog["t0_mjd_transient"][local_i]),
            "t0":           float(catalog["t0_mjd_transient"][local_i]),
            "ra":           float(catalog["ra_rad"][local_i]),
            "dec":          float(catalog["dec_rad"][local_i]),
        }

        df = run_telescope_simulator(theta0, SIM_FN, SURVEY_STR, args.OBSERVING_WINDOW)
        data_by_band = extract_observed_data_by_band(df, args.BANDS)
        times_by_band, bands = extract_times_by_band(data_by_band)

        F1, C1, _ = compute_fisher(theta0, times_by_band, bands)
        truths1 = [theta0[p] for p in PARAMS]
        true_samples = sample_truncated_gaussian(np.array(truths1), C1, N_SAMPLES_PER_THETA)

        F2, C2, truths2, noisy_samples = None, None, None, None
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                cand = sample_truncated_gaussian(np.array(truths1), C1, 50, n_draw=10000)
                theta_new = {p: cand[0][i] for i, p in enumerate(PARAMS)}
                F2, C2, _ = compute_fisher(theta_new, times_by_band, bands)
                truths2 = [theta_new[p] for p in PARAMS]
                noisy_samples = sample_truncated_gaussian(np.array(truths2), C2, N_SAMPLES_PER_THETA)
                break
            except (ValueError, np.linalg.LinAlgError, TypeError) as e:
                print(f"[BATCH {args.BATCH_ID}] event {global_i} F2 attempt {attempt+1} failed: {type(e).__name__}")
                if attempt == max_attempts - 1:
                    F2, C2, truths2, noisy_samples = F1, C1, truths1, true_samples.copy()

        if ANY_PLOTTING:
            save_event_plots(global_i, truths1, C1, true_samples, truths2, C2, noisy_samples)

        return dict(global_idx=global_i, F1=F1, C1=C1, truths1=truths1,
                    F2=F2, C2=C2, truths2=truths2,
                    true_samples=true_samples, noisy_samples=noisy_samples)
    except Exception as e:
        print(f"[CRITICAL] event {global_i} failed: {type(e).__name__}: {e}")
        return None


results = []
for local_i in range(n_events):
    global_i = args.START_IDX + local_i
    r = process_event(local_i, global_i)
    if r is not None:
        results.append(r)

out_path = f"{loc}/batches/fisher_batch_{args.BATCH_ID:05d}_{args.mbh_mass_method}_{args.nucleation_method}.h5"
with h5py.File(out_path, "w") as hf:
    hf.attrs["batch_id"]  = args.BATCH_ID
    hf.attrs["start_idx"] = args.START_IDX
    hf.attrs["end_idx"]   = end
    hf.attrs["n_events_in_batch"] = n_events
    hf.attrs["n_events_ok"] = len(results)

    for r in results:
        g = hf.create_group(f"event_{r['global_idx']}")
        g.create_dataset("fisher_matrix", data=r["F1"])
        g.create_dataset("covariance_matrix", data=r["C1"])
        g.create_dataset("errors", data=np.sqrt(np.diag(r["C1"])))
        g.create_dataset("truths", data=r["truths1"])
        g.create_dataset("fisher_matrix_new", data=r["F2"])
        g.create_dataset("covariance_matrix_new", data=r["C2"])
        g.create_dataset("errors_new", data=np.sqrt(np.diag(r["C2"])))
        g.create_dataset("truths_new", data=r["truths2"])
        g.create_dataset("true_samples", data=r["true_samples"], compression="gzip")
        g.create_dataset("noisy_samples", data=r["noisy_samples"], compression="gzip")

print(f"[BATCH {args.BATCH_ID}] wrote {len(results)}/{n_events} events -> {out_path}")
if ANY_PLOTTING:
    print(f"[BATCH {args.BATCH_ID}] wrote plots -> {plot_dir}")