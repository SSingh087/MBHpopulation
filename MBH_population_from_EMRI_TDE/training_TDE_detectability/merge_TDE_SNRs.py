"""Merge the chunked TDE detectability results into single training arrays."""
import argparse, os, glob
import numpy as np
import h5py
import matplotlib.pyplot as plt

# bpl_cooling_envelope and gaussianrise_cooling_envelope are excluded: both
# crash deterministically (KeyError: 'bands') under redback's
# SimulateOpticalTransient (confirmed still broken in redback 1.18.0).
DEFAULT_MODELS = [
    "cooling_envelope",
    "stream_stream_tde",
    "tde_analytical",
    "tde_fallback",
]

# Extra per-model parameter dataset keys. Kept in sync with
# gen_TDE_params.py / calculate_SNR_TDE_PARALLEL.py.
EXTRA_PARAM_KEYS = {
    "cooling_envelope":              [],
    "stream_stream_tde":             ["peak_time", "sigma_t", "c1", "del_omega", "f", "h_r", "inc_tcool"],
    "tde_analytical":                ["mej", "vej", "kappa", "kappa_gamma", "temperature_floor", "l0", "t_0_turn"],
    "tde_fallback":                  ["tvisc", "bb", "rph0", "lphoto", "leddlimit"],
}

SHARED_KEYS = ["lgMBH", "z_gal", "ra_rad", "dec_rad", "eta", "alpha", "beta", "t0_mjd_transient"]

parser = argparse.ArgumentParser()
parser.add_argument("--events",    type=int, required=True)
parser.add_argument("--SURVEY",    type=str, required=True)
parser.add_argument("--n_chunks",  type=int, required=True, help="number of SNR chunks PER MODEL")
parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                     help="models to merge chunks for -- each model is trained as its own "
                          "independent classifier, so no cross-model combination happens here")
args = parser.parse_args()

MODELS = args.models

output_dir = (f"./data/"
              f"PRE_TRAIN_DATA_TDE/1E{np.log10(args.events):.0f}_events/")

n_str = f"{np.log10(args.events):.0f}"

# ── Per-model chunk merge (each model stays a fully separate injection set --
#    training is per-model, so there is no cross-model concatenation/one-hot
#    step here) ────────────────────────────────────────────────────────────
for model in MODELS:
    pattern = f"{output_dir}/TDE_detectability_chunk*_1E{n_str}_{args.SURVEY}_{model}.npz"
    chunk_files = sorted(glob.glob(pattern))

    if len(chunk_files) != args.n_chunks:
        raise RuntimeError(f"[{model}] Expected {args.n_chunks} chunks, found {len(chunk_files)}. "
                           f"Some jobs likely failed.")

    param_file = f"{output_dir}/TDE_params_1E{n_str}_{args.SURVEY}_{model}.h5"
    with h5py.File(param_file, "r") as hf:
        n_total = len(hf["lgMBH"])
        model_shared = {k: hf[k][:] for k in SHARED_KEYS}
        model_extra = {k: hf[k][:] for k in EXTRA_PARAM_KEYS[model]}

    detectability_full = np.zeros(n_total, dtype=int)
    for f in chunk_files:
        data = np.load(f)
        if len(data["global_indices"]) > 0:
            detectability_full[data["global_indices"]] = data["detectability"]

    print(f"[MERGE][{model}] {detectability_full.sum()} / {n_total} detectable events.")

    out_h5 = f"{output_dir}/TDE_detectability_1E{n_str}_{args.SURVEY}_{model}.h5"
    with h5py.File(out_h5, "w") as hf:
        hf.create_dataset("detectability", data=detectability_full)
    np.save(f"{output_dir}/TDE_detectability_1E{n_str}_{args.SURVEY}_{model}.npy", detectability_full)

    # ── Per-model diagnostic plot ───────────────────────────────────────────
    det_idx = detectability_full == 1
    all_keys = SHARED_KEYS + EXTRA_PARAM_KEYS[model]
    all_data = {**model_shared, **model_extra}

    ncols = 3
    nrows = int(np.ceil(len(all_keys) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axs = np.atleast_2d(axs)
    for idx, key in enumerate(all_keys):
        r, c = divmod(idx, ncols)
        ax = axs[r, c]
        ax.hist(all_data[key],          bins="auto", density=True,
                histtype="step", color="blue", lw=2, label="All events")
        ax.hist(all_data[key][det_idx], bins="auto", density=True,
                histtype="step", color="red",  lw=2, ls="--", label="Detectable")
        ax.set_title(key); ax.set_ylabel("PDF")
    for idx in range(len(all_keys), nrows * ncols):
        r, c = divmod(idx, ncols)
        axs[r, c].axis("off")
    axs[0, 0].legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/TDE_params_detectability_1E{n_str}_{args.SURVEY}_{model}.png")
    plt.close()

print("[DONE] Merge complete (per-model outputs only -- no cross-model file is produced).")
