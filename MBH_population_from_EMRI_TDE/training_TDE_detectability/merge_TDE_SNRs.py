import argparse, os, glob
import numpy as np
import h5py
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--events",    type=int, required=True)
parser.add_argument("--SURVEY",    type=str, required=True)
parser.add_argument("--n_chunks",  type=int, required=True)
args = parser.parse_args()

output_dir = (f"./data/"
              f"PRE_TRAIN_DATA_TDE/1E{np.log10(args.events):.0f}_events/")

# ── Load all chunks ────────────────────────────────────────────────────────
pattern = (f"{output_dir}/TDE_detectability_chunk*"
           f"_1E{np.log10(args.events):.0f}_{args.SURVEY}.npz")
chunk_files = sorted(glob.glob(pattern))

if len(chunk_files) != args.n_chunks:
    raise RuntimeError(f"Expected {args.n_chunks} chunks, found {len(chunk_files)}. "
                       f"Some jobs likely failed.")

# Load original full parameter file to get n_total
with h5py.File(f"{output_dir}/TDE_params_1E{np.log10(args.events):.0f}_{args.SURVEY}.h5", "r") as hf:
    n_total = len(hf["lgMBH"])
    theta = {k: hf[k][:] for k in hf.keys()}

detectability_full = np.zeros(n_total, dtype=int)

for f in chunk_files:
    data = np.load(f)
    detectability_full[data["global_indices"]] = data["detectability"]

print(f"[MERGE] {detectability_full.sum()} / {n_total} detectable events.")

# ── Save merged output ─────────────────────────────────────────────────────
out_h5  = f"{output_dir}/TDE_detectability_1E{np.log10(args.events):.0f}_{args.SURVEY}.h5"
out_npy = f"{output_dir}/TDE_detectability_1E{np.log10(args.events):.0f}_{args.SURVEY}.npy"

with h5py.File(out_h5, "w") as hf:
    hf.create_dataset("detectability", data=detectability_full)
np.save(out_npy, detectability_full)

# ── Plot ───────────────────────────────────────────────────────────────────
det_idx = detectability_full == 1
params  = [
    ("lgMBH", (0,0)), ("z_gal", (0,1)), ("ra_rad", (0,2)),
    ("dec_rad",(1,0)), ("eta",  (1,1)), ("alpha",  (1,2)),
    ("beta",  (2,0)), ("t0_mjd_transient",(2,1)),
]

fig, axs = plt.subplots(3, 3, figsize=(15, 15))
for key, (r, c) in params:
    ax = axs[r, c]
    ax.hist(theta[key],          bins="auto", density=True,
            histtype="step", color="blue", lw=2, label="All events")
    ax.hist(theta[key][det_idx], bins="auto", density=True,
            histtype="step", color="red",  lw=2, ls="--", label="Detectable")
    ax.set_title(key);  ax.set_ylabel("PDF")

axs[0,0].legend()
axs[2,2].axis("off")
plt.tight_layout()
plt.savefig(f"{output_dir}/TDE_params_detectability_"
            f"1E{int(np.log10(args.events))}_{args.SURVEY}.png")
plt.close()
print("[DONE] Merge complete.")