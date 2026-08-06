import os, sys
sys.path.insert(0, os.path.abspath('../../poplar'))

import numpy as np
import h5py
import pickle
import torch
import argparse


parser = argparse.ArgumentParser(description="Generate data for training.")
parser.add_argument("--events", type=int, required=True, help="total events")
parser.add_argument("--TELESCOPE", type=str, required=True, help="telescope name")
args = parser.parse_args()


# ── Paths ────────────────────────────────────────────────────────────────────
TRAIN_DIR   = f"./data/PRE_TRAIN_DATA_TDE/1E{int(np.log10(args.events))}_events"
H5_PATH     = f"{TRAIN_DIR}/TDE_params_1E{int(np.log10(args.events))}_{args.TELESCOPE}.h5"
NPY_X_PATH  = f"{TRAIN_DIR}/TDE_params_scaled_1E{int(np.log10(args.events))}_{args.TELESCOPE}.npy"
NPY_Y_PATH  = f"{TRAIN_DIR}/TDE_detectability_1E{int(np.log10(args.events))}_{args.TELESCOPE}.npy"   # adjust name if different
MODEL_PATH  = (f"{TRAIN_DIR}/TDE_DETECT_BCE_no_rescaler_512_4_10/"
               "best_final_model/model/model.pth")

SEP = "=" * 60

# ── 1. HDF5 structure ────────────────────────────────────────────────────────
print(SEP)
print("1. Training HDF5 — key order, shape, dtype, min/max")
print(SEP)
with h5py.File(H5_PATH, "r") as f:
    keys = list(f.keys())
    print(f"Keys (this is the COLUMN ORDER in the .npy): {keys}\n")
    for k in keys:
        d = f[k][:]
        print(f"  {k:12s}  shape={d.shape}  dtype={d.dtype}"
              f"  min={d.min():.4g}  max={d.max():.4g}"
              f"  mean={d.mean():.4g}")

# ── 2. Scaled .npy ───────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("2. Scaled input .npy")
print(SEP)
try:
    X = np.load(NPY_X_PATH)
    print(f"Shape : {X.shape}   (rows=events, cols=features)")
    print(f"dtype : {X.dtype}")
    print(f"\nPer-column stats (col index → matches key order above):")
    print(f"  {'col':>4}  {'key':>12}  {'min':>8}  {'max':>8}  {'mean':>8}  {'std':>8}")
    for i, k in enumerate(keys):
        col = X[:, i]
        print(f"  {i:>4}  {k:>12}  {col.min():>8.4f}  {col.max():>8.4f}"
              f"  {col.mean():>8.4f}  {col.std():>8.4f}")
except FileNotFoundError:
    print(f"  [NOT FOUND] {NPY_X_PATH}")
    print("  → Check the actual filename with: ls", TRAIN_DIR)

# ── 3. Labels .npy ───────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("3. Labels .npy")
print(SEP)
try:
    y = np.load(NPY_Y_PATH).ravel()
    print(f"Shape    : {y.shape}")
    print(f"Unique   : {np.unique(y)}")
    print(f"Balance  : {(y==1).sum()} detectable  /  {(y==0).sum()} non-detectable"
          f"  ({(y==1).mean():.1%} positive)")
except FileNotFoundError:
    print(f"  [NOT FOUND] {NPY_Y_PATH}")
    print(f"  → List available .npy files:")
    import glob
    for p in sorted(glob.glob(f"{TRAIN_DIR}/*.npy")):
        print(f"      {p}")

# ── 4. Model ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("4. Model")
print(SEP)
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"Type     : {type(model)}")
    print(f"Params   : {sum(p.numel() for p in model.parameters()):,} total weights")
    print(f"\nArchitecture:\n{model}")

    # Quick forward pass to confirm input/output dimensions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dummy = torch.zeros(4, len(keys), device=device)
    with torch.no_grad():
        out = model(dummy)
    print(f"\nDummy forward: input={tuple(dummy.shape)}  output={tuple(out.shape)}")
    print(f"Output range on zeros: min={out.min().item():.4f}  max={out.max().item():.4f}")
    print(f"After sigmoid        : {torch.sigmoid(out).mean().item():.4f}")

except FileNotFoundError:
    print(f"  [NOT FOUND] {MODEL_PATH}")
except Exception as e:
    print(f"  [ERROR] {e}")

print(f"\n{SEP}")
print("Done.")