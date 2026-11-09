#!/usr/bin/env python
"""Retrain the ZTF tde_fallback detectability classifier (same architecture/optimiser/loss as the original: 512x4 SiLU MLP, dropout 0.1, BCE, AdamW 1e-4,
ReduceLROnPlateau, 1000 epochs, 10 batches, 80/20 split) on the ORIGINAL 250k events (z<=0.5) PLUS new events with z in (0.5, 2.0] (tde_classifier_test.py --set ext).
Features/scaling identical in form (min/max of the training file; eta/alpha/tvisc/rph0 log -> [-1,1]); scaling.json is written next to the model.
Output: <DATA>/models/detectability/tde/tde_fallback_512_4_10_ZTFext/{model/model.pth, scaling.json, summary.json}"""
import glob, json, os, sys, shutil
import numpy as np, h5py, torch
sys.path.insert(0, "/home/2673888s/poplar"); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fisher"))
from poplar.nn.networks import LinearModel
from poplar.nn.training import train, train_test_split
from tde_classifier import FEATURES, EXTRA
DATA = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference"; OLD = "/data/wiay/postgrads/shashwat/EMRI_TDE_data/PRE_TRAIN_DATA_TDE/1E6_events"
LOGK = {"eta", "alpha"} | {k for k, kind in EXTRA if kind == "log"}
raw = {}
with h5py.File(f"{OLD}/TDE_params_1E6_ZTF_tde_fallback.h5") as f: raw = {k: f[k][:] for k in FEATURES}
with h5py.File(f"{OLD}/TDE_detectability_1E6_ZTF_tde_fallback.h5") as f: y = f["detectability"][:].astype(np.float32)
fs = sorted(glob.glob(f"{DATA}/simulations/tde_clf_test/ZTF_ext_*.npz")); print("old events", len(y), "ext chunks", len(fs), flush=True)
E_ = [np.load(f) for f in fs]; ok = [e["y"] >= 0 for e in E_]
for k in FEATURES: raw[k] = np.concatenate([raw[k]] + [e[k][o] for e, o in zip(E_, ok)])
y = np.concatenate([y] + [e["y"][o].astype(np.float32) for e, o in zip(E_, ok)]); print("total", len(y), "prevalence old/new/all: %.4f" % y[:250000].mean(), "%.4f" % y[250000:].mean(), "%.4f" % y.mean(), flush=True)
sc = {}; X = []
for k in FEATURES:
    v = np.log10(raw[k]) if k in LOGK else raw[k]; lo, hi = float(v.min()), float(v.max()); tlo, thi = (-1., 1.) if k in LOGK else (0., 1.)
    sc[k] = ["log" if k in LOGK else "linear", lo, hi, tlo, thi]; X.append((v - lo) / (hi - lo) * (thi - tlo) + tlo)
X = np.stack(X, 1).astype(np.float32); dev = "cuda" if torch.cuda.is_available() else "cpu"
xd = torch.as_tensor(X, device=dev); yd = torch.as_tensor(y, device=dev).view(-1)
model = LinearModel(in_features=X.shape[1], out_features=1, neurons=[512] * 4, activation=torch.nn.SiLU, rescaler=None, dropout=0.1); model.set_device(dev)
torch.manual_seed(0); xtr, xte, ytr, yte = train_test_split([xd, yd], 0.8)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6); sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=10, factor=0.5)
out = f"{DATA}/models/detectability/_clf_ext_training" + os.environ.get("DST_TAG", ""); os.makedirs(out, exist_ok=True)
train(model, data=[xtr, ytr, xte, yte], n_epochs=int(os.environ.get("EPOCHS", 1000)), n_batches=10, loss_function=torch.nn.BCEWithLogitsLoss(), optimiser=opt, verbose=True, save_best=True, scheduler=sch, outdir=out, update_every=10)
best = sorted(glob.glob(f"{out}/**/best_final_model/**/model.pth", recursive=True), key=os.path.getmtime)[-1]; print("best model file", best)
dst = f"{DATA}/models/detectability/tde/tde_fallback_512_4_10_ZTFext" + os.environ.get("DST_TAG", ""); os.makedirs(f"{dst}/model", exist_ok=True)
shutil.copy(best, f"{dst}/model/model.pth"); json.dump(sc, open(f"{dst}/scaling.json", "w"), indent=1)
json.dump(dict(n_total=int(len(y)), n_old=250000, n_ext=int(len(y) - 250000), prevalence=float(y.mean()), z_max=sc["z_gal"][2]), open(f"{dst}/summary.json", "w"))
print("saved ->", dst)
