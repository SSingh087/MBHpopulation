"""Train the MLP that maps population hyperparameters Lambda to log alpha(Lambda) for one channel."""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath('../poplar'))
from poplar.nn.networks import LinearModel
from poplar.nn.rescaling import ZScoreRescaler
from poplar.nn.training import train, train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--POPULATION", type=str, required=True, choices=["A", "B"])
parser.add_argument("--CHANNEL", type=str, required=True)
parser.add_argument("--DATA_DIR", type=str, default="./data/selection_function")
parser.add_argument("--num_neurons", type=int, default=256)
parser.add_argument("--layers", type=int, default=4)
parser.add_argument("--train_test_frac", type=float, default=0.9)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--n_epochs", type=int, default=1000)
parser.add_argument("--n_batches", type=int, default=100)
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

files = sorted(glob.glob(f"{args.DATA_DIR}/data_{args.POPULATION}_{args.CHANNEL}_*.npz"))
if not files:
    raise RuntimeError(f"No training chunks for {args.POPULATION}_{args.CHANNEL} in {args.DATA_DIR}")
chunks = [np.load(f) for f in files]
names = [str(n) for n in chunks[0]["names"]]
x = torch.as_tensor(np.concatenate([c["lam"] for c in chunks]), device=args.device).float()
y = torch.as_tensor(np.concatenate([c["log_alpha"] for c in chunks]), device=args.device).float()[:, None]
print(f"{len(x)} training points, inputs: {names}")

model = LinearModel(in_features=x.shape[-1], out_features=1, neurons=[args.num_neurons] * args.layers,
                    activation=torch.nn.SiLU, rescaler=ZScoreRescaler(x, y), dropout=0.0)
model.set_device(args.device)

xtrain, xtest, ytrain, ytest = train_test_split([x, y], args.train_test_frac)
optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", patience=10, factor=0.5)

out_dir = f"{args.DATA_DIR}/{args.POPULATION}_{args.CHANNEL}"
os.makedirs(out_dir, exist_ok=True)
train(model, data=[xtrain, ytrain, xtest, ytest], n_epochs=args.n_epochs, n_batches=args.n_batches,
      loss_function=torch.nn.SmoothL1Loss(), optimiser=optimiser, update_every=args.n_epochs // 10,
      verbose=True, save_best=True, scheduler=scheduler, outdir=f"{out_dir}/training/")

err = (model.run_on_dataset(xtest) - ytest).abs()
print(f"test |delta log alpha|: median {err.median().item():.3f}, 95% {err.quantile(0.95).item():.3f}")

# Files read by selection_function.SelectionFunction.load
with open(f"{out_dir}/model.pth", "wb") as f:
    pickle.dump(model.cpu(), f)
with open(f"{out_dir}/names.json", "w") as f:
    json.dump(names, f)
