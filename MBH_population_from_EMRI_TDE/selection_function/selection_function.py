"""Selection function alpha(Lambda) = P(detected | population hyperparameters), emulated by a trained MLP.

The hierarchical likelihood divides every detected event by alpha(Lambda); one network exists per
channel (EMRI, TDE_ZTF, TDE_LSST) and population (A, B). Networks are trained by
selection_function/train_selection_function.py and predict log(alpha).
"""
import json
import os
import pickle

import torch

SELECTION_DIR = "./data/selection_function"
LOG_ALPHA_FLOOR = -30.0


class SelectionFunction:
    """MLP emulator of log alpha(Lambda) for one channel and population."""

    def __init__(self, model, names):
        self.model = model.eval()
        self.names = names

    @classmethod
    def load(cls, population, channel, directory=SELECTION_DIR, device="cpu"):
        """Load <directory>/<population>_<channel>/{model.pth, names.json}."""
        folder = os.path.join(directory, f"{population}_{channel}")
        with open(os.path.join(folder, "model.pth"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(folder, "names.json")) as f:
            names = json.load(f)
        model.to(device)
        return cls(model, names)

    @torch.inference_mode()
    def log_alpha(self, x):
        """log alpha for a dict of hyperparameters (0-d tensors or floats); returns a 0-d tensor."""
        inp = torch.stack([torch.as_tensor(x[n], dtype=torch.float32) for n in self.names])
        out = self.model(inp[None, :]).reshape(())
        return out.clamp(min=LOG_ALPHA_FLOOR, max=0.0)
