"""Covariance from a Fisher matrix, done properly.
np.linalg.pinv(F) on the RAW matrix drops singular values below 1e-15*max, which silently removes the most weakly constrained direction
(e.g. the mass in an EMRI Fisher, where raw entries span >20 orders of magnitude because the parameters have different units) and makes
sigma far too small once cond(F) > ~1e15.  Normalise by the diagonal first, invert, undo the scaling.  The normalised condition number is the
meaningful one (~1e5 for a healthy EMRI Fisher; the raw one is ~1e16)."""
import numpy as np


def covariance(F, extra_precision=None):
    """returns (C, cond_normalised, cond_raw).  extra_precision: optional diagonal prior precision added to F (TDE nuisance priors)."""
    Fs = 0.5 * (np.asarray(F, float) + np.asarray(F, float).T)
    if extra_precision is not None:
        Fs = Fs + np.diag(extra_precision)
    d = 1.0 / np.sqrt(np.clip(np.diag(Fs), 1e-300, None))
    Fn = Fs * d[:, None] * d[None, :]
    try:
        Cn = np.linalg.inv(Fn)
    except np.linalg.LinAlgError:
        Cn = np.linalg.pinv(Fn)
    return Cn * d[:, None] * d[None, :], float(np.linalg.cond(Fn)), float(np.linalg.cond(Fs))
