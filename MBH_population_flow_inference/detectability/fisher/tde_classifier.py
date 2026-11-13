"""TDE detectability classifier wrapper (tde_fallback, ZTF/LSST).  Replicates data_generation/calculate_SNR_TDE.py exactly:
features = [lgMBH, z, ra, dec, eta, alpha, beta, t0, tvisc, bb, rph0, lphoto], each rescaled with the min/max of the
classifier's TRAINING file (stored once in <model dir>/scaling.json); output = sigmoid(logit) = detection probability.
NOTE: alpha and beta are classifier inputs even though tde_fallback ignores them physically -> draw them from their priors."""
import json, os, sys, pickle, numpy as np, torch
sys.path.insert(0, "/home/2673888s/poplar")
MODELS = "/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference/models/detectability/tde"
NAME = {"ZTF": "tde_fallback_512_4_10_ZTF", "LSST": "tde_fallback_128_4_10_LSST", "ZTFext": "tde_fallback_512_4_10_ZTFext"}   # ZTFext: retrained with z up to 2
TRAIN = "/data/wiay/postgrads/shashwat/EMRI_TDE_data/PRE_TRAIN_DATA_TDE/1E6_events/TDE_params_1E6_{s}_tde_fallback.h5"
SHARED = ["lgMBH", "z_gal", "ra_rad", "dec_rad", "eta", "alpha", "beta", "t0_mjd_transient"]
EXTRA = [("tvisc", "log"), ("bb", "linear"), ("rph0", "log"), ("lphoto", "linear")]
FEATURES = SHARED + [k for k, _ in EXTRA]


def make_scaling(survey):
    import h5py
    sc = {}
    with h5py.File(TRAIN.format(s=survey), "r") as f:
        for k in SHARED:
            d = f[k][:]; sc[k] = ["log", float(np.log10(d).min()), float(np.log10(d).max()), -1., 1.] if k in ("eta", "alpha") else ["linear", float(d.min()), float(d.max()), 0., 1.]
        for k, kind in EXTRA:
            d = f[k][:]; sc[k] = ["log", float(np.log10(d).min()), float(np.log10(d).max()), -1., 1.] if kind == "log" else ["linear", float(d.min()), float(d.max()), 0., 1.]
    json.dump(sc, open(f"{MODELS}/{NAME[survey]}/scaling.json", "w"), indent=1); return sc


class TDEClassifier:
    def __init__(self, survey, device="cpu"):
        d = f"{MODELS}/{NAME[survey]}"
        if not os.path.exists(f"{d}/scaling.json"): make_scaling(survey)
        self.sc = json.load(open(f"{d}/scaling.json")); self.survey = survey
        self.model = pickle.load(open(f"{d}/model/model.pth", "rb")); self.model.to(device).eval(); self.device = device

    def prob(self, ev):
        """ev: dict of arrays with keys lgMBH,z_gal,ra_rad,dec_rad,eta,alpha,beta,t0_mjd_transient,tvisc,bb,rph0,lphoto."""
        X = []
        for k in FEATURES:
            kind, lo, hi, tlo, thi = self.sc[k]; v = np.asarray(ev[k], float); v = np.log10(v) if kind == "log" else v
            X.append((v - lo) / (hi - lo) * (thi - tlo) + tlo)
        X = torch.tensor(np.stack(X, 1), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return torch.sigmoid(self.model(X)).reshape(-1).cpu().numpy()
