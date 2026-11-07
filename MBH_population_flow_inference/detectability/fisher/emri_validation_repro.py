#!/usr/bin/env python
"""Reproduce the StableEMRIFisher validation case (M=5e5, mu=14, a=0.8, p0=11.58, e0=0.3, SNR~30, T=1 yr) and compare the Fisher sigmas with the MCMC
posterior widths shipped with the package (validation/data_files/MCMC_results/samples_large_q_SNR_30.npy), under three configurations:
  A  the package's validation settings (der_order=6, use_rk4=False, max_init_len=1e4, dt=10)
  B  validation settings but der_order=2 (the package default, what our bank used)
  C  our bank settings (dt=15, use_rk4=True, max_init_len=1e7, der_order=2)
  D/E/F  B with ONE change each: use_rk4=True / max_init_len=1e7 / dt=15  (isolates what breaks sigma_M in C)
Question answered: is our M error (sigma_M/M ~ 1e-11, cond ~ 1e18) a configuration problem?  Run on a GPU (a few minutes per configuration)."""
import argparse, os, sys, time, warnings, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fisher_cov import covariance
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser(); ap.add_argument("--configs", nargs="+", default=["A", "B", "C", "D", "E", "F"]); a = ap.parse_args()
from few.waveform import GenerateEMRIWaveform
from stableemrifisher.fisher import StableEMRIFisher
VAL = "/data/wiay/postgrads/shashwat/conda_envs/StableEMRIFisher/validation/data_files/MCMC_results/samples_large_q_SNR_30.npy"
names = ['M', 'mu', 'a', 'p0', 'e0', 'Y0', 'dist', 'qS', 'phiS', 'qK', 'phiK', 'Phi_phi0', 'Phi_theta0', 'Phi_r0']
truth = np.array([5e5, 14, 0.8, 11.58, 0.3, np.cos(0.8), 2., 1.5, 0.7, 1.2, 0.6, 2., 3., 4.])
mcmc = np.load(VAL).std(axis=1)
CONF = {"A": dict(dt=10., rk4=False, mil=int(1e4), der=6), "B": dict(dt=10., rk4=False, mil=int(1e4), der=2), "C": dict(dt=15., rk4=True, mil=int(1e7), der=2),
        "D": dict(dt=10., rk4=True, mil=int(1e4), der=2),    # B + use_rk4
        "E": dict(dt=10., rk4=False, mil=int(1e7), der=2),   # B + max_init_len 1e7
        "F": dict(dt=15., rk4=False, mil=int(1e4), der=2)}   # B + dt 15
res = {}
for c in a.configs:
    k = CONF[c]
    wave = GenerateEMRIWaveform("Pn5AAKWaveform", return_list=True, inspiral_kwargs=dict(DENSE_STEPPING=0, max_init_len=k["mil"], err=1e-10, use_rk4=k["rk4"]),
                                sum_kwargs=dict(use_gpu=True, pad_output=True), use_gpu=True)
    t = time.time()
    sef = StableEMRIFisher(*truth, dt=k["dt"], T=1.0, EMRI_waveform_gen=wave, param_names=names, use_gpu=True, filename=None, CovEllipse=False, live_dangerously=False, der_order=k["der"])
    F = np.asarray(sef()); C, cond_n, cond_raw = covariance(F); res[c] = (np.sqrt(np.clip(np.diag(C), 0, None)), cond_raw, float(np.sqrt(sef.SNR2)), time.time() - t)
    print(f"config {c} {k}: SNR {res[c][2]:.1f}, cond(F) {res[c][1]:.2e}, {res[c][3]:.0f}s", flush=True)
print("\n%-11s %12s" % ("param", "MCMC std") + "".join("%14s" % f"Fisher {c}" for c in a.configs) + "".join("%12s" % f"{c}/MCMC" for c in a.configs))
for i, n in enumerate(names):
    print("%-11s %12.4g" % (n, mcmc[i]) + "".join("%14.4g" % res[c][0][i] for c in a.configs) + "".join("%12.3g" % (res[c][0][i] / mcmc[i]) for c in a.configs))
