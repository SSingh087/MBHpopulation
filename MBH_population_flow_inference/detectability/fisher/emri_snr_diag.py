#!/usr/bin/env python
"""EMRI SNR diagnostic: why does the StableEMRIFisher SNR disagree with the SNR network (ratio 0.1x-1000x)?
For each event of the EMRI Fisher test it recomputes the SNR with the Fisher package's own SNR routine (no derivatives, one waveform)
and prints it next to the network value.  For the first event it also checks scaling laws:
  dist x10 -> SNR/10 ; mu x2 -> SNR x2 (to first order) ; T/4 -> SNR lower.   If these fail, the waveform/PSD normalisation is wrong.
Usage: python emri_snr_diag.py --gpu 1 --bank_dir DIR"""
import argparse, glob, time, warnings, numpy as np
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=1); ap.add_argument("--bank_dir", required=True)
ap.add_argument("--dt", type=float, default=15.0); ap.add_argument("--max_events", type=int, default=6); a = ap.parse_args()
from few.waveform import GenerateEMRIWaveform
from stableemrifisher.fisher import StableEMRIFisher
# use_rk4=True makes sigma_M ~5000x too small (cond ~1e17); see emri_validation_repro.py (configs D vs B)
wave = GenerateEMRIWaveform("Pn5AAKWaveform", return_list=True, inspiral_kwargs=dict(DENSE_STEPPING=0, max_init_len=int(1e7), err=1e-10, use_rk4=False),
                            sum_kwargs=dict(use_gpu=bool(a.gpu), pad_output=True), use_gpu=bool(a.gpu))
rows = []
for f in sorted(glob.glob(f"{a.bank_dir}/chunk_*.npz")):
    d = np.load(f)
    for i in range(len(d["ok"])): rows.append((d["theta"][i], float(d["Tb"][i]), float(d["snr_net"][i]), float(d["snr1"][i]), d["param_names"]))
print(f"{len(rows)} events in the test bank", flush=True)
def snr(v, T):
    s = StableEMRIFisher(*[np.float64(x) for x in v], dt=a.dt, T=T, EMRI_waveform_gen=wave, param_names=list(rows[0][4]), use_gpu=bool(a.gpu),
                         live_dangerously=True, filename=None)
    return float(s.SNRcalc_SEF())
print("\n%-3s %7s %6s %6s %6s %6s %6s %10s %12s %10s %8s" % ("i", "lgM", "a", "e0", "Y0", "z-ish", "Tb", "snr_net", "snr_fisher_run", "snr_here", "t[s]"))
for i, (v, Tb, sn, sf, names) in enumerate(rows[:a.max_events]):
    t = time.time(); s = snr(v, 0.99 * Tb)
    print("%-3d %7.2f %6.2f %6.2f %6.2f %6.3f %6.2f %10.1f %12.1f %10.1f %8.0f" % (i, np.log10(v[0]), v[2], v[4], v[5], v[6], Tb, sn, sf, s, time.time() - t), flush=True)
v, Tb = rows[0][0], rows[0][1]; s0 = snr(v, 0.99 * Tb)
print(f"\nscaling checks on event 0 (SNR {s0:.1f}):")
v2 = v.copy(); v2[6] *= 10; print("  dist x10 -> SNR ratio %.4f   (expect 0.1)" % (snr(v2, 0.99 * Tb) / s0), flush=True)
v3 = v.copy(); v3[1] *= 2; print("  mu x2    -> SNR ratio %.4f   (expect ~2)" % (snr(v3, 0.99 * Tb) / s0), flush=True)
print("  T x0.25  -> SNR ratio %.4f   (expect <1)" % (snr(v, 0.25 * 0.99 * Tb) / s0), flush=True)
v4 = v.copy(); v4[6] *= 1000; print("  dist x1000 (= units Gpc<->Mpc) -> SNR %.3f" % snr(v4, 0.99 * Tb), flush=True)
