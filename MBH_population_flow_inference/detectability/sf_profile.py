#!/usr/bin/env python
"""sf_profile.py -- time each stage of the brute-force EMRI selection-function Monte Carlo (flow sample -> inverse transform -> (e0,p0) emulator + SNR network -> detection probability)
at several batch sizes; prints samples/s per stage.  One short GPU job."""
import sys, time, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, f"{HERE}/fisher")
from sf_lib import *
import torch
dev = torch.device("cuda"); print("GPU:", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0), flush=True)
meta = {k: v for k, v in np.load(f"{DATA}/simulations/consolidated/meta.npz").items()}
sf = EMRISF("cuda"); ls = lam_scaled(lambda_set(meta, 2)[0], dev); gen = torch.Generator(device=dev).manual_seed(1)
def T(f, reps=3):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(reps): r = f()
    torch.cuda.synchronize(); return (time.time() - t) / reps, r
for n in (250_000, 500_000, 1_000_000):
    print(f"--- batch {n:,}", flush=True)
    t_s, u = T(lambda: sf.flow.sample(ls.expand(n, -1)))
    t_i, xs = T(lambda: sf.xt.inverse(u))
    Y0 = 0.1 + 0.6 * torch.rand(n, device=dev, generator=gen); Tt = 1.0 + 5.0 * torch.rand(n, device=dev, generator=gen)
    t_r, (rho, e0) = T(lambda: sf.rho_of(xs, Y0, Tt))
    t_p, _ = T(lambda: sf.pdet(rho))
    t_all, _ = T(lambda: sf.pdet_points(xs, 1, gen))
    for nm, t in (("flow.sample", t_s), ("inverse transform", t_i), ("emulator+SNR net", t_r), ("pdet lookup", t_p), ("pdet_points (all but flow)", t_all)):
        print(f"  {nm:28s} {t*1e3:9.1f} ms   {n/t/1e6:8.2f} M samples/s", flush=True)
    print(f"  total flow+pdet_points: {n/(t_s+t_i+t_all)/1e6:.2f} M samples/s -> 1e8 samples in {1e8/(n/(t_s+t_i+t_all)):.0f} s", flush=True)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
n = 1_000_000; t_s, u = T(lambda: sf.flow.sample(ls.expand(n, -1))); print(f"TF32 on: flow.sample batch {n:,}: {t_s*1e3:.1f} ms  {n/t_s/1e6:.2f} M samples/s", flush=True)
