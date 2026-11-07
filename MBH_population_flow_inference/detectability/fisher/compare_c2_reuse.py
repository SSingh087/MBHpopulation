"""Compare the C2 covariance of the same event computed with a fresh stable-delta search (reference run) and with deltas reused from the first Fisher.
Same seed + same chunk => identical event and identical noisy point, so any difference is the delta reuse.  Usage: python compare_c2_reuse.py REF_DIR NEW_DIR"""
import sys, numpy as np, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from fisher_cov import covariance
a, b = (np.load(f"{d}/chunk_0000.npz") for d in sys.argv[1:3]); names = list(a["param_names"])
print("event 0: theta identical:", np.allclose(a["theta"][0], b["theta"][0]), "| theta_hat identical:", np.allclose(a["theta_hat"][0], b["theta_hat"][0], rtol=1e-9))
sa = np.sqrt(np.diag(covariance(a["F2"][0])[0])); sb = np.sqrt(np.diag(covariance(b["F2"][0])[0]))
print("C2 sigma ratio (reuse / fresh search) per parameter:")
for n, x, y in zip(names, sa, sb): print(f"  {n:11s} fresh {x:.4g}  reuse {y:.4g}  ratio {y / x:.3f}")
print("max |ratio-1| = %.3f   (a few %% is fine; tens of %% means reuse is not safe)" % np.max(np.abs(sb / sa - 1)))
print("wall time per event: fresh %.0f s -> reuse %.0f s" % (a["wall"][0], b["wall"][0]))
