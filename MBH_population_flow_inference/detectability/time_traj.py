"""time_traj.py -- quick diagnostic for the forward-FEW generator (parallel over --workers processes): per-trajectory time,
acceptance (plunge time in [t_min, t_max]), fraction too slow, and where the slow ones are. ~1-2 minutes."""
import argparse, signal, time
import numpy as np
import multiprocessing as mp
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=400); ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--per_traj_s", type=int, default=10); ap.add_argument("--t_min", type=float, default=0.9); ap.add_argument("--t_max", type=float, default=6.2)
ap.add_argument("--verbose", action="store_true"); a = ap.parse_args()
class Slow(Exception): pass
def _al(*_): raise Slow()

def work(i):
    from few.trajectory.inspiral import EMRIInspiral
    from few.utils.utility import get_separatrix
    global _traj
    if "_traj" not in globals(): _traj = EMRIInspiral(func="pn5")
    signal.signal(signal.SIGALRM, _al)
    rng = np.random.default_rng([3, i])
    lgM = rng.uniform(3.9, 7.1); s = rng.uniform(0, .99); Y = rng.uniform(.05, .75); e0 = rng.uniform(0, .95); off = 10 ** rng.uniform(np.log10(.3), np.log10(40))
    p0 = get_separatrix(s, e0, Y) + off; t = time.time(); status = "ok"; d = np.nan
    signal.alarm(a.per_traj_s)
    try: r = _traj(10 ** lgM, 10., s, p0, e0, Y, Phi_phi0=0., Phi_theta0=0., Phi_r0=0., dt=10., T=a.t_max + 1.0); d = r[0][-1] / 3.15576e7
    except Slow: status = "slow"
    except Exception as ex: status = "fail"
    finally: signal.alarm(0)
    return time.time() - t, lgM, s, e0, off, d, status

if __name__ == "__main__":
    t0 = time.time()
    with mp.get_context("fork").Pool(a.workers) as pool: R = pool.map(work, range(a.n), chunksize=1)
    wall = time.time() - t0
    sec, lgM, sp, e0, off, dur = (np.array([r[k] for r in R]) for k in range(6)); st = np.array([r[6] for r in R])
    if a.verbose:
        for i, r in enumerate(R): print(f"{i:3d} lgM={r[1]:.2f} a={r[2]:.2f} e0={r[3]:.2f} dp={r[4]:6.2f} {r[0]:6.2f}s dur={r[5]:6.2f}yr {r[6]}")
    acc = (dur >= a.t_min) & (dur <= a.t_max)
    print(f"N={a.n} on {a.workers} workers, wall {wall:.0f}s | per-trajectory: mean {sec.mean():.2f}s median {np.median(sec):.2f}s max {sec.max():.1f}s")
    print(f"accepted {acc.mean():.3f} | too slow (> {a.per_traj_s}s) {np.mean(st=='slow'):.3f} | failed {np.mean(st=='fail'):.3f}")
    print(f"=> cost per ACCEPTED row ~ {sec.sum()/max(acc.sum(),1):.1f} CPU-s ; a 2000-row chunk ~ {2000*sec.sum()/max(acc.sum(),1)/60:.0f} min")
    print("time by lgM bin (mean s | accepted frac):")
    for lo in np.arange(3.9, 7.1, 0.8):
        m = (lgM >= lo) & (lgM < lo + 0.8)
        if m.any(): print(f"  lgM {lo:.1f}-{lo+0.8:.1f}: {sec[m].mean():5.2f}s | acc {acc[m].mean():.2f} | slow {np.mean(st[m]=='slow'):.2f}  (n={m.sum()})")
    print("time by p0-offset bin (mean s | accepted frac):")
    for lo, hi in [(0.3, 1), (1, 3), (3, 10), (10, 40.1)]:
        m = (off >= lo) & (off < hi)
        if m.any(): print(f"  dp {lo}-{hi}: {sec[m].mean():5.2f}s | acc {acc[m].mean():.2f} | slow {np.mean(st[m]=='slow'):.2f}  (n={m.sum()})")
