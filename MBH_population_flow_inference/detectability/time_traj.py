"""time_traj.py -- quick diagnostic for the forward-FEW generator: how long does one trajectory take, what fraction is accepted
(plunge time in [t_min, t_max]), what fraction is too slow, and where (in lgM / p0 offset) are the slow ones. ~2-4 minutes."""
import argparse, signal, time
import numpy as np
from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_separatrix
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60); ap.add_argument("--per_traj_s", type=int, default=10)
ap.add_argument("--t_min", type=float, default=0.9); ap.add_argument("--t_max", type=float, default=6.2); a = ap.parse_args()
class Slow(Exception): pass
def _al(*_): raise Slow()
signal.signal(signal.SIGALRM, _al)
traj = EMRIInspiral(func="pn5"); rng = np.random.default_rng(3); R = []
for i in range(a.n):
    lgM = rng.uniform(3.9, 7.1); s = rng.uniform(0, .99); Y = rng.uniform(.05, .75); e0 = rng.uniform(0, .95); off = 10 ** rng.uniform(np.log10(.3), np.log10(40))
    p0 = get_separatrix(s, e0, Y) + off; t = time.time(); status = "ok"; d = np.nan
    signal.alarm(a.per_traj_s)
    try: r = traj(10 ** lgM, 10., s, p0, e0, Y, Phi_phi0=0., Phi_theta0=0., Phi_r0=0., dt=10., T=a.t_max + 1.0); d = r[0][-1] / 3.15576e7
    except Slow: status = "slow"
    except Exception as ex: status = "fail:" + str(ex)[:40]
    finally: signal.alarm(0)
    R.append((time.time() - t, lgM, off, d, status)); print(f"{i:3d} lgM={lgM:.2f} a={s:.2f} e0={e0:.2f} dp={off:6.2f}  {R[-1][0]:6.2f}s  dur={d:6.2f}yr  {status}", flush=True)
sec = np.array([r[0] for r in R]); dur = np.array([r[3] for r in R]); st = np.array([r[4] for r in R])
acc = (dur >= a.t_min) & (dur <= a.t_max)
print(f"\nN={a.n}: mean {sec.mean():.2f}s  median {np.median(sec):.2f}s  max {sec.max():.1f}s | accepted {acc.mean():.2f} | slow {np.mean(st=='slow'):.2f} | failed {np.mean([s.startswith('fail') for s in st]):.2f}")
print(f"=> expected cost per ACCEPTED row ~ {sec.sum()/max(acc.sum(),1):.1f}s ; 2000 rows/chunk ~ {2000*sec.sum()/max(acc.sum(),1)/60:.0f} min")
