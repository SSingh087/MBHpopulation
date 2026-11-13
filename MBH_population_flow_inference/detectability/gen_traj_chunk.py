"""gen_traj_chunk.py -- training data for the (e_plunge, T) -> (e0, p0) emulator, from FORWARD FEW integrations.

Why forward: the old backward integration (EMRIInspiral(integrate_backwards=True)) does nothing in the available FEW
(it returns p0 = separatrix, e0 = e_plunge). So: draw (lgM, a, Y0, e0) and a target plunge time T_target ~ U(t_min, t_max) yr,
then BISECT the initial p0 (offset above the separatrix) until the forward inspiral reaches plunge after ~T_target.
This gives ~100% acceptance at every mass (a fixed offset range could not reach short/long plunge times for light/heavy systems).
Row = (lgM, a, Y0, e_plunge, T_to_plunge) -> (e0, p0).

Robustness: the integrator is run in a child process; if it produces nothing for --row_timeout_s the parent kills and restarts it
(a signal-based timeout does not work because FEW's C code does not return to Python). A chunk also stops after --max_wall_s.
"""
import argparse, os, time
import numpy as np
import multiprocessing as mp

YR = 3.15576e7

def child(q, seed, a):
    from few.trajectory.inspiral import EMRIInspiral
    from few.utils.utility import get_separatrix
    rng = np.random.default_rng(seed); traj = EMRIInspiral(func="pn5"); cap = a.t_max + 1.0
    def run(lgM, spin, Y0, e0, off):
        p0 = get_separatrix(spin, e0, Y0) + off
        t, p, e, *_ = traj(10 ** lgM, 10.0, spin, p0, e0, Y0, Phi_phi0=0., Phi_theta0=0., Phi_r0=0., dt=a.dt, T=cap)
        return t[-1] / YR, e[-1], p0, p[-1]
    while True:
        lgM = rng.uniform(3.9, 7.1); spin = rng.uniform(0.0, 0.99); Y0 = rng.uniform(0.05, 0.75); e0 = rng.uniform(0.0, 0.95)
        Tt = rng.uniform(a.t_min, a.t_max); lo, hi = np.log10(0.3), np.log10(a.max_offset); res = None; status = "nobracket"
        try:
            for _ in range(a.n_bisect):
                mid = 0.5 * (lo + hi); d, ep, p0, pp = run(lgM, spin, Y0, e0, 10 ** mid)
                if abs(d - Tt) < a.tol * Tt: res = (lgM, spin, Y0, ep, d, e0, p0, pp); status = "ok"; break
                if d > Tt: hi = mid          # plunges too late (or hit the cap) -> smaller offset
                else: lo = mid
            else:
                if a.t_min <= d <= a.t_max: res = (lgM, spin, Y0, ep, d, e0, p0, pp); status = "ok"
        except Exception:
            status = "fail"
        q.put((status, res))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, required=True); ap.add_argument("--n_keep", type=int, default=500)
    ap.add_argument("--out_dir", required=True); ap.add_argument("--master_seed", type=int, default=7)
    ap.add_argument("--t_min", type=float, default=0.9); ap.add_argument("--t_max", type=float, default=6.2)
    ap.add_argument("--dt", type=float, default=300.0, help="output sampling step (s); only the end state is used")
    ap.add_argument("--max_offset", type=float, default=500.0); ap.add_argument("--n_bisect", type=int, default=14); ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--row_timeout_s", type=int, default=180, help="kill+restart the integrator child if no row for this long")
    ap.add_argument("--max_wall_s", type=int, default=5400)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True); fn = f"{a.out_dir}/chunk_{a.chunk:04d}.npz"
    if os.path.exists(fn): print("exists, skipping", fn); raise SystemExit
    ctx = mp.get_context("fork"); rows = []; counts = dict(ok=0, nobracket=0, fail=0, hang=0); t0 = time.time(); restarts = 0
    def start():
        global restarts
        q = ctx.Queue(); pr = ctx.Process(target=child, args=(q, [a.master_seed, a.chunk, restarts], a), daemon=True); pr.start(); restarts += 1; return q, pr
    q, pr = start()
    while len(rows) < a.n_keep and time.time() - t0 < a.max_wall_s:
        try: status, res = q.get(timeout=a.row_timeout_s)
        except Exception:
            counts["hang"] += 1; pr.kill(); pr.join(); q, pr = start(); continue
        counts[status] += 1
        if status == "ok": rows.append(res)
        if sum(counts.values()) % 50 == 0: print(f"  rows {len(rows)} {counts} {time.time()-t0:.0f}s", flush=True)
    pr.kill(); pr.join()
    r = np.array(rows, dtype=np.float64).reshape(-1, 8)
    np.savez(fn, lgM=r[:, 0], a=r[:, 1], Y0=r[:, 2], e_plunge=r[:, 3], T=r[:, 4], e0=r[:, 5], p0=r[:, 6], p_plunge=r[:, 7], **counts)
    print(f"chunk {a.chunk}: kept {len(r)} rows; {counts}; {time.time()-t0:.0f}s")
