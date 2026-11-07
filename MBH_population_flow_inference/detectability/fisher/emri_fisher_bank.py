#!/usr/bin/env python
"""
EMRI Fisher bank (GPU): StableEMRIFisher covariances for events drawn from WIDE priors over the event-level
parameters, independent of Lambda, so a conditional flow  C(theta_event)  can be trained once and used for every
universe.

Event parameters come from the forward-integrated p0/e0 trajectory rows (gen_traj_chunk.py): (lgM, a, Y0, e0, p0, T)
are a true FEW inspiral that plunges T years after the start, so no emulator is needed here.  On top of that we draw
  mu = 10 Msun (fixed, as in the old pipeline), z log-uniform in [z_min, z_max] (rejection: optimal SNR > snr_min,
  evaluated with the SNR network, so only events that could possibly be detected cost GPU time),
  sky/spin angles isotropic, three phases uniform.
Fisher (14 params: M mu a p0 e0 Y0 dist qS phiS qK phiK Phi_phi0 Phi_theta0 Phi_r0) at the TRUE parameters -> F1, C1.
C2 step (as in the old TDE pipeline): draw theta_hat ~ N(truth, C1), re-evaluate the Fisher there -> F2, C2.  The
flow is therefore trained on both truth-points and noisy-estimate points.

Output: one npz per chunk.  All arrays have one row per attempted event; ok=0 rows hold NaNs.
Usage:  python emri_fisher_bank.py --chunk 0 --n_events 20 --traj_dir DIR --out_dir DIR
"""
import argparse, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")

p = argparse.ArgumentParser()
p.add_argument("--chunk", type=int, required=True)
p.add_argument("--n_events", type=int, default=20)
p.add_argument("--traj_dir", required=True, help="dir with chunk_*.npz from gen_traj_chunk.py")
p.add_argument("--out_dir", required=True)
p.add_argument("--master_seed", type=int, default=21)
p.add_argument("--z_min", type=float, default=0.01)
p.add_argument("--z_max", type=float, default=8.0)
p.add_argument("--snr_min", type=float, default=5.0, help="optimal-SNR prefilter (data-driven, well below the detection cut)")
p.add_argument("--max_z_tries", type=int, default=300)
p.add_argument("--co_mass", type=float, default=10.0)
p.add_argument("--t_obs", type=float, default=6.0, help="observation window [yr]")
p.add_argument("--t_margin", type=float, default=0.99, help="Fisher T = t_margin * min(T_plunge, t_obs) (avoid the plunge edge)")
p.add_argument("--dt", type=float, default=15.0)
p.add_argument("--der_order", type=int, default=2, help="order of the numerical derivatives (package default 2; its MCMC validation uses 6)")
p.add_argument("--ndelta", type=int, default=8)
p.add_argument("--live_dangerously", type=int, default=0, help="1: skip the stable-delta search (fast, less safe)")
p.add_argument("--target_snr", type=float, default=0.0,
               help=">0: set dist so the Fisher-package SNR is ~target (F = rho^2 Gamma, Gamma independent of distance; the package SNR scales exactly as 1/dist). "
                    "Errors are then SNR-normalised (sigma*rho) and are rescaled with the SNR NETWORK at application time. 0 = physical dist (old behaviour).")
p.add_argument("--proposal", choices=["prior", "mixed"], default="prior",
               help="prior: z log-uniform + trajectory row at random; mixed: a share of events is built from a SOURCE-CATALOGUE EMRI (all universes, equal weight): its (lgM, a, e_plunge) selects the nearest "
                    "forward-integrated trajectory row (-> Y0, e0, p0, T) and its z is used, kept only if the SNR network gives rho > snr_min")
p.add_argument("--catalogue_fraction", type=float, default=0.7)
p.add_argument("--catalogue_file", default="/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference/simulations/consolidated/emri.npz")
p.add_argument("--max_cat_tries", type=int, default=3000)
p.add_argument("--reuse_deltas", type=int, default=1, help="1: C2 reuses the stable deltas found for the first Fisher (the noisy estimate is within ~sigma of the truth); skips ~half the cost")
p.add_argument("--n_jitter", type=int, default=1, help="number of noisy-estimate (C2) re-evaluations per event")
p.add_argument("--max_wall_s", type=float, default=5400)
p.add_argument("--snr_model", default="/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference/models/detectability/emri_snr/model.pth")
a = p.parse_args()

sys.path.insert(0, "/home/2673888s/LISA_selection_function/lisa_snr_package")
os.environ.setdefault("POPLAR_PATH", "/home/2673888s/poplar"); sys.path.insert(0, "/home/2673888s/poplar")
from snr_model import load_snr_model, build_features
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fisher_cov import covariance
from astropy.cosmology import Planck18
from few.waveform import GenerateEMRIWaveform
from few.utils.utility import get_separatrix
from stableemrifisher.fisher import StableEMRIFisher

NAMES = ["M", "mu", "a", "p0", "e0", "Y0", "dist", "qS", "phiS", "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0"]
P = len(NAMES)
snr_net = load_snr_model(a.snr_model)

# ---- pool of true trajectory rows
import glob
rows = [np.load(f) for f in sorted(glob.glob(f"{a.traj_dir}/chunk_*.npz"))]
pool = {k: np.concatenate([r[k] for r in rows]) for k in ["lgM", "a", "Y0", "e0", "p0", "T", "e_plunge"]}
print(f"[pool] {len(pool['lgM'])} trajectory rows from {len(rows)} chunks", flush=True)
rng = np.random.default_rng([a.master_seed, a.chunk])
if a.proposal == "mixed":
    from scipy.spatial import cKDTree
    _x = np.load(a.catalogue_file)["x"]                                   # lgM, a, z, e_plunge  (all universes, equal weight per universe)
    _x = _x[(_x[:, 2] >= a.z_min) & (_x[:, 2] <= a.z_max)]
    _key = lambda lgM, a_, ep: np.stack([lgM / 3.0, a_ / 1.0, ep / 1.0], -1)   # roughly unit scale: lgM spans 3, a and e_plunge span ~1
    TREE = cKDTree(_key(pool["lgM"], pool["a"], pool["e_plunge"]))
    print(f"[proposal] mixed: {len(_x)} catalogue EMRIs, matched to {len(pool['lgM'])} trajectory rows; catalogue_fraction={a.catalogue_fraction}", flush=True)

# use_rk4=True makes sigma_M ~5000x too small (cond ~1e17); see emri_validation_repro.py (configs D vs B)
wave = GenerateEMRIWaveform("Pn5AAKWaveform", return_list=True,
                            inspiral_kwargs=dict(DENSE_STEPPING=0, max_init_len=int(1e7), err=1e-10, use_rk4=False),
                            sum_kwargs=dict(use_gpu=True, pad_output=True), use_gpu=True)


def optimal_snr(lgM, mu, a_, e0, p0, Y0, Tb, z):
    X = build_features(np.array([[10 ** lgM, mu, a_, e0, p0, Y0, Tb]], dtype=float))
    dl = Planck18.luminosity_distance(z).to("Mpc").value
    return float(snr_net.run_on_dataset(X, n_batches=1).cpu().numpy().ravel()[0] / dl)


def finish_event(e, z, snr):
    Tb = min(e["T"], a.t_obs)
    e.update(z=z, snr_net=snr, Tb=Tb, dist=float(Planck18.luminosity_distance(z).to("Gpc").value),
             qS=float(np.arccos(rng.uniform(-1, 1))), phiS=rng.uniform(0, 2 * np.pi),
             qK=float(np.arccos(rng.uniform(-1, 1))), phiK=rng.uniform(0, 2 * np.pi),
             Phi_phi0=rng.uniform(0, 2 * np.pi), Phi_theta0=rng.uniform(0, 2 * np.pi), Phi_r0=rng.uniform(0, 2 * np.pi))
    return e


def draw_catalogue_event():
    """catalogue EMRI -> nearest trajectory row (Y0, e0, p0, T) at the catalogue z; accept if the network SNR clears snr_min"""
    for _ in range(a.max_cat_tries):
        c = _x[rng.integers(len(_x))]
        _, j = TREE.query(_key(c[0], c[1], c[3]), k=8); i = int(rng.choice(j))     # one of the 8 nearest rows, to avoid reusing a single trajectory
        e = {k: float(pool[k][i]) for k in ["lgM", "a", "Y0", "e0", "p0", "T", "e_plunge"]}
        snr = optimal_snr(e["lgM"], a.co_mass, e["a"], e["e0"], e["p0"], e["Y0"], min(e["T"], a.t_obs), float(c[2]))
        if snr > a.snr_min:
            e = finish_event(e, float(c[2]), snr); e["from_catalogue"] = 1.0; return e
    return None


def draw_event():
    if a.proposal == "mixed" and rng.random() < a.catalogue_fraction:
        e = draw_catalogue_event()
        if e is not None:
            return e
    i = rng.integers(len(pool["lgM"]))
    e = {k: float(pool[k][i]) for k in ["lgM", "a", "Y0", "e0", "p0", "T", "e_plunge"]}
    Tb = min(e["T"], a.t_obs)
    for _ in range(a.max_z_tries):
        z = float(np.exp(rng.uniform(np.log(a.z_min), np.log(a.z_max))))
        snr = optimal_snr(e["lgM"], a.co_mass, e["a"], e["e0"], e["p0"], e["Y0"], Tb, z)
        if snr > a.snr_min:
            break
    else:
        return None
    e.update(z=z, snr_net=snr, Tb=Tb, dist=float(Planck18.luminosity_distance(z).to("Gpc").value),
             qS=float(np.arccos(rng.uniform(-1, 1))), phiS=rng.uniform(0, 2 * np.pi),
             qK=float(np.arccos(rng.uniform(-1, 1))), phiK=rng.uniform(0, 2 * np.pi),
             Phi_phi0=rng.uniform(0, 2 * np.pi), Phi_theta0=rng.uniform(0, 2 * np.pi), Phi_r0=rng.uniform(0, 2 * np.pi))
    return e


def vec(e):
    return np.array([10 ** e["lgM"], a.co_mass, e["a"], e["p0"], e["e0"], e["Y0"], e["dist"], e["qS"], e["phiS"],
                     e["qK"], e["phiK"], e["Phi_phi0"], e["Phi_theta0"], e["Phi_r0"]], dtype=float)


def fisher_at(v, Tb, deltas=None):
    sef = StableEMRIFisher(*[np.float64(x) for x in v], dt=a.dt, T=a.t_margin * Tb, EMRI_waveform_gen=wave,
                           param_names=NAMES, Ndelta=a.ndelta, der_order=a.der_order, live_dangerously=bool(a.live_dangerously),
                           deltas=deltas, stats_for_nerds=False, use_gpu=True, filename=None, CovEllipse=False)
    F = np.asarray(sef())
    return F, float(np.sqrt(sef.SNR2)), sef.deltas


def snr_only(v, Tb):
    sef = StableEMRIFisher(*[np.float64(x) for x in v], dt=a.dt, T=a.t_margin * Tb, EMRI_waveform_gen=wave, param_names=NAMES, Ndelta=a.ndelta, der_order=a.der_order,
                           live_dangerously=True, stats_for_nerds=False, use_gpu=True, filename=None, CovEllipse=False)
    return float(sef.SNRcalc_SEF())


def valid(v):
    M, mu, a_, p0, e0, Y0, dist = v[:7]
    if not (M > 0 and mu > 0 and 0 <= a_ < 0.999 and 0 <= e0 < 0.95 and 0 < abs(Y0) < 1 and dist > 0 and p0 > 0):
        return False
    return p0 > get_separatrix(a_, e0, Y0) + 0.05


def pinv_cov(F):
    """covariance via the diagonally-normalised inverse (see fisher_cov.py); returns (C, raw cond).  NOT np.linalg.pinv on the raw matrix."""
    C, _, cond_raw = covariance(F)
    return C, cond_raw


R = dict(from_catalogue=[], e_plunge=[], dist_phys=[], rho_target=[], F1=[], C1=[], theta=[], snr1=[], cond1=[], F2=[], C2=[], theta_hat=[], snr2=[], cond2=[],
         ok=[], ok2=[], wall=[], snr_net=[], Tb=[], z=[])
path = f"{a.out_dir}/chunk_{a.chunk:04d}.npz"; os.makedirs(a.out_dir, exist_ok=True)
nan_mat = np.full((P, P), np.nan); nan_vec = np.full(P, np.nan)


def save():
    np.savez(path, **{k: np.array(v) for k, v in R.items()}, param_names=np.array(NAMES), co_mass=a.co_mass,
             dt=a.dt, ndelta=a.ndelta, live_dangerously=a.live_dangerously, t_margin=a.t_margin)


t_start = time.time()
for n in range(a.n_events):
    if time.time() - t_start > a.max_wall_s:
        print(f"[stop] wall budget reached after {n} events", flush=True); break
    t1 = time.time(); e = draw_event()
    if e is None:
        print(f"[skip] {n}: no z with SNR > {a.snr_min}", flush=True); continue
    v = vec(e); dist_phys = float(v[6]); rho_target = np.nan
    if a.target_snr > 0:
        try:
            rho0 = snr_only(v, e["Tb"]); v[6] = dist_phys * rho0 / a.target_snr; rho_target = a.target_snr   # SNR ~ 1/dist exactly
            print(f"    [target_snr] package SNR at physical dist {rho0:.1f} (network {e['snr_net']:.1f}) -> dist {dist_phys:.4g} -> {v[6]:.4g} Gpc", flush=True)
        except Exception as ex:
            print(f"[fail] {n}: snr_only {type(ex).__name__}: {str(ex)[:150]}", flush=True); continue
    row = dict(F1=nan_mat, C1=nan_mat, snr1=np.nan, cond1=np.nan, F2=nan_mat, C2=nan_mat, theta_hat=nan_vec,
                           snr2=np.nan, cond2=np.nan, ok=0, ok2=0)
    try:
        F1, s1, deltas1 = fisher_at(v, e["Tb"]); C1, c1 = pinv_cov(F1)
        row.update(F1=F1, C1=C1, snr1=s1, cond1=c1, ok=1)
        sig = np.sqrt(np.clip(np.diag(C1), 0, None))
        print(f"[ok] {n} lgM={e['lgM']:.2f} a={e['a']:.2f} e0={e['e0']:.2f} z={e['z']:.3f} Tb={e['Tb']:.2f} snr_net={e['snr_net']:.1f} "
              f"snr_fisher={s1:.1f} sig_lgM={sig[0] / (v[0] * np.log(10)):.3g} cond={c1:.1e} {time.time() - t1:.0f}s", flush=True)
        for _ in range(a.n_jitter):                     # C2: Fisher at a noisy estimate
            th = None
            for _try in range(20):
                if a.target_snr > 0:        # noise draw with the PHYSICAL covariance: all sigma scale by s = rho_fisher / rho_network; dist also by dist_phys/dist_used
                    sc = np.full(P, s1 / e["snr_net"]); sc[6] *= dist_phys / v[6]
                    vp = v.copy(); vp[6] = dist_phys
                    cand = rng.multivariate_normal(vp, C1 * np.outer(sc, sc), check_valid="ignore")
                    if valid(cand): cand[6] = v[6] * cand[6] / dist_phys         # back to the fixed-SNR distance scale for the Fisher evaluation
                else:
                    cand = rng.multivariate_normal(v, C1, check_valid="ignore")
                if valid(cand):
                    th = cand; break
            if th is None:
                print(f"    [jitter] {n}: no valid noisy estimate in 20 draws", flush=True); break
            F2, s2, _ = fisher_at(th, e["Tb"], deltas=deltas1 if a.reuse_deltas else None); C2, c2 = pinv_cov(F2)
            row.update(F2=F2, C2=C2, theta_hat=th, snr2=s2, cond2=c2, ok2=1)
            print(f"    [C2] snr={s2:.1f} cond={c2:.1e} {time.time() - t1:.0f}s", flush=True)
    except Exception as ex:
        print(f"[fail] {n}: {type(ex).__name__}: {str(ex)[:200]}", flush=True)
    for k in ["F1", "C1", "snr1", "cond1", "F2", "C2", "theta_hat", "snr2", "cond2", "ok", "ok2"]: R[k].append(row[k])
    R["from_catalogue"].append(e.get("from_catalogue", 0.0)); R["e_plunge"].append(e["e_plunge"]); R["dist_phys"].append(dist_phys); R["rho_target"].append(rho_target); R["theta"].append(v); R["wall"].append(time.time() - t1); R["snr_net"].append(e["snr_net"]); R["Tb"].append(e["Tb"]); R["z"].append(e["z"])
    save()
    try:      # release cupy's cached blocks between events (long signals grew the pool to ~8 GB and one event hit an OOM on an 11 GB card)
        import cupy as cp; cp.get_default_memory_pool().free_all_blocks(); cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
save()
print(f"wrote {len(R['ok'])} events ({int(np.sum(R['ok']))} ok, {int(np.sum(R['ok2']))} with C2) -> {path}")
