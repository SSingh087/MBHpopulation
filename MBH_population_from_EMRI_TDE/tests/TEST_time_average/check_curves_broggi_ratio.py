"""Same as check_curves.py, using the ratio to the Broggi et al. rates."""
# (2) Curve check: does integral_0^t_f Gamma(t) dt from our fitted UniversalRate * (Gamma_hat, t_E) reproduce Broggi+22 Table N_x?
# (1) Self-consistent averaging: rate averaged over 0..t_f from the SAME curves (no change of normalisation), vs. rate at the observed cusp age.
# Run from repo root.
import sys, numpy as np
sys.path.insert(0, 'astrophysical_setup')
from scipy.integrate import quad
from rate import UniversalRate as U, ConstantRate as C
from config import MBH_sigma0, BROGGI_SIGMA0, BROGGI_MBH_PIVOT, BROGGI_INV_SLOPE

MBH = 10 ** C._log10_MBH_grid
sig0 = BROGGI_SIGMA0 * (MBH / BROGGI_MBH_PIVOT) ** BROGGI_INV_SLOPE
t_f = np.minimum(10 * C._t_E, 10.0)                                    # Gyr
res = {}
print("MBH     s/s0  | t_E fit/table | N_E fit/table | N_T fit/table | <fT>/fT(tau_obs=1) ... see bottom")
for i, M in enumerate(MBH):
    for j, r in enumerate(C._sigma_ratio_grid):
        s = r * sig0[i]
        GE = 6.2e-6 * M ** -0.25 * (r) ** 3.09           # 1/yr
        GT = 1.6e-4 * M ** -0.22 * (r) ** 1.16
        tE = 6.4e-9 * M ** 1.29 * (r) ** -2.97           # Gyr
        tau_f = t_f[i, j] / tE
        IE = quad(lambda x: float(U.EMRI_rate(x)), 0, tau_f, limit=200)[0]
        IT = quad(lambda x: float(U.TDE_rate(x)), 0, tau_f, limit=200)[0]
        NE, NT = GE * tE * 1e9 * IE, GT * tE * 1e9 * IT
        res[i, j] = dict(tE=tE / C._t_E[i, j], NE=NE / C._N_E[i, j], NT=NT / C._N_T[i, j], tau_f=tau_f, IE=IE, IT=IT, GE=GE, GT=GT)
        print(f"{M:8.0e} {r:5.2f} | {res[i,j]['tE']:8.2f}      | {res[i,j]['NE']:8.2f}      | {res[i,j]['NT']:8.2f}      | tau_f={tau_f:6.1f}")
for k in ('tE', 'NE', 'NT'):
    v = np.array([res[key][k] for key in res]); print(f"{k}: median ratio fit/table = {np.median(v):.2f}, 16-84% = [{np.percentile(v,16):.2f}, {np.percentile(v,84):.2f}]")

# (1) shape-only effect of averaging, for the paper's Fig-6 curves: <f>_0^tau_f vs f(tau) at several ages
print("\nShape-only: time-average <f> over 0..tau_f vs instantaneous f(tau)")
for tf in (3, 10, 30):
    aT = quad(lambda x: float(U.TDE_rate(x)), 0, tf)[0] / tf; aE = quad(lambda x: float(U.EMRI_rate(x)), 0, tf)[0] / tf
    print(f"tau_f={tf:3d}: <f_TDE>={aT:.3f} <f_EMRI>={aE:.3f} | ratio avg TDE/EMRI={aT/aE:.2f}", end=" | ")
    for t in (0.3, 1, 3, 10):
        print(f"tau={t}: [TDE/EMRI]_inst={float(U.TDE_rate(t)/U.EMRI_rate(t)):.2f} boost(avg/inst)={aT/aE/float(U.TDE_rate(t)/U.EMRI_rate(t)):.2f}", end="; ")
    print()
