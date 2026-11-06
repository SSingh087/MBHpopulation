"""
source_sim.py -- importable EMRI + TDE *source* simulator  Lambda -> source populations (no detectability).

Re-implements the forward model of the (read-only) EMRI_TDE_population pipeline
(data_generation/sources_num.py) as a function of Lambda = (A, B, scatter, logM0, delta, theta).
It IMPORTS the old astrophysical_setup modules (cosmology, NSC, relaxation, evolution, rate) unchanged;
only the pieces that must depend on Lambda are re-written here:

  * M-sigma relation with free A, B, scatter            (old: config.py constants)
  * occupation fraction with free (logM0, delta, theta) (old: defaults only)
  * lgMBH / sigma / Re drawn ONCE and carried through to the saved events (old: redrawn after PPS)
  * one seeded numpy Generator for everything we control (old: global np.random mixed in)
    -- the old LastMajorMerger sampler still uses global np.random, so we also np.random.seed(seed).

Fixed for v1 (CONFIG.md): use_sigma, hannah nucleation, oleary21 merger model, T_obs=6 yr, z<8,
4 < lgMBH <= 7, sBH = 10 Msun.  NO detectability here (that is Part 2).
"""
import os, sys, time
import numpy as np

OLD_ROOT = "/home/2673888s/EMRI_TDE_population"          # read-only reference
_OLD_ASTRO = os.path.join(OLD_ROOT, "astrophysical_setup")

# import the old modules without modifying them
_saved_path = list(sys.path)
sys.path.insert(0, _OLD_ASTRO)
import config as _oldcfg                                  # noqa: E402  (old astrophysical_setup/config.py)
import galaxy as _oldgal                                  # noqa: E402
from galaxy import Galaxy                                 # noqa: E402
from nsc import NSC, CompactObject                        # noqa: E402
from density import DehnenProfile                         # noqa: E402
from relaxation import RelaxationModel                    # noqa: E402
from rate import RateModel, ConstantRate                  # noqa: E402
from evolution import CuspEvolution                       # noqa: E402
from cosmology import LastMajorMerger, CosmologyModel, GalaxyStellarMassFunction  # noqa: E402
sys.path[:] = _saved_path

# ---------------------------------------------------------------- fixed v1 settings
FIXED = dict(
    mbh_mass_method="use_sigma", nucleation_method="hannah", merger_model="oleary21",
    T_OBS_EMRI_YR=6.0, T_OBS_TDE_YR=17.25, Z_MIN=1e-5, Z_MAX=8.0, LGMBH_MIN=4.0, LGMBH_MAX=7.0, SIGMA_0=160.0,
    SBH_MASS=10.0,
)
LAMBDA_NAMES = ["A", "B", "scatter", "logM0", "delta", "theta"]
FIDUCIAL = dict(A=7.87, B=4.55, scatter=0.55, logM0=9.46, delta=3.91, theta=0.80)
PRIOR_BOX = dict(A=(7.57, 8.17), B=(3.25, 5.85), scatter=(0.35, 0.75),
                 logM0=(8.5, 10.5), delta=(0.5, 20.0), theta=(0.2, 2.0))   # delta is log-uniform
EVENT_KEYS = ["lgMBH", "spin", "z", "e_plunge", "lgMgal", "sigma_km_s"]       # EMRI sources
TDE_KEYS = ["lgMBH", "z", "lgMgal", "sigma_km_s"]                               # TDE sources (spin not modelled)


def sample_prior(rng):
    """One Lambda draw from the v1 prior box (delta log-uniform, rest uniform)."""
    lam = {}
    for k, (lo, hi) in PRIOR_BOX.items():
        lam[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if k == "delta" else float(rng.uniform(lo, hi))
    return lam


def f_occ(lgMgal, logM0, delta, theta):
    """Occupation fraction, Zou+2025 form (same expression as galaxy._f_occupation_varying_)."""
    return (1.0 + delta * np.exp(-theta * delta * (lgMgal - logM0))) ** (-1.0 / delta)


# ---------------------------------------------------------------- shared (Lambda independent) tables
class _Shared:
    """Built once per process: cosmology, GSMF, z grid, volume element, merger model."""
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self):
        t0 = time.time()
        self.cosmo = CosmologyModel()
        self.gsmf = GalaxyStellarMassFunction()
        self.Rmin = self.cosmo.comoving_distance(FIXED["Z_MIN"])
        self.Rmax = self.cosmo.comoving_distance(FIXED["Z_MAX"])
        z_table = np.linspace(FIXED["Z_MIN"], FIXED["Z_MAX"], 600)
        self.V_tot = np.trapz(self.cosmo.dVc_dz(z_table), z_table)     # comoving volume, Mpc^3
        self.lgM_full = (float(self.gsmf.lgMgal_data.min()), float(self.gsmf.lgMgal_data.max()))
        self.lmm = LastMajorMerger(self.cosmo, merger_model=FIXED["merger_model"])
        self.build_time = time.time() - t0

    def draw_z(self, n, rng):
        u = rng.random(n)
        r = (self.Rmin ** 3 + (self.Rmax ** 3 - self.Rmin ** 3) * u) ** (1.0 / 3.0)
        return self.cosmo.z_from_comoving_distance_fast(r)


# ---------------------------------------------------------------- galaxy-level forward model
def _sigma_from_lgM(lg, rng):
    """One draw of (Re scatter) -> sigma in km/s and cgs, as in old Galaxy.R_eff + Galaxy.sigma (kvir=1)."""
    lgRe = 0.24 * (lg - 10.7) + 0.82 + rng.normal(0.0, 0.20, size=lg.shape)
    Re_cm = 10.0 ** lgRe * _oldcfg.kpc_to_cm
    M_enc_g = 0.5 * 10.0 ** lg * _oldcfg.Msun_to_grams
    return np.sqrt(_oldcfg.G_cgs * M_enc_g / Re_cm)                    # cm/s


def _galaxy_properties(lgMgal, lam, rng, legacy=False, force_nuc=False, force_occ=False):
    """Nucleation (hannah, fixed) -> Re -> sigma -> lgMBH (M-sigma with Lambda) -> occupation (Lambda).
    Returns dict of arrays restricted to the NUCLEATED subset, plus the nucleation mask.

    legacy=True reproduces the OLD pipeline's quirk that Galaxy.lgMBH() calls self.sigma() again, i.e. the sigma
    used in the M-sigma relation is an INDEPENDENT second draw of the size scatter, different from the stored
    sigma that the rate model / relaxation use.  legacy=False (default): one sigma, used consistently.
    force_nuc / force_occ: used when rebuilding event galaxies (old build_event_catalog sets both True)."""
    if force_nuc:
        nuc = np.ones(lgMgal.shape, dtype=bool)
    else:
        nuc = rng.uniform(size=lgMgal.shape) < _oldgal._f_NSC_Hannah_(lgMgal)
    lg = lgMgal[nuc]
    sig_cgs = _sigma_from_lgM(lg, rng)
    sigma_km_s = sig_cgs / 1e5
    sigma_for_mbh = _sigma_from_lgM(lg, rng) / 1e5 if legacy else sigma_km_s
    lgMBH = lam["A"] + lam["B"] * np.log10(sigma_for_mbh / FIXED["SIGMA_0"]) + rng.normal(0.0, lam["scatter"], size=lg.shape)
    if force_occ:
        occ = np.ones(lg.shape, dtype=bool)
    else:
        occ = rng.uniform(size=lg.shape) < f_occ(lg, lam["logM0"], lam["delta"], lam["theta"])
    return dict(nuc=nuc, lgMgal=lg, sigma_km_s=sigma_km_s,
                sigma_pc_yr=sig_cgs * _oldcfg.sec_per_year / _oldcfg.pc_to_cm,
                lgMBH=lgMBH, occ=occ)


class _LGalaxy(Galaxy):
    """Galaxy container with the attributes the old NSC/Rate/Relaxation/Evolution classes read.
    Bypasses Galaxy.__init__ so nothing is redrawn."""
    def __init__(self, z, props, lam, rng):
        self.rng = rng
        self.z_gal = z
        self.lgMgal = props["lgMgal"]
        self.sigma_km_s = props["sigma_km_s"]
        self.sigma_pc_yr = props["sigma_pc_yr"]
        self.lgMBH_mass = props["lgMBH"]
        self.occupation_occurs = props["occ"]
        self.nucleation_occurs = np.ones_like(props["occ"], dtype=bool)
        self.MBH_A, self.MBH_B, self.MBH_scatter = lam["A"], lam["B"], lam["scatter"]


def _find_lgMgal_band(S, lam, rng, n_pilot, legacy=False):
    """Same idea as the old find_lgMgal_rate_range: lgMgal band that maps to lgMBH in (4,7] and occupied,
    using 0.01/99.99 percentiles (a proposal-efficiency device; the importance weight absorbs it)."""
    lo, hi = S.lgM_full
    lg = rng.uniform(lo, hi, size=n_pilot)
    pr = _galaxy_properties(lg, lam, rng, legacy=legacy)
    ok = (pr["lgMBH"] > FIXED["LGMBH_MIN"]) & (pr["lgMBH"] <= FIXED["LGMBH_MAX"]) & pr["occ"]
    if ok.sum() < 50:
        return None
    return float(np.percentile(pr["lgMgal"][ok], 0.01)), float(np.percentile(pr["lgMgal"][ok], 99.99))


def _eccentricity(lgMBH, rng):
    """Mass-dependent Beta mixture, identical to sources_num.emri_parameters/sample_emri_eccentricity, vectorised."""
    d = lgMBH - 6.0
    a1 = np.maximum(4.0 + 0.5 * d, 1.0); b1 = np.maximum(10.0 - 0.8 * d, 1.0)
    a2 = np.maximum(3.0 - 0.3 * d, 1.0); b2 = np.maximum(6.5 + 0.5 * d, 1.0)
    w = 1.0 / (1.0 + (1e5 / 10.0 ** lgMBH) ** 1.2)
    classical = rng.random(lgMBH.shape) < w
    return np.where(classical, rng.beta(a1, b1), rng.beta(a2, b2))


def _spin(lgMBH, rng):
    """Initial MBH spin, identical to nsc.MBH_properties.initial_MBHspin, with our rng."""
    lo = lgMBH < 6
    alpha = np.where(lo, 2.5 + 0.1 * (lgMBH - 6.0), 7.0 + 0.1 * (lgMBH - 6.0))
    return rng.beta(alpha, 5.0)


# ---------------------------------------------------------------- main entry point
def simulate_sources(lam, seed, n_pool=100_000, n_pilot=100_000, return_pool_diagnostics=False, legacy=False,
                     max_tde_store=20_000):
    """
    Simulate one universe's EMRI source catalogue (all sources, no detectability cut).

    lam   : dict with keys A, B, scatter, logM0, delta, theta
    seed  : int, controls everything (also seeds global np.random for the old LMM sampler)
    Returns dict:
        events      : dict of 1-D arrays (EVENT_KEYS), length n_events
        Lambda_tot  : expected number of EMRIs, sum_i w_i Gamma_i  (this is mu(Lambda) before detection)
        n_events    : Poisson draw actually realised
        n_eff       : effective sample size of the combined weights
        timings     : dict
    """
    t_start = time.time()
    S = _Shared.get()
    rng = np.random.default_rng(seed)
    np.random.seed(seed % (2 ** 32 - 1))      # old cosmology.LastMajorMerger uses global np.random

    band = _find_lgMgal_band(S, lam, rng, n_pilot, legacy=legacy)
    if band is None:
        return dict(events={k: np.array([]) for k in EVENT_KEYS}, Lambda_tot=0.0, n_events=0, n_eff=0.0,
                    band=None, timings=dict(total=time.time() - t_start), failed="pilot_empty")
    lgM_lo, lgM_hi = band
    t_pilot = time.time()

    # --- pool: z ~ dVc/dz, lgMgal uniform in band, importance weights (GSMF phi * V * width / N)
    z = S.draw_z(n_pool, rng)
    lgMgal = rng.uniform(lgM_lo, lgM_hi, size=n_pool)
    log10phi = S.gsmf.interp(np.column_stack([z, lgMgal]))
    phi = np.nan_to_num(10.0 ** log10phi, nan=0.0, posinf=0.0, neginf=0.0)
    mc_weight = phi * S.V_tot * (lgM_hi - lgM_lo) / n_pool

    props = _galaxy_properties(lgMgal, lam, rng, legacy=legacy)   # default: drawn ONCE and carried through
    nuc = props["nuc"]
    z_nuc = z[nuc]
    w_nuc = mc_weight[nuc]

    gal = _LGalaxy(z_nuc, props, lam, rng)
    nsc = NSC(gal)
    co = CompactObject(nsc=nsc, masses={"sBH": FIXED["SBH_MASS"], "star": 1.0},
                       total_mass_in_MBH_mass_units={"sBH": 0.2, "star": 20.0},
                       types_CO=["sBH", "star"], types_masses="same_mass", type_CO_limits=None)
    dehnen = DehnenProfile(nsc=nsc, compact_object=co)
    relax = RelaxationModel(nsc=nsc, compact_object=co, profile=dehnen)
    rate = RateModel(nsc=nsc)
    const_rate = ConstantRate(nsc=nsc)
    cusp = CuspEvolution(nsc=nsc, compact_object=co, relaxation=relax, rate_model=rate, LastMajorMerger=S.lmm)
    cusp.cusp_turn_on_time(kvir=1.0, force_resample=True)
    T_c_Gyr = np.atleast_1d(cusp.cusp_age(kvir=1.0))
    t_rlx_Gyr = relax.t_relax_at_rinfl(kvir=1.0, unit="Gyr")
    active = (T_c_Gyr < t_rlx_Gyr).astype(float)

    mbh_mask = (props["lgMBH"] > FIXED["LGMBH_MIN"]) & (props["lgMBH"] <= FIXED["LGMBH_MAX"]) & props["occ"]
    rate_gyr = const_rate.constant_rate(kind="EMRI")
    T_gyr = FIXED["T_OBS_EMRI_YR"] / 1e9
    Gamma = np.where(mbh_mask, rate_gyr * T_gyr * active / (1.0 + z_nuc), 0.0)
    weight = w_nuc * Gamma
    # TDEs: SAME galaxies, same cusp draw / active mask / M-sigma values (consistent physics), TDE rate and T_obs
    rate_tde_gyr = const_rate.constant_rate(kind="TDE")
    Gamma_T = np.where(mbh_mask, rate_tde_gyr * (FIXED["T_OBS_TDE_YR"] / 1e9) * active / (1.0 + z_nuc), 0.0)
    weight_T = w_nuc * Gamma_T
    Lambda_tot_tde = float(weight_T.sum())
    n_eff_tde = float(weight_T.sum() ** 2 / np.sum(weight_T ** 2)) if Lambda_tot_tde > 0 else 0.0
    Lambda_tot = float(weight.sum())
    n_eff = float(weight.sum() ** 2 / np.sum(weight ** 2)) if Lambda_tot > 0 else 0.0
    t_phys = time.time()

    n_events = int(rng.poisson(Lambda_tot)) if Lambda_tot > 0 else 0
    if n_events > 0:
        idx = rng.choice(len(weight), size=n_events, replace=True, p=weight / weight.sum())
        if legacy:
            # old build_event_catalog: rebuild galaxies for the selected events (fresh Re/sigma/M-sigma scatter,
            # nucleation and occupation forced True), then keep only 4 < lgMBH <= 7
            ev = _galaxy_properties(props["lgMgal"][idx], lam, rng, legacy=True, force_nuc=True, force_occ=True)
            keep = (ev["lgMBH"] > FIXED["LGMBH_MIN"]) & (ev["lgMBH"] <= FIXED["LGMBH_MAX"])
            lgMBH_e = ev["lgMBH"][keep]; z_e = z_nuc[idx][keep]
            lgMgal_e = ev["lgMgal"][keep]; sig_e = ev["sigma_km_s"][keep]
        else:
            lgMBH_e = props["lgMBH"][idx]; z_e = z_nuc[idx]
            lgMgal_e = props["lgMgal"][idx]; sig_e = props["sigma_km_s"][idx]
        events = dict(lgMBH=lgMBH_e, spin=_spin(lgMBH_e, rng), z=z_e,
                      e_plunge=_eccentricity(lgMBH_e, rng), lgMgal=lgMgal_e, sigma_km_s=sig_e)
    else:
        events = {k: np.array([]) for k in EVENT_KEYS}

    # --- TDE events: Poisson count, PPS resample, (legacy: redraw + mask), store at most max_tde_store
    n_tde_poisson = int(rng.poisson(Lambda_tot_tde)) if Lambda_tot_tde > 0 else 0
    tde = {k: np.array([]) for k in TDE_KEYS}
    n_tde_total = 0
    if n_tde_poisson > 0:
        idt = rng.choice(len(weight_T), size=n_tde_poisson, replace=True, p=weight_T / weight_T.sum())
        if legacy:
            ev = _galaxy_properties(props["lgMgal"][idt], lam, rng, legacy=True, force_nuc=True, force_occ=True)
            keep = (ev["lgMBH"] > FIXED["LGMBH_MIN"]) & (ev["lgMBH"] <= FIXED["LGMBH_MAX"])
            tde = dict(lgMBH=ev["lgMBH"][keep], z=z_nuc[idt][keep], lgMgal=ev["lgMgal"][keep], sigma_km_s=ev["sigma_km_s"][keep])
        else:
            tde = dict(lgMBH=props["lgMBH"][idt], z=z_nuc[idt], lgMgal=props["lgMgal"][idt], sigma_km_s=props["sigma_km_s"][idt])
        n_tde_total = int(len(tde["z"]))
        if n_tde_total > max_tde_store:                      # random subsample (iid events, so this loses nothing)
            sub = rng.choice(n_tde_total, size=max_tde_store, replace=False)
            tde = {k: v[sub] for k, v in tde.items()}

    out = dict(events=events, tde_events=tde, Lambda_tot_tde=Lambda_tot_tde, n_tde_total=n_tde_total,
               n_tde_poisson=n_tde_poisson, n_eff_tde=n_eff_tde, Lambda_tot=Lambda_tot, n_events=int(len(events['z'])), n_events_poisson=n_events, legacy=legacy, n_eff=n_eff,
               band=(lgM_lo, lgM_hi), active_frac=float(active.mean()), n_nucleated=int(nuc.sum()),
               timings=dict(pilot=t_pilot - t_start, physics=t_phys - t_pilot,
                            total=time.time() - t_start, shared_build=S.build_time))
    if return_pool_diagnostics:
        out["pool"] = dict(z=z_nuc, w=weight, lgMBH=props["lgMBH"], occ=props["occ"])
    return out


if __name__ == "__main__":
    t = time.time()
    r = simulate_sources(FIDUCIAL, seed=0)
    print({k: v for k, v in r.items() if k not in ("events", "tde_events", "pool")})
    print("n events", r["n_events"], "wall", time.time() - t)
