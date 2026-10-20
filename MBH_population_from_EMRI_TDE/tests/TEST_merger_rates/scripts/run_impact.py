"""Rerun of the event-count pipeline with a chosen galaxy merger-rate model, to measure its impact on EMRI/TDE counts."""
# AUTO-GENERATED from data_generation/sources_num.py (lines 1-248) with the merger model made selectable.
# Physics/sampling identical to the repo; only LastMajorMerger is swapped and extra diagnostics are dumped.
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
from merger_models import make_models
import os, sys, argparse
sys.path.insert(0, os.path.abspath('./astrophysical_setup'))
sys.path.insert(0, os.path.abspath('./'))

from utils import ASTROPHYSICAL_DATA_DIR, MOCK_GALAXY_CATALOG_DATA

import numpy as np
from scipy.interpolate import interp1d
from galaxy import Galaxy
from nsc import NSC, CompactObject, MBH_properties
from density import DehnenProfile
from relaxation import RelaxationModel
from rate import RateModel, ConstantRate
from evolution import CuspEvolution
from cosmology import LastMajorMerger, CosmologyModel, GalaxyStellarMassFunction, MBHMassFunction
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("--UNIVERSE_MODEL", type=int, required=True, help="Universe model index")
parser.add_argument("--GALAXIES", type=int, required=True,
                     help="Monte Carlo sample size (NOT the literal real galaxy count)")
parser.add_argument("--Z_MAX", type=float, required=False, default=10.0, help="Maximum redshift")
parser.add_argument("--T_OBS_EMRI", type=float, required=True)
parser.add_argument("--T_OBS_TDE",  type=float, required=True)
parser.add_argument("--mbh_mass_method", type=str, required=True, default="use_sigma", choices=["use_sigma", "use_lgMgal"])
parser.add_argument("--nucleation_method", type=str, required=True, default="hannah", choices=["hannah", "neumayer"])
parser.add_argument("--occupation_method", type=str, required=True, default="uniform", choices=["uniform", "varying"])
parser.add_argument("--MERGER_MODEL", type=str, default="baseline")
parser.add_argument("--OUT", type=str, default=None)
parser.add_argument("--N_GRID", type=int, default=None, help="override LastMajorMerger.n_grid (repo=100)")
parser.add_argument("--KAPPA", type=float, default=None, help="override kappa_cusp (repo=0.25)")
parser.add_argument("--DUMP", type=str, default=None, help="save per-galaxy arrays to this .npz")
parser.add_argument("--SEED", type=int, required=False, default=42, help="Random seed for reproducibility")

args = parser.parse_args()

rng = np.random.default_rng(seed=args.SEED)
np.random.seed(args.SEED)  # LastMajorMerger.sample_z_LMM uses the global RNG

cosmo_model = CosmologyModel()

Z_MAX = args.Z_MAX
Z_MIN = 1e-5

N_objs_sim = args.GALAXIES
T_obs_EMRI_Gyr = args.T_OBS_EMRI / 1e9
T_obs_TDE_Gyr  = args.T_OBS_TDE  / 1e9

Rmin = cosmo_model.comoving_distance(Z_MIN)
Rmax = cosmo_model.comoving_distance(Z_MAX)

GSMF = GalaxyStellarMassFunction()

# ============================================================
# Shared z-grid table: used for BOTH N_gal_true AND the
# mass-integrated density n(z) needed for importance weights.
# Built once, reused everywhere -- avoids redundant integration.
# ============================================================
def build_density_tables(gsmf, z_min, z_max, n_z=600, n_points_mass=5000):
    z_table = np.linspace(z_min, z_max, n_z)
    lgM_grid, log10phi_grid = gsmf.get_gsmf(z_table, n_points_mass=n_points_mass)
    phi_grid = np.nan_to_num(10.0 ** log10phi_grid, nan=0.0, posinf=0.0, neginf=0.0)

    n_density_table = np.trapz(phi_grid, lgM_grid, axis=1)  # Mpc^-3, shape (n_z,)

    n_density_interp = interp1d(z_table, n_density_table, kind='linear',
                                 bounds_error=False,
                                 fill_value=(n_density_table[0], n_density_table[-1]))
    return z_table, n_density_table, n_density_interp



# ============================================================
# Step 1: Pilot run to find the lgMgal range that maps to the
# calibrated rate range lgMBH ∈ [4, 7].
#
# Why: lgMBH is derived from lgMgal via M-sigma (with scatter),
# so there is no clean analytic inversion. A small pilot sample
# of ~20k galaxies is cheap and gives a robust empirical mapping.
# The main run then concentrates all N_objs_sim samples in the
# productive lgMgal band, boosting N_eff by ~60x.
# ============================================================
def find_lgMgal_rate_range(GSMF, cosmo_model, Rmin, Rmax, rng, lgMBH_min=4.0, lgMBH_max=7.0, N_pilot=100000):
    """
    Returns (lgM_low, lgM_high): the lgMgal sampling range that
    maps to lgMBH ∈ [lgMBH_min, lgMBH_max] under the M-sigma
    relation. Uses 0.5th / 99.5th percentiles to be robust
    against scatter tails without discarding real contributors.
    """
    lgM_min_full = GSMF.lgMgal_data.min()
    lgM_max_full = GSMF.lgMgal_data.max()

    # --- pilot z samples (same scheme as main run) ---
    u_p = rng.random(N_pilot)
    r_p = (Rmin**3 + (Rmax**3 - Rmin**3) * u_p) ** (1/3)
    z_p = cosmo_model.z_from_comoving_distance_fast(r_p)

    # --- pilot mass samples: uniform over full range ---
    lgMgal_p    = rng.uniform(lgM_min_full, lgM_max_full, size=N_pilot)
    nuc_idx_p   = Galaxy.check_nucleation(lgMgal_p, nucleation_method=args.nucleation_method)
    gal_p       = Galaxy(lgMgal=lgMgal_p, z_gal=z_p, nucleation_occurs=nuc_idx_p, mbh_mass_method=args.mbh_mass_method, nucleation_method=args.nucleation_method, occupation_method=args.occupation_method)

    lgMBH_p     = gal_p.lgMBH_mass                    # nucleated subset
    lgMgal_nuc_p = lgMgal_p[nuc_idx_p]

    in_range = (lgMBH_p >= lgMBH_min) & (lgMBH_p <= lgMBH_max) & gal_p.occupation_occurs

    if in_range.sum() < 50:
        raise RuntimeError(
            f"Only {in_range.sum()} pilot galaxies in lgMBH [{lgMBH_min},{lgMBH_max}]. "
            f"Increase N_pilot (currently {N_pilot}).")

    # Percentiles rather than min/max — robust against scatter outliers
    lgM_low  = np.percentile(lgMgal_nuc_p[in_range], 0.01)
    lgM_high = np.percentile(lgMgal_nuc_p[in_range], 99.99)

    print(f"[Pilot] lgMBH ∈ [{lgMBH_min}, {lgMBH_max}] maps to "
          f"lgMgal ∈ [{lgM_low:.3f}, {lgM_high:.3f}]  "
          f"({in_range.sum()} / {len(lgMBH_p)} nucleated pilot galaxies in range)")

    return lgM_low, lgM_high


# ============================================================
# Step 2: Shared density tables — built over FULL lgMgal range.
# N_gal_true is a property of the universe, not of our sampler.
# ============================================================
z_table, n_density_table, n_density_at_z_interp = build_density_tables(GSMF, Z_MIN, Z_MAX)
dVc_dz_table = cosmo_model.dVc_dz(z_table)
N_gal_true   = np.trapz(n_density_table * dVc_dz_table, z_table)
print(f"N_gal_true (real galaxy count out to z={Z_MAX}): {N_gal_true:.6e}")

# ============================================================
# Step 3: z sampling — unchanged, proportional to dVc/dz.
# ============================================================
u        = rng.random(N_objs_sim)
r_samples = (Rmin**3 + (Rmax**3 - Rmin**3) * u) ** (1/3)
z_grid   = cosmo_model.z_from_comoving_distance_fast(r_samples)

# ============================================================
# Step 4: Restricted lgMgal proposal.
# Only sample the band that produces lgMBH ∈ [4, 7].
# The importance weight absorbs the narrower integration width.
# ============================================================
lgM_min_full = GSMF.lgMgal_data.min()
lgM_max_full = GSMF.lgMgal_data.max()

lgM_low, lgM_high = find_lgMgal_rate_range(GSMF, cosmo_model, Rmin, Rmax, rng,
                lgMBH_min=4.0, lgMBH_max=7.0, N_pilot=100000)

lgMgal_samples = rng.uniform(lgM_low, lgM_high, size=N_objs_sim)
q_lgM = 1.0 / (lgM_high - lgM_low)    # updated proposal density assuming uniform sampling over the restricted band

# Importance weight: only the integration width changes, N_gal_true (the full-universe count) is untouched.
log10phi_pointwise = GSMF.interp(np.column_stack([z_grid, lgMgal_samples]))
phi_pointwise = np.nan_to_num(10.0 ** log10phi_pointwise, nan=0.0, posinf=0.0, neginf=0.0)

V_tot = np.trapz(dVc_dz_table, z_table)
mc_weight_i = phi_pointwise * V_tot * (lgM_high - lgM_low) / N_objs_sim

print(f"Proposal range: lgMgal ∈ [{lgM_low:.3f}, {lgM_high:.3f}]  "
      f"(was [{lgM_min_full:.2f}, {lgM_max_full:.2f}])")
print(f"mc_weight_i range: [{mc_weight_i.min():.3e}, {mc_weight_i.max():.3e}], "
      f"median {np.median(mc_weight_i):.3e}")

# ============================================================
# Physics pipeline
# ============================================================

nucleation_indices = Galaxy.check_nucleation(lgMgal_samples, nucleation_method=args.nucleation_method)
gal_obj = Galaxy(lgMgal=lgMgal_samples, z_gal=z_grid, nucleation_occurs=nucleation_indices, mbh_mass_method=args.mbh_mass_method, nucleation_method=args.nucleation_method, occupation_method=args.occupation_method)

NSC_obj = NSC(gal_obj)
MBH_obj = MBH_properties(nsc=NSC_obj)

CO_objs = CompactObject(nsc=NSC_obj, masses={'sBH': 10.0, 'star': 1.0},
                         total_mass_in_MBH_mass_units={'sBH': 0.2, 'star': 20.0},
                         types_CO=['sBH', 'star'], types_masses='same_mass', type_CO_limits=None)

dehnen_obj = DehnenProfile(nsc=NSC_obj, compact_object=CO_objs)
relax_obj = RelaxationModel(nsc=NSC_obj, compact_object=CO_objs, profile=dehnen_obj)
rate_obj = RateModel(nsc=NSC_obj)
const_rate_obj  = ConstantRate(nsc=NSC_obj)

merger_obj = make_models(cosmo_model)[args.MERGER_MODEL]
if args.N_GRID is not None:
    merger_obj.n_grid = args.N_GRID
if hasattr(merger_obj, "set_mass"):
    merger_obj.set_mass(lgMgal_samples[nucleation_indices])
_kw = {} if args.KAPPA is None else dict(kappa=args.KAPPA)
cusp_evolution_object = CuspEvolution(nsc=NSC_obj, compact_object=CO_objs, relaxation=relax_obj,
                                       rate_model=rate_obj, LastMajorMerger=merger_obj, **_kw)

cusp_evolution_object.cusp_turn_on_time(kvir=1.0, force_resample=True)

T_c_Gyr = np.atleast_1d(cusp_evolution_object.cusp_age(kvir=1.0))

t_rlx_Gyr = relax_obj.t_relax_at_rinfl(kvir=1.0, unit='Gyr')

active = (T_c_Gyr < t_rlx_Gyr).astype(float)

MBH_MASK = (gal_obj.lgMBH_mass > 4.0) & (gal_obj.lgMBH_mass <= 7.0) & gal_obj.occupation_occurs

rate_EMRI_Gyr = const_rate_obj.constant_rate(kind='EMRI')   # 0.0 outside [1e4,1e7]
rate_TDE_Gyr  = const_rate_obj.constant_rate(kind='TDE')

z_gal = NSC_obj.gal.z_gal

Gamma_EMRI_i = np.where(MBH_MASK, rate_EMRI_Gyr * T_obs_EMRI_Gyr * active / (1.0 + z_gal), 0.0)
Gamma_TDE_i  = np.where(MBH_MASK, rate_TDE_Gyr  * T_obs_TDE_Gyr  * active / (1.0 + z_gal), 0.0)

mc_weight_i_nuc = mc_weight_i[nucleation_indices] # only keep the importance weights for galaxies that actually nucleate

z_nuc = z_grid[nucleation_indices]
mask_EMRI_extra = MBH_MASK & (z_nuc < 8.0)
mask_TDE_extra = MBH_MASK & (z_nuc < 5.0)

print(f"Active fraction (cusp young enough): {active.mean():.3f}")
print(f"Lambda_TDE before active mask: {np.sum(rate_TDE_Gyr * T_obs_TDE_Gyr * mc_weight_i_nuc / (1.0 + z_gal)):.1f}")
print(f"Lambda_TDE after  active mask: {np.sum(Gamma_TDE_i * mc_weight_i_nuc):.1f}")
print(f"Lambda_EMRI before active mask: {np.sum(rate_EMRI_Gyr * T_obs_EMRI_Gyr * mc_weight_i_nuc / (1.0 + z_gal)):.1f}")
print(f"Lambda_EMRI after  active mask: {np.sum(Gamma_EMRI_i * mc_weight_i_nuc):.1f}")

# ============================================================
# Combined weights (importance weight * rate) and N_eff check
# ============================================================
combined_EMRI_weight = mc_weight_i_nuc * Gamma_EMRI_i
combined_TDE_weight = mc_weight_i_nuc * Gamma_TDE_i

def effective_sample_size(weights):
    w = np.asarray(weights, float)
    s = w.sum()
    if s <= 0:
        return 0.0
    return (s ** 2) / np.sum(w ** 2)


n_eff_EMRI = effective_sample_size(combined_EMRI_weight)
n_eff_TDE = effective_sample_size(combined_TDE_weight)
print(f"N_eff (EMRI): {n_eff_EMRI:.2f} out of {N_objs_sim:.0e} simulated galaxies")
print(f"N_eff (TDE):  {n_eff_TDE:.2f} out of {N_objs_sim:.0e} simulated galaxies")


# Fraction of weight now in calibrated range
w_in_range = mc_weight_i_nuc[MBH_MASK].sum()
w_total    = mc_weight_i_nuc.sum()
print(f"Fraction of GSMF weight in lgMBH [4,7]: {w_in_range/w_total:.3f}")
print(f"N_eff (EMRI): {n_eff_EMRI:.0f}")
print(f"N_eff (TDE):  {n_eff_TDE:.0f}")
print(f"N_eff / N_sim: {n_eff_EMRI/N_objs_sim:.4f} (EMRI),  {n_eff_TDE/N_objs_sim:.4f} (TDE)")

Lambda_EMRI_tot = np.sum(combined_EMRI_weight)
Lambda_TDE_tot = np.sum(combined_TDE_weight)
Lambda_EMRI_zcut = np.sum(combined_EMRI_weight[mask_EMRI_extra])
Lambda_TDE_zcut = np.sum(combined_TDE_weight[mask_TDE_extra])

print(f"Total expected EMRIs, 4.0 < log10(M_BH/Msun) < 7.0: {Lambda_EMRI_tot:.3f}")
print(f"Total expected TDEs, 4.0 < log10(M_BH/Msun) < 7.0: {Lambda_TDE_tot:.3f}")
print(f"Total expected EMRIs, z < 8.0 and 4.0 < log10(M_BH/Msun) < 7.0: {Lambda_EMRI_zcut:.3f}")
print(f"Total expected TDEs,  z < 5.0 and  4.0 < log10(M_BH/Msun) < 7.0: {Lambda_TDE_zcut:.3f}")
import json
z_LMM = np.atleast_1d(cusp_evolution_object.z_LMM)
res = dict(model=args.MERGER_MODEL, seed=args.SEED, galaxies=int(N_objs_sim),
    Lambda_TDE=float(Lambda_TDE_tot), Lambda_EMRI=float(Lambda_EMRI_tot),
    Lambda_TDE_zcut=float(Lambda_TDE_zcut), Lambda_EMRI_zcut=float(Lambda_EMRI_zcut),
    active_fraction=float(active.mean()),
    active_frac_in_MBH_mask=float(active[MBH_MASK].mean()) if MBH_MASK.any() else None,
    censored_frac_zLMM_at_zmax=float(np.mean(z_LMM >= cusp_evolution_object.LastMajorMerger.z_max - 1e-9)),
    median_cusp_age_Gyr=float(np.median(T_c_Gyr)), median_t_relax_Gyr=float(np.median(t_rlx_Gyr)),
    n_eff_TDE=float(n_eff_TDE), n_eff_EMRI=float(n_eff_EMRI))
# per-redshift-bin TDE expectation (sum of importance*rate weights)
zb = np.linspace(0, args.Z_MAX, 17)
res["z_bins"] = zb.tolist()
res["Lambda_TDE_per_zbin"] = [float(np.sum(combined_TDE_weight[(z_nuc >= a) & (z_nuc < b)])) for a, b in zip(zb[:-1], zb[1:])]
res["Lambda_EMRI_per_zbin"] = [float(np.sum(combined_EMRI_weight[(z_nuc >= a) & (z_nuc < b)])) for a, b in zip(zb[:-1], zb[1:])]
print(json.dumps({k: v for k, v in res.items() if not k.startswith(("z_bins", "Lambda_TDE_per", "Lambda_EMRI_per"))}, indent=1))
if args.DUMP:
    np.savez_compressed(args.DUMP, lgMgal=lgMgal_samples[nucleation_indices], lgMBH=gal_obj.lgMBH_mass, occ=gal_obj.occupation_occurs,
        z=z_gal, w=mc_weight_i_nuc, T_c=T_c_Gyr, t_rlx=t_rlx_Gyr, z_LMM=z_LMM, t_on=cusp_evolution_object._t_on_cache,
        t_obs=np.atleast_1d(cusp_evolution_object.t_obs_at_z_obs), t_LMM=np.atleast_1d(cusp_evolution_object.t_LMM),
        active=active, mask=MBH_MASK, G_TDE=Gamma_TDE_i, G_EMRI=Gamma_EMRI_i, rate_TDE=np.atleast_1d(rate_TDE_Gyr))
if args.OUT:
    json.dump(res, open(args.OUT, "w"), indent=1)
