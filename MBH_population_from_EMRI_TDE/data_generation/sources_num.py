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
parser.add_argument("--SEED", type=int, required=False, default=42, help="Random seed for reproducibility")

args = parser.parse_args()

rng = np.random.default_rng(seed=args.SEED)

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

cusp_evolution_object = CuspEvolution(nsc=NSC_obj, compact_object=CO_objs, relaxation=relax_obj,
                                       rate_model=rate_obj, LastMajorMerger=LastMajorMerger(cosmo_model))

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

N_EMRI_events = np.random.poisson(Lambda_EMRI_tot)
N_TDE_events = np.random.poisson(Lambda_TDE_tot)

# ============================================================
# PPS resampling using the combined weight (importance * rate)
# ============================================================
def pps_resample(weights, n_draws, rng):
    w = np.asarray(weights, float)
    total = w.sum()
    if total <= 0 or n_draws <= 0:
        return np.array([], dtype=int)
    p = w / total
    return rng.choice(len(w), size=n_draws, replace=True, p=p)

idx_EMRI = pps_resample(combined_EMRI_weight, N_EMRI_events, rng)
idx_TDE = pps_resample(combined_TDE_weight, N_TDE_events, rng)

print(f"Resampled EMRI catalog size: {len(idx_EMRI)}")
print(f"Resampled TDE catalog size: {len(idx_TDE)}")


# ============================================================
# Save output
# ============================================================
loc = f'{ASTROPHYSICAL_DATA_DIR}/{args.GALAXIES}'
if not os.path.exists(loc):
    os.makedirs(loc)
 
hf = h5py.File(f'{loc}/{MOCK_GALAXY_CATALOG_DATA}_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}_{args.UNIVERSE_MODEL}.h5', 'w')
 
hf.attrs['N_gal_true'] = N_gal_true
hf.attrs['N_objs_sim'] = N_objs_sim
hf.attrs['T_obs_EMRI'] = args.T_OBS_EMRI
hf.attrs['T_obs_TDE'] = args.T_OBS_TDE
hf.attrs['Z_MAX'] = Z_MAX
hf.attrs['Lambda_EMRI_tot'] = Lambda_EMRI_tot
hf.attrs['Lambda_TDE_tot'] = Lambda_TDE_tot
hf.attrs['n_eff_EMRI'] = n_eff_EMRI
hf.attrs['n_eff_TDE'] = n_eff_TDE
hf.attrs['sBH_mass'] = CO_objs.masses['sBH']
hf.attrs['star_mass'] = CO_objs.masses['star']
 
def build_event_catalog(idx, rng):
    if len(idx) == 0:
        return None, None
    lgMgal_event = lgMgal_samples[nucleation_indices][idx]
    z_event = z_nuc[idx]
    nucleation_all_true = np.ones_like(lgMgal_event, dtype=bool)
    occupation_all_true = np.ones_like(lgMgal_event, dtype=bool)

 
    event_gal = Galaxy(lgMgal=lgMgal_event, z_gal=z_event, rng=rng,
                        nucleation_occurs=nucleation_all_true, 
                        occupation_occurs=occupation_all_true,
                        mbh_mass_method=args.mbh_mass_method, 
                        nucleation_method=args.nucleation_method)

    event_nsc = NSC(event_gal)
    event_mbh = MBH_properties(nsc=event_nsc)
    return event_gal, event_mbh
 
 
emri_gal, emri_mbh = build_event_catalog(idx_EMRI, rng)
tde_gal, tde_mbh = build_event_catalog(idx_TDE, rng)

emri_mask = (emri_gal.lgMBH_mass > 4.0) & (emri_gal.lgMBH_mass <= 7.0) if emri_gal is not None else np.array([], dtype=bool) 
tde_mask = (tde_gal.lgMBH_mass > 4.0) & (tde_gal.lgMBH_mass <= 7.0) if tde_gal is not None else np.array([], dtype=bool)
 

# ============================================================
# EMRI eccentricity model
# ============================================================
def emri_parameters(M):
    lgM = np.log10(M)
    alpha1 = max(4.0 + 0.5 * (lgM - 6), 1.0)
    beta1 = max(10.0 - 0.8 * (lgM - 6), 1.0)
    alpha2 = max(3.0 - 0.3 * (lgM - 6), 1.0)
    beta2 = max(6.5 + 0.5 * (lgM - 6), 1.0)
    M0, gamma = 1e5, 1.2
    w = 1.0 / (1.0 + (M0 / M) ** gamma)
    return alpha1, beta1, alpha2, beta2, w


def sample_emri_eccentricity(M, size=1, rng=rng):
    a1, b1, a2, b2, w = emri_parameters(M)
    is_classical = rng.random(size) < w
    e = np.zeros(size)
    n1 = np.sum(is_classical)
    if n1 > 0:
        e[is_classical] = rng.beta(a1, b1, n1)
    n2 = size - n1
    if n2 > 0:
        e[~is_classical] = rng.beta(a2, b2, n2)
    return e if size > 1 else e[0]



grp_emri = hf.create_group('EMRI_catalog')
if emri_gal is not None:
    grp_emri.create_dataset('lgMgal', data=emri_gal.lgMgal[emri_mask])
    grp_emri.create_dataset('z_gal', data=emri_gal.z_gal[emri_mask])
    grp_emri.create_dataset('lgMBH', data=emri_gal.lgMBH_mass[emri_mask])
    grp_emri.create_dataset('sigma_km_s', data=emri_gal.sigma_km_s[emri_mask])
    grp_emri.create_dataset('ra_deg', data=emri_gal.ra_deg[emri_mask])
    grp_emri.create_dataset('dec_deg', data=emri_gal.dec_deg[emri_mask])
    grp_emri.create_dataset('initial_MBHspin', data=emri_mbh.initial_MBHspin[emri_mask])
    grp_emri.create_dataset('sBH', data=CO_objs.masses['sBH'] * np.ones(np.sum(emri_mask)))
    grp_emri.create_dataset('star_mass', data=CO_objs.masses['star'] * np.ones(np.sum(emri_mask)))
 
    eccentricities_at_plunge = np.array(
        [sample_emri_eccentricity(10 ** lgM, rng=rng) for lgM in emri_gal.lgMBH_mass[emri_mask]]
    )
    grp_emri.create_dataset('eccentricities_at_plunge', data=eccentricities_at_plunge)
else:
    for name in ['lgMgal', 'z_gal', 'lgMBH', 'sigma_km_s', 'ra_deg', 'dec_deg',
                 'initial_MBHspin', 'eccentricities_at_plunge', 'star_mass', 'sBH']:
        grp_emri.create_dataset(name, data=np.array([]))
 
grp_tde = hf.create_group('TDE_catalog')
if tde_gal is not None:
    grp_tde.create_dataset('lgMgal', data=tde_gal.lgMgal[tde_mask])
    grp_tde.create_dataset('z_gal', data=tde_gal.z_gal[tde_mask])
    grp_tde.create_dataset('lgMBH', data=tde_gal.lgMBH_mass[tde_mask])
    grp_tde.create_dataset('sigma_km_s', data=tde_gal.sigma_km_s[tde_mask])
    grp_tde.create_dataset('ra_deg', data=tde_gal.ra_deg[tde_mask])
    grp_tde.create_dataset('dec_deg', data=tde_gal.dec_deg[tde_mask])
    grp_tde.create_dataset('initial_MBHspin', data=tde_mbh.initial_MBHspin[tde_mask])
    grp_tde.create_dataset('star_mass', data=CO_objs.masses['star'] * np.ones(np.sum(tde_mask)))
    grp_tde.create_dataset('sBH', data=CO_objs.masses['sBH'] * np.ones(np.sum(tde_mask)))
    grp_tde.create_dataset('t0_mjd_transient', data=np.random.uniform(58200, 64423, size=np.sum(tde_mask)))
else:
    for name in ['lgMgal', 'z_gal', 'lgMBH', 'sigma_km_s', 'ra_deg', 'dec_deg', 'initial_MBHspin', 'star_mass', 'sBH', 't0_mjd_transient']:
        grp_tde.create_dataset(name, data=np.array([]))
 
hf.close()

print(f"After resampling and mass cutoff (4.0 < log10(M_BH/Msun) < 7.0 ), EMRI catalog size: {np.sum(emri_mask)}")
print(f"After resampling and mass cutoff (4.0 < log10(M_BH/Msun) < 7.0), TDE catalog size: {np.sum(tde_mask)}")


import matplotlib.pyplot as plt

if emri_gal is not None and np.sum(emri_mask) > 0:
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].hist(emri_gal.lgMBH_mass[emri_mask], bins=30, alpha=0.5, color='blue', density=True, label='MBH Mass')
    axs[0, 1].hist(emri_gal.z_gal[emri_mask], bins=30, alpha=0.5, color='blue', density=True, label='Redshift')
    axs[1, 0].hist(emri_mbh.initial_MBHspin[emri_mask], bins=30, alpha=0.5, color='blue', density=True, label='Initial BH Spin')
    axs[1, 1].hist(eccentricities_at_plunge, bins=30, alpha=0.5, color='blue', density=True, label='Velocity Dispersion')
    axs[0, 0].set_xlabel(r'$\log_{10}(M_{\rm BH}/M_\odot)$')
    axs[0, 1].set_xlabel(r'$z$')
    axs[1, 0].set_xlabel(r'$a$')
    axs[1, 1].set_xlabel(r'$e_{\rm plunge}$')
    plt.suptitle(f'EMRI generated = {np.sum(emri_mask)}')
    plt.savefig(f'{loc}/emri_catalog_histograms_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}_{args.UNIVERSE_MODEL}.png')
    plt.close()
else:
    print(f"[WARN] EMRI catalog is empty (0 events after resampling+mass cut) -- "
          f"skipping EMRI diagnostic plot. Check Lambda_EMRI_tot printed above: "
          f"if that's already << 1, the shortfall is in the RATE, not detection.")

if tde_gal is not None and np.sum(tde_mask) > 0:
    fig, axs = plt.subplots(1, 3, figsize=(12, 10))
    axs[0].hist(tde_gal.lgMBH_mass[tde_mask], bins=30, alpha=0.5, color='orange', density=True, label='MBH Mass')
    axs[1].hist(tde_gal.z_gal[tde_mask], bins=30, alpha=0.5, color='orange', density=True, label='Redshift')
    axs[2].hist(tde_mbh.initial_MBHspin[tde_mask], bins=30, alpha=0.5, color='orange', density=True, label='Initial BH Spin')
    axs[0].set_xlabel(r'$\log_{10}(M_{\rm BH}/M_\odot)$')
    axs[1].set_xlabel(r'$z$')
    axs[2].set_xlabel(r'$a$')
    plt.suptitle(f'TDEs generated = {np.sum(tde_mask)}')
    plt.savefig(f'{loc}/tde_catalog_histograms_{args.mbh_mass_method}_{args.nucleation_method}_{args.occupation_method}_{args.UNIVERSE_MODEL}.png')
    plt.close()
else:
    print(f"[WARN] TDE catalog is empty (0 events after resampling+mass cut) -- "
          f"skipping TDE diagnostic plot. Check Lambda_TDE_tot printed above: "
          f"if that's already << 1, the shortfall is in the RATE, not detection.")