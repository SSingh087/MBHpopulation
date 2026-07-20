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

rng = np.random.default_rng(seed=42)

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES", type=int, required=True,
                     help="Monte Carlo sample size (NOT the literal real galaxy count)")
parser.add_argument("--Z_MAX", type=float, required=False, default=10.0, help="Maximum redshift")
parser.add_argument("--T_OBS_EMRI", type=float, required=True)
parser.add_argument("--T_OBS_TDE",  type=float, required=True)
parser.add_argument("--mbh_mass_method", type=str, required=False, default="M_sigma", choices=["use_sigma", "use_lgMgal"])
parser.add_argument("--nucleation_method", type=str, required=False, default="hannah", choices=["hannah", "neumayer"])
args = parser.parse_args()

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
def build_density_tables(gsmf, z_min, z_max, n_z=300, n_points_mass=1000):
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
def find_lgMgal_rate_range(GSMF, cosmo_model, Rmin, Rmax, rng, lgMBH_min=4.0, lgMBH_max=7.0, N_pilot=20000):
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
    nuc_idx_p   = Galaxy.check_nucleation(lgMgal_p, z_p, nucleation_method=args.nucleation_method)
    gal_p       = Galaxy(lgMgal=lgMgal_p, z_gal=z_p, nucleation_occurs=nuc_idx_p, mbh_mass_method=args.mbh_mass_method, nucleation_method=args.nucleation_method)

    lgMBH_p     = gal_p.lgMBH_mass                    # nucleated subset
    lgMgal_nuc_p = lgMgal_p[nuc_idx_p]

    in_range = (lgMBH_p >= lgMBH_min) & (lgMBH_p <= lgMBH_max)

    if in_range.sum() < 50:
        raise RuntimeError(
            f"Only {in_range.sum()} pilot galaxies in lgMBH [{lgMBH_min},{lgMBH_max}]. "
            f"Increase N_pilot (currently {N_pilot}).")

    # Percentiles rather than min/max — robust against scatter outliers
    lgM_low  = np.percentile(lgMgal_nuc_p[in_range], 0.05)
    lgM_high = np.percentile(lgMgal_nuc_p[in_range], 99.95)

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
                lgMBH_min=4.0, lgMBH_max=7.0, N_pilot=20000)

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

nucleation_indices = Galaxy.check_nucleation(lgMgal_samples, z_grid, nucleation_method=args.nucleation_method)
gal_obj = Galaxy(lgMgal=lgMgal_samples, z_gal=z_grid, nucleation_occurs=nucleation_indices, mbh_mass_method=args.mbh_mass_method, nucleation_method=args.nucleation_method)

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

MBH_MASK = (gal_obj.lgMBH_mass > 4.0) & (gal_obj.lgMBH_mass <= 7.0)

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

# ΛEMRI​=∫∫ΓEMRI​(M,z)⋅ϕ(M,z)⋅dz/dVc * ​​dz * dlogM
# ΓEMRI​(M,z) - expected EMRIs from one galaxy of mass M at redshift z during T\mathrm{obs}
# ϕ(M,z)\phi(M, z) - GSMF: how many such galaxies exist per unit volume per unit log⁡M\log M
# dVc​/dz — how much comoving volume sits at redshift z


# ΛEMRI​≈i∑​Gamma_EMRI_iΓEMRI,i​​​×mc_weight_iϕ(Mi​,zi​)⋅Vtot​⋅ΔlogM/N​
# np.sum(combined_EMRI_weight)


# # Also check the top contributors
# top_idx = np.argsort(Gamma_TDE_i * mc_weight_i_nuc)[-10:]
# print("Top 10 TDE weight contributors:")
# for i in top_idx:
#     print(f"  lgMBH={gal_obj.lgMBH_mass[i]:.2f}, z={z_nuc[i]:.3f}, "
#           f"weight={mc_weight_i_nuc[i]:.2e}, Gamma={Gamma_TDE_i[i]:.2e}, "
#           f"product={Gamma_TDE_i[i]*mc_weight_i_nuc[i]:.2e}, active={active[i]:.0f}")



# print(f"Fraction of nucleated galaxies with lgMBH > 4: {MBH_MASK.mean():.3f}")
# print(f"Fraction of weighted mass in lgMBH > 4: "
#       f"{(mc_weight_i_nuc[MBH_MASK]).sum() / mc_weight_i_nuc.sum():.3f}")



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
 
hf = h5py.File(f'{loc}/{MOCK_GALAXY_CATALOG_DATA}_{args.mbh_mass_method}_{args.nucleation_method}.h5', 'w')
 
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
 
# ============================================================
# Build FRESH host-galaxy realizations for the resampled catalog rows.
#
# Why: idx_EMRI / idx_TDE come from PPS resampling WITH REPLACEMENT, so
# the same proxy-pool index can appear multiple times. Slicing directly
# into gal_obj.*/MBH_obj.* would give those repeated rows IDENTICAL
# sigma_km_s, lgMBH, ra_deg, dec_deg, spin -- as if several real events
# came from the exact same point in the sky with the exact same BH mass.
# In reality each repeat almost always represents a DIFFERENT real
# galaxy that happens to share the same (z, lgMgal) patch.
#
# Fix: only (z, lgMgal) -- the rate-determining variables tied to the
# weighting -- stay fixed to the resampled values. Everything that's
# genuinely independent per real galaxy (the M-sigma/size-mass scatter,
# sky position, spin) is re-drawn fresh by constructing a NEW Galaxy/
# NSC/MBH_properties instance from the resampled (z, lgMgal) pairs --
# Galaxy.__init__ draws independent scatter/sky-location per array
# element, even for repeated (z, lgMgal) inputs.
#
# NOTE: Galaxy.R_eff() and Galaxy.lgMBH() draw their scatter via the
# global np.random (not the rng object passed into the constructor) --
# only sky_location() respects the passed-in rng. So this step is
# reproducible only insofar as the GLOBAL numpy random state is fixed
# (e.g. via np.random.seed(...)) -- worth knowing if you need exact
# reproducibility, since `rng` here is the local Generator instance and
# won't control those two draws.
# ============================================================

def build_event_catalog(idx, rng):
    if len(idx) == 0:
        return None, None
    lgMgal_event = lgMgal_samples[nucleation_indices][idx]
    z_event = z_nuc[idx]
    nucleation_all_true = np.ones_like(lgMgal_event, dtype=bool)
 
    event_gal = Galaxy(lgMgal=lgMgal_event, z_gal=z_event, rng=rng,
                        nucleation_occurs=nucleation_all_true)
    event_nsc = NSC(event_gal)
    event_mbh = MBH_properties(nsc=event_nsc)
    return event_gal, event_mbh
 
 
emri_gal, emri_mbh = build_event_catalog(idx_EMRI, rng)
tde_gal, tde_mbh = build_event_catalog(idx_TDE, rng)

# reapply the MBH mass cut to the resampled catalogs, since the PPS resampling can draw some low-mass galaxies that were originally filtered out

# This cutoff needs to be consistent iwth the training model. This also needs to be changed to MBH mass to 4.

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
    # also save for t0_MJD for all the TDE events, since this is needed for the transient model
    # this should account for ZTF to LSST observation dates which is 
    # 1 Mar 2018 (MJD 58178) → Dec 2026 (MJD 61396) for ZTF
    # 1 Jan 2023 (MJD 59953) → 31 Dec 2025 (MJD 61396) for LSST
    # giving a total of 58200 to 64423 for the transient model to sample from
    
    grp_tde.create_dataset('t0_mjd_transient', data=np.random.uniform(58200, 64423, size=np.sum(tde_mask)))
else:
    for name in ['lgMgal', 'z_gal', 'lgMBH', 'sigma_km_s', 'ra_deg', 'dec_deg', 'initial_MBHspin', 'star_mass', 'sBH', 't0_mjd_transient']:
        grp_tde.create_dataset(name, data=np.array([]))
 
hf.close()

print(f"After resampling and mass cutoff (4.0 < log10(M_BH/Msun) < 7.0 ), EMRI catalog size: {np.sum(emri_mask)}")
print(f"After resampling and mass cutoff (4.0 < log10(M_BH/Msun) < 7.0), TDE catalog size: {np.sum(tde_mask)}")
print(f"Saved catalog to {loc}/data_cusp_evolution_{args.mbh_mass_method}_{args.nucleation_method}.h5")



# Plot histograms of the EMRI and TDE catalogs and all the parameters that are saved in the h5 file. This is to check if the distributions are reasonable and to check if the resampling is working correctly.
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

# All parameters saved per catalog, with axis labels
EMRI_PARAMS = {
    "lgMBH_mass"              : r"$\log_{10}(M_{\rm BH}/M_\odot)$",
    "z_gal"                   : r"$z$",
    "initial_MBHspin"         : r"$a$",
    "sigma_km_s"              : r"$\sigma\ [\rm km/s]$",
    "ra_deg"                  : r"RA [deg]",
    "dec_deg"                 : r"Dec [deg]",
    "lgMgal"                  : r"$\log_{10}(M_{\rm gal}/M_\odot)$",
    "eccentricities_at_plunge": r"$e_{\rm plunge}$",
    "star_mass"               : r"$m_\star\ [M_\odot]$",
    "sBH"                     : r"$m_{\rm sBH}\ [M_\odot]$",
}
TDE_PARAMS = {
    "lgMBH_mass"      : r"$\log_{10}(M_{\rm BH}/M_\odot)$",
    "z_gal"           : r"$z$",
    "initial_MBHspin" : r"$a$",
    "sigma_km_s"      : r"$\sigma\ [\rm km/s]$",
    "ra_deg"          : r"RA [deg]",
    "dec_deg"         : r"Dec [deg]",
    "lgMgal"          : r"$\log_{10}(M_{\rm gal}/M_\odot)$",
    "t0_mjd_transient": r"$t_0\ [\rm MJD]$",
    "star_mass"        : r"$m_\star\ [M_\odot]$",
    "sBH"             : r"$m_{\rm sBH}\ [M_\odot]$",
}

# Build data dicts — only non-empty catalogs
emri_vals = {}
if emri_gal is not None and emri_mask.sum() > 0:
    emri_vals = {
        "lgMBH_mass"              : emri_gal.lgMBH_mass[emri_mask],
        "z_gal"                   : emri_gal.z_gal[emri_mask],
        "initial_MBHspin"         : emri_mbh.initial_MBHspin[emri_mask],
        "sigma_km_s"              : emri_gal.sigma_km_s[emri_mask],
        "ra_deg"                  : emri_gal.ra_deg[emri_mask],
        "dec_deg"                 : emri_gal.dec_deg[emri_mask],
        "lgMgal"                  : emri_gal.lgMgal[emri_mask],
        "eccentricities_at_plunge": eccentricities_at_plunge,
        "star_mass"               : CO_objs.masses['star'] * np.ones(emri_mask.sum()),
        "sBH"                     : CO_objs.masses['sBH']  * np.ones(emri_mask.sum()),
    }

tde_vals = {}
if tde_gal is not None and tde_mask.sum() > 0:
    tde_vals = {
        "lgMBH_mass"      : tde_gal.lgMBH_mass[tde_mask],
        "z_gal"           : tde_gal.z_gal[tde_mask],
        "initial_MBHspin" : tde_mbh.initial_MBHspin[tde_mask],
        "sigma_km_s"      : tde_gal.sigma_km_s[tde_mask],
        "ra_deg"          : tde_gal.ra_deg[tde_mask],
        "dec_deg"         : tde_gal.dec_deg[tde_mask],
        "lgMgal"          : tde_gal.lgMgal[tde_mask],
        "t0_mjd_transient": np.random.uniform(58200, 64423, size=tde_mask.sum()),
        "star_mass"        : CO_objs.masses['star'] * np.ones(tde_mask.sum()),
        "sBH"             : CO_objs.masses['sBH']  * np.ones(tde_mask.sum()),
    }

# All unique params across both catalogs
all_params = dict.fromkeys(list(EMRI_PARAMS.keys()) + list(TDE_PARAMS.keys()))
n_params   = len(all_params)
ncols      = 4
nrows      = (n_params + ncols - 1) // ncols


# ── Theoretical MF overlay ─────────────────────────────────────────────────
# Marginalize MBHMF over z weighted by dVc/dz to get one smooth curve.
# This is the "truth" line the catalog should track.
# def compute_marginalized_mbhmf(MBHMF, cosmo_model, z_table, dVc_dz_table,
#                                 lgMBH_range=(4.0, 7.0), n_points=200):
#     """
#     Returns (lgMBH_grid, phi_marginalised):
#       phi_marginalised[j] = integral_z  phi_MBHMF(lgMBH_j, z) * dVc/dz  dz
#     normalised so that the integral over lgMBH equals 1.
#     """
#     lgMBH_grid = np.linspace(lgMBH_range[0], lgMBH_range[1], n_points)
#     phi_marg   = np.zeros(n_points)

#     for iz, z in enumerate(z_table):
#         lgMBH_z, phi_z = MBHMF.get_mbhmf(z_gal=np.array([z]), n_points_mass=n_points)
#         # lgMBH_z shape: (1, n_points) or (n_points,) — squeeze
#         lgMBH_z = lgMBH_z.squeeze()
#         phi_z   = 10.0 ** phi_z.squeeze()       # log10(phi) → phi
#         phi_z   = np.nan_to_num(phi_z, nan=0.0, posinf=0.0, neginf=0.0)

#         # Interpolate onto common lgMBH_grid
#         phi_interp = np.interp(lgMBH_grid, lgMBH_z, phi_z, left=0.0, right=0.0)
#         phi_marg  += phi_interp * dVc_dz_table[iz]

#     # Normalise to unit area (so it overlays a density=True histogram)
#     norm = np.trapz(phi_marg, lgMBH_grid)
#     if norm > 0:
#         phi_marg /= norm
#     return lgMBH_grid, phi_marg


# ── Proxy pool overlays (consistency check against MC noise) ──────────────
def _proxy_overlay(ax, proxy_lgMBH, weights, bins, color, label, ls='--'):
    w = np.asarray(weights, float)
    w = np.where(np.isfinite(w), w, 0.0)
    if w.sum() <= 0:
        return
    # Normalise to density so it's comparable with density=True histogram
    ax.hist(proxy_lgMBH, bins=bins, weights=w / w.sum(),
            histtype='step', linewidth=1.4, linestyle=ls,
            color=color, label=label, zorder=2)


def _plot_one_catalog(catalog_name, params_dict, vals_dict, n_events,
                      proxy_lgMBH, w_gsmf, w_rate, color, loc,
                      lgMBH_theory=None, phi_theory=None):
    keys  = list(params_dict.keys())
    ncols = 4
    nrows = (len(keys) + ncols - 1) // ncols
    mbh_bins = np.linspace(4.0, 7.0, 31)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.2))
    axes = axes.flatten()

    for i, key in enumerate(keys):
        ax    = axes[i]
        label = params_dict[key]

        if key not in vals_dict or len(vals_dict[key]) == 0:
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                    ha='center', va='center', color='grey', fontsize=10)
            ax.set_xlabel(label, fontsize=11)
            continue

        data = vals_dict[key]

        if key == 'lgMBH_mass':
            # ── Catalog histogram ──────────────────────────────────────────
            ax.hist(data, bins=mbh_bins, histtype='stepfilled',
                    color=color, alpha=0.35, density=True,
                    label=f'{catalog_name} events (N={n_events})', zorder=3)
            ax.hist(data, bins=mbh_bins, histtype='step',
                    color=color, linewidth=1.8, density=True, zorder=4)

            # ── Theoretical MBHMF (marginalised over z) ───────────────────
            # if lgMBH_theory is not None:
            #     ax.plot(lgMBH_theory, phi_theory,
            #             color='black', linewidth=2.0, linestyle='-',
            #             label=r'MBHMF  $\int \phi_{\rm BH}(M,z)\,\frac{dV_c}{dz}\,dz$',
            #             zorder=6)

            # ── Proxy pool overlays (MC diagnostic) ───────────────────────
            _proxy_overlay(ax, proxy_lgMBH, w_gsmf, mbh_bins,
                           'grey', r'MC proxy: GSMF weight  $\phi(M,z)$', '--')
            _proxy_overlay(ax, proxy_lgMBH, w_rate, mbh_bins,
                           'red',  r'MC proxy: rate weight  $\phi\times\Gamma$', '-.')

            ax.set_yscale('log')
            ax.set_xlim(4.0, 7.0)
            ax.legend(fontsize=7, loc='upper left')
            ax.set_title(
                'Catalog should track black curve (theory)\n'
                'Red MC proxy is finite-sample estimate of same',
                fontsize=7.5, color='dimgrey')

        elif key == 'lgMgal':
            # ── Galaxy MF overlay on lgMgal panel ─────────────────────────
            ax.hist(data, bins=30, histtype='stepfilled',
                    color=color, alpha=0.5, density=True,
                    label=f'N={n_events}')
            ax.hist(data, bins=30, histtype='step',
                    color=color, linewidth=1.5, density=True)
            # Proxy GSMF overlay (normalised)
            _proxy_overlay(ax, gal_obj.lgMgal, w_gsmf,
                           np.linspace(data.min(), data.max(), 31),
                           'grey', r'GSMF proxy weight', '--')
            ax.set_yscale('log')
            ax.set_ylim(bottom=1e-6)
            ax.legend(fontsize=8)

        else:
            ax.hist(data, bins=30, histtype='stepfilled',
                    color=color, alpha=0.5, label=f'N={n_events}')
            ax.hist(data, bins=30, histtype='step',
                    color=color, linewidth=1.5)
            ax.set_yscale('log')
            ax.set_ylim(bottom=0.5)
            ax.legend(fontsize=8)

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('density  (log)' if key in ('lgMBH_mass','lgMgal') else 'N  (log)',
                      fontsize=10)

    for j in range(len(keys), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'{catalog_name} resampled catalog  —  {n_events} events',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    out = f'{loc}/catalog_histograms_{catalog_name}_{args.mbh_mass_method}_{args.nucleation_method}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {out}")


# ── Compute theoretical MBHMF once, reuse for both catalogs ───────────────
# z_table and dVc_dz_table are already in scope from the main script
# lgMBH_theory, phi_theory = compute_marginalized_mbhmf(
#     MBHMF       = MBHMassFunction(gsmf=GSMF),
#     cosmo_model = cosmo_model,
#     z_table     = z_table,
#     dVc_dz_table= dVc_dz_table,
# )

# ── Proxy pool ────────────────────────────────────────────────────────────
proxy_lgMBH = gal_obj.lgMBH_mass    # nucleated subset, shape (n_nuc,)

# ── EMRI ──────────────────────────────────────────────────────────────────
if emri_vals:
    _plot_one_catalog(
        catalog_name = 'EMRI',
        params_dict  = EMRI_PARAMS,
        vals_dict    = emri_vals,
        n_events     = emri_mask.sum(),
        proxy_lgMBH  = proxy_lgMBH,
        w_gsmf       = mc_weight_i_nuc,
        w_rate       = combined_EMRI_weight,
        color        = 'steelblue',
        loc          = loc,
        # lgMBH_theory = lgMBH_theory,
        # phi_theory   = phi_theory,
    )

# ── TDE ───────────────────────────────────────────────────────────────────
if tde_vals:
    _plot_one_catalog(
        catalog_name = 'TDE',
        params_dict  = TDE_PARAMS,
        vals_dict    = tde_vals,
        n_events     = tde_mask.sum(),
        proxy_lgMBH  = proxy_lgMBH,
        w_gsmf       = mc_weight_i_nuc,
        w_rate       = combined_TDE_weight,
        color        = 'darkorange',
        loc          = loc,
        # lgMBH_theory = lgMBH_theory,
        # phi_theory   = phi_theory,
    )