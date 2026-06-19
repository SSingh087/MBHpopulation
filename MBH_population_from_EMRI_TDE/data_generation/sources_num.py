import os, sys, argparse
sys.path.insert(0, os.path.abspath('./astrophysical_setup'))
sys.path.insert(0, os.path.abspath('./'))

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import beta
from galaxy import Galaxy
from nsc import NSC, CompactObject, MBH_properties
from density import DehnenProfile
from relaxation import RelaxationModel
from rate import RateModel
from evolution import CuspEvolution
from cosmology import LastMajorMerger, CosmologyModel, GalaxyStellarMassFunction
import h5py

rng = np.random.default_rng(seed=42)

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES", type=int, required=True,
                     help="Monte Carlo sample size (NOT the literal real galaxy count)")
parser.add_argument("--OBSERVING_WINDOW", type=float, required=True, help="Observing window in years")
parser.add_argument("--Z_MAX", type=float, required=False, default=10.0, help="Maximum redshift")
args = parser.parse_args()

cosmo_model = CosmologyModel()
Z_MAX = args.Z_MAX
Z_MIN = 1e-5

N_objs_sim = args.GALAXIES
T_obs = args.OBSERVING_WINDOW

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


z_table, n_density_table, n_density_at_z_interp = build_density_tables(GSMF, Z_MIN, Z_MAX)

dVc_dz_table = cosmo_model.dVc_dz(z_table)
N_gal_true = np.trapz(n_density_table * dVc_dz_table, z_table)

print(f"N_gal_true (real galaxy count out to z={Z_MAX}): {N_gal_true:.6e}")


# ============================================================
# z sampling: unchanged, already correctly proportional to the
# true volume-element marginal density.
# ============================================================
u = rng.random(N_objs_sim)
r_samples = (Rmin**3 + (Rmax**3 - Rmin**3) * u)**(1/3)
z_grid = cosmo_model.z_from_comoving_distance_fast(r_samples)

# ============================================================
# lgMgal sampling: UNIFORM over the table's mass range (not
# GSMF-shaped), to guarantee the high-Gamma tail gets covered.
# ============================================================
lgM_min, lgM_max = GSMF.lgMgal_data.min(), GSMF.lgMgal_data.max()
lgMgal_samples = rng.uniform(lgM_min, lgM_max, size=N_objs_sim)
q_lgM = 1.0 / (lgM_max - lgM_min)   # constant proposal density

# True conditional density p(lgM | z) at each drawn point, via the exact
# interpolator (not the coarse multinomial table sample_gsmf uses)
log10phi_pointwise = GSMF.interp(np.column_stack([z_grid, lgMgal_samples]))
phi_pointwise = np.nan_to_num(10.0 ** log10phi_pointwise, nan=0.0, posinf=0.0, neginf=0.0)

V_tot = np.trapz(dVc_dz_table, z_table)  # total comoving volume in [Z_MIN, Z_MAX], Mpc^3

mc_weight_i = phi_pointwise * V_tot * (lgM_max - lgM_min) / N_objs_sim   # shape (N_objs_sim,)

print(f"mc_weight_i range: [{mc_weight_i.min():.3e}, {mc_weight_i.max():.3e}], "
      f"median {np.median(mc_weight_i):.3e}")


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


# ============================================================
# Physics pipeline
# ============================================================
nucleation_indices = Galaxy.check_nucleation(lgMgal_samples, z_grid)
gal_obj = Galaxy(lgMgal=lgMgal_samples, z_gal=z_grid, nucleation_occurs=nucleation_indices)

NSC_obj = NSC(gal_obj)
MBH_obj = MBH_properties(nsc=NSC_obj)

CO_objs = CompactObject(nsc=NSC_obj, masses={'sBH': 10.0, 'star': 1.0},
                         total_mass_in_MBH_mass_units={'sBH': 0.2, 'star': 20.0},
                         types_CO=['sBH', 'star'], types_masses='same_mass', type_CO_limits=None)

dehnen_obj = DehnenProfile(nsc=NSC_obj, compact_object=CO_objs)
relax_obj = RelaxationModel(nsc=NSC_obj, compact_object=CO_objs, profile=dehnen_obj)
rate_obj = RateModel(nsc=NSC_obj)

cusp_evolution_object = CuspEvolution(nsc=NSC_obj, compact_object=CO_objs, relaxation=relax_obj,
                                       rate_model=rate_obj, LastMajorMerger=LastMajorMerger(cosmo_model))

Gamma_EMRI_i = cusp_evolution_object.expected_objects_in_time(T_obs=T_obs, kvir=1.0, kind='EMRI')
Gamma_TDE_i = cusp_evolution_object.expected_objects_in_time(T_obs=T_obs, kvir=1.0, kind='TDE')

MBH_mask = gal_obj.lgMBH_mass > 4.0
Gamma_EMRI_i = np.where(MBH_mask, Gamma_EMRI_i, 0.0)
Gamma_TDE_i = np.where(MBH_mask, Gamma_TDE_i, 0.0)

# IMPORTANT: mc_weight_i was built over the FULL N_objs_sim pool, but gal_obj
# (and therefore Gamma_*_i) is already restricted to the nucleated subset.
# Subset mc_weight_i the same way before combining.
mc_weight_i_nuc = mc_weight_i[nucleation_indices]

z_nuc = z_grid[nucleation_indices]
mask_TDE_extra = MBH_mask & (z_nuc < 4.0)
mask_EMRI_extra = MBH_mask & (z_nuc < 8.0)

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


Lambda_EMRI_tot = np.sum(combined_EMRI_weight)
Lambda_TDE_tot = np.sum(combined_TDE_weight)
Lambda_EMRI_zcut = np.sum(combined_EMRI_weight[mask_EMRI_extra])
Lambda_TDE_zcut = np.sum(combined_TDE_weight[mask_TDE_extra])

print(f"Total expected EMRIs, log10(M_BH/Msun) > 4.0: {Lambda_EMRI_tot:.3f}")
print(f"Total expected TDEs,  log10(M_BH/Msun) > 4.0: {Lambda_TDE_tot:.3f}")
print(f"Total expected EMRIs, z < 8.0 and log10(M_BH/Msun) > 4.0: {Lambda_EMRI_zcut:.3f}")
print(f"Total expected TDEs,  z < 4.0 and log10(M_BH/Msun) > 4.0: {Lambda_TDE_zcut:.3f}")

N_EMRI_events = int(round(Lambda_EMRI_tot))
N_TDE_events = int(round(Lambda_TDE_tot))

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
loc = f'./data/astrophysical_data/{args.GALAXIES}'
if not os.path.exists(loc):
    os.makedirs(loc)
 
hf = h5py.File(f'{loc}/data_cusp_evolution_no_clustering.h5', 'w')
 
hf.attrs['N_gal_true'] = N_gal_true
hf.attrs['N_objs_sim'] = N_objs_sim
hf.attrs['T_obs_years'] = T_obs
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
 
grp_emri = hf.create_group('EMRI_catalog')
if emri_gal is not None:
    grp_emri.create_dataset('lgMgal', data=emri_gal.lgMgal)
    grp_emri.create_dataset('z_gal', data=emri_gal.z_gal)
    grp_emri.create_dataset('lgMBH', data=emri_gal.lgMBH_mass)
    grp_emri.create_dataset('sigma_km_s', data=emri_gal.sigma_km_s)
    grp_emri.create_dataset('ra_deg', data=emri_gal.ra_deg)
    grp_emri.create_dataset('dec_deg', data=emri_gal.dec_deg)
    grp_emri.create_dataset('initial_MBHspin', data=emri_mbh.initial_MBHspin)
 
    eccentricities_at_plunge = np.array(
        [sample_emri_eccentricity(10 ** lgM, rng=rng) for lgM in emri_gal.lgMBH_mass]
    )
    grp_emri.create_dataset('eccentricities_at_plunge', data=eccentricities_at_plunge)
else:
    for name in ['lgMgal', 'z_gal', 'lgMBH', 'sigma_km_s', 'ra_deg', 'dec_deg',
                 'initial_MBHspin', 'eccentricities_at_plunge']:
        grp_emri.create_dataset(name, data=np.array([]))
 
grp_tde = hf.create_group('TDE_catalog')
if tde_gal is not None:
    grp_tde.create_dataset('lgMgal', data=tde_gal.lgMgal)
    grp_tde.create_dataset('z_gal', data=tde_gal.z_gal)
    grp_tde.create_dataset('lgMBH', data=tde_gal.lgMBH_mass)
    grp_tde.create_dataset('sigma_km_s', data=tde_gal.sigma_km_s)
    grp_tde.create_dataset('ra_deg', data=tde_gal.ra_deg)
    grp_tde.create_dataset('dec_deg', data=tde_gal.dec_deg)
    grp_tde.create_dataset('initial_MBHspin', data=tde_mbh.initial_MBHspin)
else:
    for name in ['lgMgal', 'z_gal', 'lgMBH', 'sigma_km_s', 'ra_deg', 'dec_deg', 'initial_MBHspin']:
        grp_tde.create_dataset(name, data=np.array([]))
 
hf.close()
print(f"Saved catalog to {loc}/data_cusp_evolution_no_clustering.h5")