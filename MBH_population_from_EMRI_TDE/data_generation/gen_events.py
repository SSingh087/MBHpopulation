# python -m cProfile -o output_no_clustering.prof test_galaxies.py
# snakeviz output_no_clustering.prof

import os, sys, argparse
sys.path.insert(0, os.path.abspath('./astrophysical_setup'))
sys.path.insert(0, os.path.abspath('./'))
import argparse

import numpy as np
from scipy.stats import beta
from galaxy import Galaxy
from nsc import NSC, CompactObject, MBH_properties
from density import DehnenProfile
from relaxation import RelaxationModel
from rate import RateModel, ConstantRate
from evolution import CuspEvolution
from cosmology import LastMajorMerger, CosmologyModel, GalaxyStellarMassFunction
import h5py

rng = np.random.default_rng(seed=42)

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES", type=int, required=True, help="Number of galaxies")
parser.add_argument("--OBSERVING_WINDOW", type=float, required=True, help="Observing window in years")
parser.add_argument("--Z_MAX", type=float, required=False, default=10.0, help="Maximum redshift")
args = parser.parse_args()

cosmo_model = CosmologyModel()
Z_MAX = args.Z_MAX

N_objs = args.GALAXIES
T_obs = args.OBSERVING_WINDOW

Rmin = cosmo_model.comoving_distance(1e-5)
Rmax = cosmo_model.comoving_distance(Z_MAX)

rng = np.random.default_rng()
u = rng.random(N_objs)
r_samples = (Rmin**3 + (Rmax**3 - Rmin**3) * u)**(1/3)

z_grid = cosmo_model.z_from_comoving_distance_fast(r_samples)

# z_grid = np.random.uniform(1E-5, Z_MAX, N_objs)  # this is N_objs redshifts 
GSMF = GalaxyStellarMassFunction()
lgMgal_samples = GSMF.sample_gsmf(z_gal=z_grid, n_samples_per_z=1)


def emri_parameters(M):
    """
    Returns alpha1, beta1, alpha2, beta2, w for a given MBH mass M.
    """

    lgM = np.log10(M)

    # ----------------------------
    # Classical EMRIs (stable peak ~0.2)
    # ----------------------------
    alpha1 = 4.0 + 0.5 * (lgM - 6)
    beta1  = 10.0 - 0.8 * (lgM - 6)

    # ----------------------------
    # Wide / cliffhanger EMRIs
    # ----------------------------
    alpha2 = 3.0 - 0.3 * (lgM - 6)
    beta2  = 6.5 + 0.5 * (lgM - 6)

    # avoid invalid values

    alpha1 = max(alpha1, 1.0)
    beta1  = max(beta1,  1.0)
    alpha2 = max(alpha2, 1.0)
    beta2  = max(beta2,  1.0)


    # ----------------------------
    # Mixing weight (strong MBH dependence)
    # ----------------------------
    M0 = 1e5
    gamma = 1.2
    w = 1.0 / (1.0 + (M0 / M)**gamma)

    return alpha1, beta1, alpha2, beta2, w

def emri_eccentricity_pdf(e, M):
    """
    p(e | M_BH) for EMRIs (mixture of classical + wide).
    """

    a1, b1, a2, b2, w = emri_parameters(M)

    return (
        w * beta.pdf(e, a1, b1)
        + (1 - w) * beta.pdf(e, a2, b2)
    )

def sample_emri_eccentricity(M, size=1):
    """
    Draw samples of eccentricity at plunge for a given MBH mass.
    """

    a1, b1, a2, b2, w = emri_parameters(M)

    # choose component
    is_classical = np.random.rand(size) < w

    e = np.zeros(size)

    # classical
    n1 = np.sum(is_classical)
    if n1 > 0:
        e[is_classical] = np.random.beta(a1, b1, n1)

    # wide
    n2 = size - n1
    if n2 > 0:
        e[~is_classical] = np.random.beta(a2, b2, n2)

    return e if size > 1 else e[0]


nucleation_indices = Galaxy.check_nucleation(lgMgal_samples, z_grid)
 
gal_obj = Galaxy(lgMgal=lgMgal_samples, z_gal=z_grid, nucleation_occurs=nucleation_indices)

NSC_obj = NSC(gal_obj)

MBH_obj = MBH_properties(nsc=NSC_obj)

CO_objs = CompactObject(nsc=NSC_obj, masses={'sBH': 10.0, 'star': 1.0}, total_mass_in_MBH_mass_units={'sBH': 0.2, 'star': 20.0}, types_CO=['sBH', 'star'], types_masses='same_mass', type_CO_limits=None)

dehnen_obj = DehnenProfile(nsc=NSC_obj, compact_object=CO_objs)

relax_obj = RelaxationModel(nsc=NSC_obj, compact_object=CO_objs, profile=dehnen_obj)

rate_obj = RateModel(nsc=NSC_obj)

cusp_evolution_object = CuspEvolution(nsc=NSC_obj, compact_object=CO_objs, relaxation=relax_obj, rate_model=rate_obj, LastMajorMerger=LastMajorMerger(cosmo_model))

# generated_EMRIs = cusp_evolution_object.number_of_objects_in_time(T_obs=T_obs, kvir=1.0, kind='EMRI')
# generated_TDEs = cusp_evolution_object.number_of_objects_in_time(T_obs=T_obs, kvir=1.0, kind='TDE')


generated_EMRIs = cusp_evolution_object.number_of_objects_in_time_constant_rate(T_obs_yr=T_obs, kvir=1.0, kind='EMRI', RateModel=ConstantRate(nsc=NSC_obj))
generated_TDEs = cusp_evolution_object.number_of_objects_in_time_constant_rate(T_obs_yr=T_obs, kvir=1.0, kind='TDE', RateModel=ConstantRate(nsc=NSC_obj))
breakpoint()

MBH_mask =  gal_obj.lgMBH_mass > 4.0 
print(f"Generated EMRIs for  log10(M_BH/Msun) > 4.0: {np.sum(generated_EMRIs[MBH_mask])}")
print(f"Generated TDEs for  log10(M_BH/Msun) > 4.0: {np.sum(generated_TDEs[MBH_mask])}") 

z_nuc = z_grid[nucleation_indices]

mask_TDE = MBH_mask & (z_nuc < 4.0)
mask_EMRI = MBH_mask & (z_nuc < 8.0)


print(f"Generated TDEs for  z < 4.0 and log10(M_BH/Msun) > 4.0: {np.sum(generated_TDEs[mask_TDE])}")
print(f"Generated EMRIs for  z < 8.0 and log10(M_BH/Msun) > 4.0: {np.sum(generated_EMRIs[mask_EMRI])}")

loc = f'./data/astrophysical_data/{args.GALAXIES}'
if not os.path.exists(loc):
    os.makedirs(loc)

hf = h5py.File(f'{loc}/data_cusp_evolution_no_clustering.h5', 'w')

# apart from nucleation fraction check we also need to apply check on the S/MBH mass 
# since we are dealing with S/MBHs, the masses should be greater than 10^4

# Filter galaxies based on MBH mass
# these massive black holes are source frame


hf.create_dataset('lgMgal', data=lgMgal_samples[nucleation_indices][MBH_mask]) # apply both nucleation and MBH mass filters
hf.create_dataset('sigma_km_s', data=gal_obj.sigma_km_s[MBH_mask]) # nucleation_index filter already applied

# we don't need to save nucleation indices since we are already filtering based on the MBH mass

hf.create_dataset('z_gal', data=z_grid[nucleation_indices][MBH_mask]) 
hf.create_dataset('ra_deg', data=gal_obj.ra_deg[MBH_mask])
hf.create_dataset('dec_deg', data=gal_obj.dec_deg[MBH_mask])

hf.create_dataset('lgMBH', data=gal_obj.lgMBH_mass[MBH_mask])
hf.create_dataset('initial_MBHspin', data=MBH_obj.initial_MBHspin[MBH_mask])

eccentricities_at_plunge = np.array([sample_emri_eccentricity(10**lgM) for lgM in gal_obj.lgMBH_mass[MBH_mask]])
hf.create_dataset('eccentricities_at_plunge', data=eccentricities_at_plunge)


# since this is the same mass case so we just save the scalar values hence no masking is required
hf.create_dataset('sBH_masses', data=CO_objs.masses['sBH'])
hf.create_dataset('star_masses', data=CO_objs.masses['star'])

hf.create_dataset('generated_EMRIs', data=generated_EMRIs[MBH_mask])
hf.create_dataset('generated_TDEs', data=generated_TDEs[MBH_mask])
hf.close()
