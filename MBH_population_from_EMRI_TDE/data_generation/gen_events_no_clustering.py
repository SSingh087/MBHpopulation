# python -m cProfile -o output_no_clustering.prof test_galaxies.py
# snakeviz output_no_clustering.prof

import os, sys, argparse
sys.path.insert(0, os.path.abspath('./astrophysical_setup'))
sys.path.insert(0, os.path.abspath('./'))
import argparse

import numpy as np
from galaxy import Galaxy
from nsc import NSC, CompactObject, MBH_properties
from density import DehnenProfile
from relaxation import RelaxationModel
from rate import RateModel
from evolution import CuspEvolution
from cosmology import LastMajorMerger, CosmologyModel, GalaxyStellarMassFunction
import h5py

rng = np.random.default_rng(seed=42)
Z_MAX = 10.0

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

nucleation_indices = Galaxy.check_nucleation(lgMgal_samples, z_grid)
 
gal_obj = Galaxy(lgMgal=lgMgal_samples, z_gal=z_grid, nucleation_occurs=nucleation_indices)

NSC_obj = NSC(gal_obj)

MBH_obj = MBH_properties(nsc=NSC_obj)

CO_objs = CompactObject(nsc=NSC_obj, masses={'sBH': 10.0, 'star': 1.0}, total_mass={'sBH': 20.0, 'star': 100.0}, types_CO=['sBH', 'star'], types_masses='same_mass', type_CO_limits=None)

dehnen_obj = DehnenProfile(nsc=NSC_obj, compact_object=CO_objs)

relax_obj = RelaxationModel(nsc=NSC_obj, compact_object=CO_objs, profile=dehnen_obj)

rate_obj = RateModel(nsc=NSC_obj)

cusp_evolution_object = CuspEvolution(nsc=NSC_obj, compact_object=CO_objs, relaxation=relax_obj, rate_model=rate_obj, LastMajorMerger=LastMajorMerger(cosmo_model))

generated_EMRIs = cusp_evolution_object.number_of_objects_in_time(T_obs=T_obs, kvir=1.0, kind='EMRI', unit='Gyr')
generated_TDEs = cusp_evolution_object.number_of_objects_in_time(T_obs=T_obs, kvir=1.0, kind='TDE', unit='Gyr')

print(np.sum(generated_EMRIs), generated_EMRIs.max(), np.sum(generated_TDEs), generated_TDEs.max())

loc = f'./data/astrophysical_data/{args.GALAXIES}'
if not os.path.exists(loc):
    os.makedirs(loc)

hf = h5py.File(f'{loc}/data_cusp_evolution_no_clustering.h5', 'w')

# apart from nucleation fraction check we also need to apply check on the S/MBH mass 
# since we are dealing with S/MBHs, the masses should be greater than 10^4

# Filter galaxies based on MBH mass
# these massive black holes are source frame
mbh_mask = (gal_obj.lgMBH_mass >= 4)

hf.create_dataset('lgMgal', data=lgMgal_samples[nucleation_indices][mbh_mask]) # apply both nucleation and MBH mass filters
hf.create_dataset('sigma_km_s', data=gal_obj.sigma_km_s[mbh_mask]) # nucleation_index filter already applied

# we don't need to save nucleation indices since we are already filtering based on the MBH mass

hf.create_dataset('z_gal', data=z_grid[nucleation_indices][mbh_mask]) 
hf.create_dataset('ra_deg', data=gal_obj.ra_deg[mbh_mask])
hf.create_dataset('dec_deg', data=gal_obj.dec_deg[mbh_mask])

hf.create_dataset('lgMBH', data=gal_obj.lgMBH_mass[mbh_mask])
hf.create_dataset('initial_MBHspin', data=MBH_obj.initial_MBHspin[mbh_mask])

# since this is the same mass case so we just save the scalar values hence no masking is required
hf.create_dataset('sBH_masses', data=CO_objs.masses['sBH'])
hf.create_dataset('star_masses', data=CO_objs.masses['star'])

hf.create_dataset('generated_EMRIs', data=generated_EMRIs[mbh_mask])
hf.create_dataset('generated_TDEs', data=generated_TDEs[mbh_mask])
hf.close()

print("After MBH mass filter")
print(np.sum(generated_EMRIs[mbh_mask]), generated_EMRIs[mbh_mask].max(), np.sum(generated_TDEs[mbh_mask]), generated_TDEs[mbh_mask].max())
