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
from utils import Plotting
import matplotlib.pyplot as plt
import matplotlib
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


# Rmax = cosmo_model.comoving_distance(Z_MAX)
# Rmin = cosmo_model.comoving_distance(1e-56)
# u = rng.random(N_objs)
# r = (Rmin**3 + (Rmax**3 - Rmin**3) * u)**(1/3)

# # Ensure monotonic inversion behaves well
# r.sort()

# z_grid = cosmo_model.z_from_comoving_distance(r)

from scipy.optimize import brentq

Rmin = cosmo_model.comoving_distance(1e-5)
Rmax = cosmo_model.comoving_distance(Z_MAX)

rng = np.random.default_rng()
u = rng.random(N_objs)
r_samples = (Rmin**3 + (Rmax**3 - Rmin**3) * u)**(1/3)

# Function to invert comoving distance
def z_from_r(r):
    return brentq(lambda z: cosmo_model.comoving_distance(z) - r, 1e-5, Z_MAX)

# Vectorized inversion
z_grid = np.array([z_from_r(r) for r in r_samples])

# z_grid = np.random.uniform(1E-5, Z_MAX, N_objs)  # this is N_objs redshifts 

GSMF = GalaxyStellarMassFunction()
lgMgal_samples = GSMF.sample_gsmf(z_gal=z_grid, size=N_objs)

# Check nucleation
nucleation_indices = Galaxy.check_nucleation(lgMgal_samples, z_grid)

# ========== FILTER FOR NUCLEATED GALAXIES EARLY ==========
# Convert indices to boolean mask
nucleation_mask = np.zeros(N_objs, dtype=bool)
nucleation_mask[nucleation_indices] = True

# Filter all data to keep only nucleated galaxies
lgMgal_samples_nuc = lgMgal_samples[nucleation_mask]
z_grid_nuc = z_grid[nucleation_mask]
N_nuc = len(lgMgal_samples_nuc)

print(f"Nucleation filter: {N_objs} → {N_nuc} galaxies")

# Now create Galaxy object with ONLY nucleated galaxies
gal_obj = Galaxy(lgMgal=lgMgal_samples_nuc, z_gal=z_grid_nuc, nucleation_occurs=np.arange(N_nuc))

NSC_obj = NSC(gal_obj)

MBH_obj = MBH_properties(nsc=NSC_obj)

CO_objs = CompactObject(nsc=NSC_obj, masses={'sBH': 10.0, 'star': 1.0}, total_mass={'sBH': 20.0, 'star': 100.0}, types_CO=['sBH', 'star'], types_masses='same_mass', type_CO_limits=None)

dehnen_obj = DehnenProfile(nsc=NSC_obj, compact_object=CO_objs)

relax_obj = RelaxationModel(nsc=NSC_obj, compact_object=CO_objs, profile=dehnen_obj)

rate_obj = RateModel(nsc=NSC_obj)

cusp_evolution_object = CuspEvolution(nsc=NSC_obj, compact_object=CO_objs, relaxation=relax_obj, rate_model=rate_obj, LastMajorMerger=LastMajorMerger(cosmo_model))

generated_EMRIs = cusp_evolution_object.number_of_objects_in_time(T_obs=T_obs, kvir=1.0, kind='EMRI', unit='Gyr')
generated_TDEs = cusp_evolution_object.number_of_objects_in_time(T_obs=T_obs, kvir=1.0, kind='TDE', unit='Gyr')

print(f"Total before MBH mass filter:")
print(f"  EMRIs: {np.sum(generated_EMRIs)}, max: {generated_EMRIs.max()}")
print(f"  TDEs: {np.sum(generated_TDEs)}, max: {generated_TDEs.max()}")

# ========== APPLY MBH MASS FILTER ==========
# Now apply the MBH mass filter (only nucleated galaxies are in gal_obj)
mbh_mass_mask = (gal_obj.lgMBH_mass >= 4)

print(f"\nMBH mass filter (>= 10^4):")
print(f"  Passing: {np.sum(mbh_mass_mask)} / {N_nuc}")

loc = f'./data/astrophysical_data/{args.GALAXIES}'
if not os.path.exists(loc):
    os.makedirs(loc)

hf = h5py.File(f'{loc}/data_cusp_evolution_no_clustering.h5', 'w')

# ========== SAVE DATA WITH MBH MASS FILTER ==========
# All arrays are already in nucleated space, just apply mbh_mass_mask directly
hf.create_dataset('lgMgal', data=lgMgal_samples_nuc[mbh_mass_mask])
hf.create_dataset('sigma_km_s', data=gal_obj.sigma_km_s[mbh_mass_mask])
hf.create_dataset('z_gal', data=z_grid_nuc[mbh_mass_mask])

sky_locs = gal_obj.sky_location()[mbh_mass_mask]
ra, dec = sky_locs[:, 0], sky_locs[:, 1]

hf.create_dataset('ra_deg', data=np.array(ra))
hf.create_dataset('dec_deg', data=np.array(dec))

hf.create_dataset('lgMBH', data=gal_obj.lgMBH_mass[mbh_mass_mask])
hf.create_dataset('initial_MBHspin', data=MBH_obj.initial_MBHspin[mbh_mass_mask])

# Since this is the same mass case, save scalar values
hf.create_dataset('sBH_masses', data=CO_objs.masses['sBH'])
hf.create_dataset('star_masses', data=CO_objs.masses['star'])

hf.create_dataset('generated_EMRIs', data=generated_EMRIs[mbh_mass_mask])
hf.create_dataset('generated_TDEs', data=generated_TDEs[mbh_mass_mask])
hf.close()

print(f"\nAfter MBH mass filter (final data saved):")
print(f"  Total galaxies: {np.sum(mbh_mass_mask)}")
print(f"  EMRIs: {np.sum(generated_EMRIs[mbh_mass_mask])}, max: {generated_EMRIs[mbh_mass_mask].max()}")
print(f"  TDEs: {np.sum(generated_TDEs[mbh_mass_mask])}, max: {generated_TDEs[mbh_mass_mask].max()}")
