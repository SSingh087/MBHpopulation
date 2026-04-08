# python -m cProfile -o output.prof test_galaxies.py
# snakeviz output.prof

import os, sys, argparse
sys.path.insert(0, os.path.abspath('../astrophysical_setup'))

import argparse

from utils import *
from galaxy import *
from nsc import NSC, CompactObject, MBH_properties
from density import DehnenProfile
from relaxation import RelaxationModel
from rate import RateModel, UniversalRate
from evolution import CuspEvolution
from cosmology import LastMajorMerger, CosmologyModel, GalaxyStellarMassFunction, MBHMassFunction

import matplotlib.pyplot as plt
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES", type=int, required=True, help="Number of galaxies")
parser.add_argument("--OBSERVING_WINDOW", type=float, required=True, help="Observing window in years")
args = parser.parse_args()

cosmo_model = CosmologyModel()

N_objs = args.GALAXIES
T_obs = args.OBSERVING_WINDOW
z_grid = np.random.uniform(1E-5, 8.0, N_objs) 

GSMF = GalaxyStellarMassFunction()
lgMgal_samples = GSMF.sample_gsmf(z_gal=z_grid, size=N_objs)
nucleation_indices = Galaxy.check_nucleation(lgMgal_samples, z_grid)
 
gal_obj = Galaxy(lgMgal=lgMgal_samples, z_gal=z_grid, nucleation_occurs=nucleation_indices)

NSC_obj = NSC(gal_obj)

MBH_obj = MBH_properties(nsc=NSC_obj)

CO_objs = CompactObject(nsc=NSC_obj, masses={'sBH': 10.0, 'star': 1.0}, total_mass={'sBH': 20.0, 'star': 100.0}, types_CO=['sBH', 'star'], types_masses='same_mass', type_CO_limits=None)

dehnen_obj = DehnenProfile(nsc=NSC_obj, compact_object=CO_objs)

relax_obj = RelaxationModel(nsc=NSC_obj, compact_object=CO_objs, profile=dehnen_obj)

rate_obj = RateModel(nsc=NSC_obj)

cusp_evolution_object = CuspEvolution(nsc=NSC_obj, compact_object=CO_objs, relaxation=relax_obj, rate_model=rate_obj, LastMajorMerger=LastMajorMerger(cosmo_model))

observed_EMRIs = cusp_evolution_object.number_of_objects_in_time(T_obs=T_obs, kvir=1.0, kind='EMRI', unit='Gyr')
observed_TDEs = cusp_evolution_object.number_of_objects_in_time(T_obs=T_obs, kvir=1.0, kind='TDE', unit='Gyr')

print(np.sum(observed_EMRIs), observed_EMRIs.max(), np.sum(observed_TDEs), observed_TDEs.max())

hf = h5py.File('./DATA/data_cusp_evolution.h5', 'w')

# apart from nucleation fraction check we also need to apply check on the S/MBH mass 
# since we are dealing with S/MBHs, the masses should be greater than 10^4

# Filter galaxies based on MBH mass
# these massive black holes are source frame
mbh_mask = (gal_obj.lgMBH_mass >= 4)

hf.create_dataset('lgMgal', data=lgMgal_samples[nucleation_indices][mbh_mask]) # apply both nucleation and MBH mass filters
hf.create_dataset('sigma_km_s', data=gal_obj.sigma_km_s[mbh_mask]) # nucleation_index filter already applied

# we don't need to save nucleation indices since we are already filtering based on the MBH mass

hf.create_dataset('z_gal', data=z_grid[nucleation_indices][mbh_mask]) 
sky_locs = gal_obj.sky_location()[nucleation_indices][mbh_mask]
ra, dec = sky_locs[:, 0], sky_locs[:, 1]

hf.create_dataset('ra_deg', data=np.array(ra))
hf.create_dataset('dec_deg', data=np.array(dec))

hf.create_dataset('lgMBH', data=gal_obj.lgMBH_mass[mbh_mask]) # nucleation_index filter already applied
hf.create_dataset('MBHspin', data=MBH_obj.MBHspin[mbh_mask]) # nucleation_index filter already applied

# since this is the same mass case so we just save the scalar values hence no masking is required
hf.create_dataset('sBH_masses', data=CO_objs.masses['sBH'])
hf.create_dataset('star_masses', data=CO_objs.masses['star'])

hf.create_dataset('observed_EMRIs', data=observed_EMRIs[mbh_mask]) # nucleation_index filter already applied
hf.create_dataset('observed_TDEs', data=observed_TDEs[mbh_mask]) # nucleation_index filter already applied
hf.close()

print("After MBH mass filter")
print(np.sum(observed_EMRIs[mbh_mask]), observed_EMRIs[mbh_mask].max(), np.sum(observed_TDEs[mbh_mask]), observed_TDEs[mbh_mask].max())


plt.scatter(gal_obj.lgMBH_mass[mbh_mask], observed_EMRIs[mbh_mask], c=z_grid[nucleation_indices][mbh_mask], marker='o', alpha=0.9)
plt.scatter(gal_obj.lgMBH_mass[mbh_mask], observed_TDEs[mbh_mask], c=z_grid[nucleation_indices][mbh_mask], marker='d', alpha=0.9)
plt.colorbar(label='Redshift')
plt.yscale('log')
plt.xlabel('$\log_{10}(M_{\mathrm{BH}} / M_\odot)$ (source-frame)')
plt.ylabel(f'Number of seeded events within {T_obs} years')
plt.legend(['EMRIs', 'TDEs'])
plt.savefig('observed_objects.pdf', dpi=300)
plt.show()