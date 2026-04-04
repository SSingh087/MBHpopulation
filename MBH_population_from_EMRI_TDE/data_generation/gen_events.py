# python -m cProfile -o output.prof test_galaxies.py
# snakeviz output.prof

import os, sys, argparse
sys.path.insert(0, os.path.abspath('../astrophysical_setup'))

import argparse

from utils import *
from galaxy import *
from nsc import NSC, CompactObject
from density import DehnenProfile
from relaxation import RelaxationModel
from rate import RateModel, UniversalRate
from evolution import CuspEvolution
from cosmology import LastMajorMerger, CosmologyModel, GalaxyStellarMassFunction, MBHMassFunction

import matplotlib.pyplot as plt
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("--galaxies", type=int, required=True, help="Number of galaxies")
args = parser.parse_args()

cosmo_model = CosmologyModel()

N_objs = args.galaxies
z_grid = np.random.uniform(0.01, 8.0, N_objs) 

GSMF = GalaxyStellarMassFunction()
lgMgal_samples = GSMF.sample_gsmf(z_gal=z_grid, size=N_objs)
nucleation_indices = Galaxy.check_nucleation(lgMgal_samples, z_grid)
 
gal_obj = Galaxy(lgMgal=lgMgal_samples, z_gal=z_grid, nucleation_occurs=nucleation_indices)

NSC_obj = NSC(gal_obj)

CO_objs = CompactObject(nsc=NSC_obj, masses={'sBH': 10.0, 'star': 1.0}, total_mass={'sBH': 20.0, 'star': 100.0}, types_CO=['sBH', 'star'], types_masses='same_mass', type_CO_limits=None)

dehnen_obj = DehnenProfile(nsc=NSC_obj, compact_object=CO_objs)

relax_obj = RelaxationModel(nsc=NSC_obj, compact_object=CO_objs, profile=dehnen_obj)

rate_obj = RateModel(nsc=NSC_obj)

cusp_evolution_object = CuspEvolution(nsc=NSC_obj, compact_object=CO_objs, relaxation=relax_obj, rate_model=rate_obj, LastMajorMerger=LastMajorMerger(cosmo_model))

cusp_age = cusp_evolution_object.cusp_age(kvir=1.0, unit='Gyr')

accumulated_EMRIs = cusp_evolution_object.accumulated_objects_within_time(kvir=1.0, kind='EMRI', unit='Gyr')

accumulated_TDEs = cusp_evolution_object.accumulated_objects_within_time(kvir=1.0, kind='TDE', unit='Gyr')

print(accumulated_EMRIs.shape, accumulated_TDEs.shape)


hf = h5py.File('data_cusp_evolution.h5', 'w')
hf.create_dataset('lgMgal', data=lgMgal_samples[nucleation_indices])
hf.create_dataset('z_gal', data=z_grid[nucleation_indices])
hf.create_dataset('nucleation_occurs', data=nucleation_indices)
hf.create_dataset('lgMBH', data=gal_obj.lgMBH_mass)
hf.create_dataset('sigma_km_s', data=gal_obj.sigma_km_s)
hf.create_dataset('cusp_age', data=cusp_age)
hf.create_dataset('accumulated_EMRIs', data=accumulated_EMRIs)
hf.create_dataset('accumulated_TDEs', data=accumulated_TDEs)
hf.close()


# plt.scatter(cusp_age, accumulated_EMRIs, marker='o', alpha=0.5)
# plt.scatter(cusp_age, accumulated_TDEs.T, marker='o', alpha=0.5)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Cusp Age (Gyr)')
# plt.ylabel('Accumulated Objects')
# plt.legend(['EMRIs', 'TDEs'])
# plt.savefig('accumulated_objects_vs_cusp_age.png', dpi=300)
# plt.show()