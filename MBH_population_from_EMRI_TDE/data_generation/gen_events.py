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
hf.create_dataset('lgMgal', data=lgMgal_samples[nucleation_indices])
hf.create_dataset('nucleation_occurs', data=nucleation_indices)
hf.create_dataset('sigma_km_s', data=gal_obj.sigma_km_s)

hf.create_dataset('z_gal', data=z_grid[nucleation_indices])
ra, dec = zip(*[gal_obj.sky_location() for _ in range(len(gal_obj.lgMBH_mass))])
hf.create_dataset('ra_deg', data=np.array(ra))
hf.create_dataset('dec_deg', data=np.array(dec))

hf.create_dataset('lgMBH', data=gal_obj.lgMBH_mass)
hf.create_dataset('MBHspin', data=MBH_obj.MBHspin)

hf.create_dataset('sBH_masses', data=CO_objs.masses['sBH'])
hf.create_dataset('star_masses', data=CO_objs.masses['star'])

hf.create_dataset('observed_EMRIs', data=observed_EMRIs)
hf.create_dataset('observed_TDEs', data=observed_TDEs)
hf.close()

plt.scatter(gal_obj.lgMBH_mass, observed_EMRIs, marker='o', alpha=0.5)
plt.scatter(gal_obj.lgMBH_mass, observed_TDEs, marker='o', alpha=0.5)
plt.yscale('log')
plt.xlabel('$\log_{10}(M_{\mathrm{BH}} / M_\odot)$')
plt.ylabel(f'Observed Events within {T_obs} years')
plt.legend(['EMRIs', 'TDEs'])
plt.savefig('observed_objects.pdf', dpi=300)
# plt.show()