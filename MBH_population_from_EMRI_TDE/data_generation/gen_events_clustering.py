# python -m cProfile -o output.prof test_galaxies.py
# snakeviz output.prof

import os, sys, argparse
sys.path.insert(0, os.path.abspath('./astrophysical_setup'))
import argparse

import numpy as np
from galaxy import Galaxy
from nsc import NSC, CompactObject, MBH_properties
from density import DehnenProfile
from relaxation import RelaxationModel
from rate import RateModel
from evolution import CuspEvolution
from cosmology import LastMajorMerger, CosmologyModel, GalaxyStellarMassFunction

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='serif', serif=['Computer Modern'], size=15)
matplotlib.rc('text', usetex=True)
import seaborn as sns
from matplotlib import gridspec

import h5py

rng = np.random.default_rng(seed=42)
Z_MAX = 10.0

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES", type=int, required=True, help="Number of galaxies")
parser.add_argument("--OBSERVING_WINDOW", type=float, required=True, help="Observing window in years")
args = parser.parse_args()

cosmo_model = CosmologyModel()

def generate_lognormal_field(N, L, rng, A=100.0, n=1.0, k0=0.05, kc=0.5):
    """
    N^3 grid, box size L [Mpc]
    Normalization A, power-law index n, pivot scale k0 [1/Mpc], cutoff scale kc [1/Mpc]
        - A controls the overall variance of the field
        - n controls the slope of the power spectrum (n=1 is scale-invariant)
        - k0 is the pivot scale where the power spectrum is normalized
        - kc is the cutoff scale where power is exponentially suppressed (models small-scale damping)

        Returns:
        delta_ln: Log-normal density field
        dx: Grid spacing

    Eg:
        - Strong clustering (cosmic web-like): A = 30, n = 0.0, kc = 0.3
        - Smooth Universe (cosmic web-like): A = 2, n = -1.0, kc = 0.2
        - Highly nonlinear / clumpy (cosmic web-like): A = 80, n = 1.5, kc = 0.6
    """
    kfreq = 2*np.pi*np.fft.fftfreq(N, d=L/N) # All allowed wave numbers k for the grid 
    kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing='ij') # Create 3D grid of wave numbers in Fourier space
    k = np.sqrt(kx*kx + ky*ky + kz*kz) # Magnitude of wave number vector

    Pk = np.zeros_like(k)
    mask = k > 0
    Pk[mask] = A * (k[mask]/k0)**n * np.exp(-(k[mask]/kc)**2) # Power spectrum with cutoff

    a = rng.normal(size=(N,N,N)) # Random numbers from a Gaussian distribution for the real part of the Fourier modes
    b = rng.normal(size=(N,N,N)) # Random numbers from a Gaussian distribution for the imaginary part of the Fourier modes
    dk = (a + 1j*b) * np.sqrt(Pk * (L/N)**3 / 2.0) # complex Gaussian random number with variance Pk/2 for each mode (factor of 2 because of real and imaginary parts)

    delta_g = np.fft.ifftn(dk).real
    var = np.var(delta_g)
    delta_ln = np.exp(delta_g - 0.5*var) - 1.0
    print("Var(delta_g) =", np.var(delta_g))
    print("Var(delta_ln) =", np.var(delta_ln))
    print("min/max delta_ln =", delta_ln.min(), delta_ln.max())
    return delta_ln, L/N

def sample_spherical_lognormal(delta_ln, dx, Rmax, nbar, rng):
    N = delta_ln.shape[0]
    L = N*dx

    coords = (np.arange(N)+0.5)*dx - L/2
    x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
    r = np.sqrt(x*x + y*y + z*z)

    inside = r <= Rmax
    lam = nbar * (1 + delta_ln) * dx**3
    lam = np.where(inside, np.maximum(lam, 0.0), 0.0)

    counts = rng.poisson(lam)

    pts = []
    idx = np.argwhere(counts>0)
    for i,j,k in idx:
        for _ in range(counts[i,j,k]):
            pts.append([
                coords[i] + rng.uniform(-0.5*dx, 0.5*dx),
                coords[j] + rng.uniform(-0.5*dx, 0.5*dx),
                coords[k] + rng.uniform(-0.5*dx, 0.5*dx)
            ])
    return np.array(pts)

def evaluate_z_ra_dec(xyz):
    x, y, z = xyz.T
    r = np.sqrt(x*x + y*y + z*z)
    ra = np.degrees(np.arctan2(y,x)) % 360
    dec = np.degrees(np.arcsin(z/r))
    z_gal = cosmo_model.z_from_comoving_distance(r)
    return z_gal, ra, dec

N_objs = args.GALAXIES
T_obs = args.OBSERVING_WINDOW

Rmax = cosmo_model.comoving_distance(Z_MAX)
VOL = (4/3) * np.pi * Rmax**3
nbar = N_objs / VOL

print(f"Target number of galaxies = {N_objs}, nbar = {nbar:.3e} galaxies/Mpc^3")
print(f"Comoving radius Rmax = {Rmax:.1f} Mpc")

pad = 0.6
Lbox = 2 * Rmax * (1 + pad)
delta_ln, dx = generate_lognormal_field(N=256, L=Lbox, rng=rng)
xyz = sample_spherical_lognormal(delta_ln, dx, Rmax, nbar, rng)

print(f"Generated {len(xyz):,} galaxies")

z_grid, ra, dec = evaluate_z_ra_dec(xyz)

GSMF = GalaxyStellarMassFunction()
lgMgal_samples = GSMF.sample_gsmf(z_gal=z_grid, size=len(z_grid))
nucleation_indices = Galaxy.check_nucleation(lgMgal_samples, z_grid)

gal_obj = Galaxy(lgMgal=lgMgal_samples, z_gal=z_grid, ra=ra, dec=dec, nucleation_occurs=nucleation_indices)

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

loc = f'/data/wiay/postgrads/shashwat/EMRI_TDE_data/astrophysical_data/{args.GALAXIES}'
if not os.path.exists(loc):
    os.makedirs(loc)

hf = h5py.File(f'{loc}/data_cusp_evolution.h5', 'w')

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
hf.create_dataset('initial_MBHspin', data=MBH_obj.initial_MBHspin[mbh_mask]) # nucleation_index filter already applied

# since this is the same mass case so we just save the scalar values hence no masking is required
hf.create_dataset('sBH_masses', data=CO_objs.masses['sBH'])
hf.create_dataset('star_masses', data=CO_objs.masses['star'])

hf.create_dataset('generated_EMRIs', data=generated_EMRIs[mbh_mask]) # nucleation_index filter already applied
hf.create_dataset('generated_TDEs', data=generated_TDEs[mbh_mask]) # nucleation_index filter already applied
hf.close()

print("After MBH mass filter")
print(np.sum(generated_EMRIs[mbh_mask]), generated_EMRIs[mbh_mask].max(), np.sum(generated_TDEs[mbh_mask]), generated_TDEs[mbh_mask].max())


# with h5py.File(f'{loc}/data_cusp_evolution.h5', 'r') as hf:
#     mbh = np.array(hf['lgMBH'])
#     spin = np.array(hf['initial_MBHspin'])
#     z = np.array(hf['z_gal'])
