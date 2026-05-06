# python -m cProfile -o output.prof test_galaxies.py
# snakeviz output_clustering.prof

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
matplotlib.rc('font', family='serif', serif=['Computer Modern'], size=15)
matplotlib.rc('text', usetex=True)
import seaborn as sns
from matplotlib import gridspec

import h5py

rng = np.random.default_rng(seed=42)

parser = argparse.ArgumentParser()
parser.add_argument("--GALAXIES", type=int, required=True, help="Number of galaxies")
parser.add_argument("--OBSERVING_WINDOW", type=float, required=True, help="Observing window in years")
parser.add_argument("--Z_MAX", type=float, required=False, default=10.0, help="Maximum redshift")
args = parser.parse_args()

N_objs = args.GALAXIES
T_obs = args.OBSERVING_WINDOW
Z_MAX = args.Z_MAX

cosmo_model = CosmologyModel()

def generate_lognormal_field(N, L, rng, A=2.0, n=-1.0, k0=0.05, kc=0.2):
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
    delta_ln = np.clip(delta_ln, -0.9, 10)
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

Rmax = cosmo_model.comoving_distance(Z_MAX)
VOL = (4/3) * np.pi * Rmax**3
nbar = N_objs / VOL

print(f"Target number of galaxies = {N_objs}, nbar = {nbar:.3e} galaxies/Mpc^3")
print(f"Comoving radius Rmax = {Rmax:.1f} Mpc")

pad = 0.6
Lbox = 2 * Rmax * (1 + pad)
delta_ln, dx = generate_lognormal_field(N=256, L=Lbox, rng=rng)
xyz = sample_spherical_lognormal(delta_ln, dx, Rmax, nbar, rng)

r = np.linalg.norm(xyz, axis=1)

Rmin = cosmo_model.comoving_distance(1e-5)

mask = r >= Rmin
xyz = xyz[mask]

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

hf = h5py.File(f'{loc}/data_cusp_evolution_clustering.h5', 'w')

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



# PLOTTING 
Plotting.MBHmass_vs_spin(gal_obj.lgMBH_mass[mbh_mask], MBH_obj.initial_MBHspin[mbh_mask], z_grid[nucleation_indices][mbh_mask], loc=f'{loc}/MBHmass_vs_spin_clustering.png', cmap='plasma')

import matplotlib.colors as mcolors

# fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

# # --- Color normalization across full z range ---
# norm = mcolors.Normalize(vmin=np.min(z_grid), vmax=np.max(z_grid))
# colors = z_grid[nucleation_indices][mbh_mask]

# # --- Top: EMRIs ---
# sc = axes[0].scatter(gal_obj.lgMBH_mass[mbh_mask], generated_EMRIs[mbh_mask], c=colors, cmap='plasma', norm=norm,
#                     marker='o', alpha=0.9, s=20)

# axes[0].set_title(f'EMRIs = {np.sum(generated_EMRIs[mbh_mask])}')
# axes[0].set_yscale('log')
# axes[0].set_ylabel(f'Number of events in {T_obs} yrs')

# axes[1].scatter(gal_obj.lgMBH_mass[mbh_mask], generated_TDEs[mbh_mask], c=colors, cmap='plasma', norm=norm,
#                 marker='d', alpha=0.9, s=20)

# axes[1].set_title(f'TDEs = {np.sum(generated_TDEs[mbh_mask])}')
# axes[1].set_yscale('log')
# axes[1].set_xlabel(r'$\log_{10}(M_{\mathrm{MBH}} / M_\odot)$')
# axes[1].set_ylabel(f'Number of events in {T_obs} yrs')

# # --- Shared colorbar ---
# cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
# fig.colorbar(sc, cax=cbar_ax, label='Redshift')

# plt.tight_layout(rect=[0, 0, 0.9, 1])
# plt.savefig(f'{loc}/generated_objects_clustering.png', dpi=300)
# plt.close()


# fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

# mass_bins = np.linspace(np.min(gal_obj.lgMBH_mass[mbh_mask]), 
#                         np.max(gal_obj.lgMBH_mass[mbh_mask]), 100)
# z_bins = np.linspace(np.min(z_grid[nucleation_indices][mbh_mask]), 
#                      np.max(z_grid[nucleation_indices][mbh_mask]), 100)

# # --- Top: EMRIs ---
# H_emri, xedges, yedges = np.histogram2d(
#     gal_obj.lgMBH_mass[mbh_mask],
#     z_grid[nucleation_indices][mbh_mask],
#     bins=[mass_bins, z_bins],
#     weights=generated_EMRIs[mbh_mask],
# )

# # Optionally smooth
# # H_emri = gaussian_filter(H_emri, sigma=1.5)

# X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
# pcm = axes[0].pcolormesh(X, Y, H_emri, cmap='plasma', shading='auto')

# axes[0].set_title(f'EMRIs = {np.sum(generated_EMRIs[mbh_mask])}')
# axes[0].set_ylabel('Redshift (z)')

# # Colorbar for EMRIs
# cbar_emri = fig.colorbar(pcm, ax=axes[0], label='Number of EMRI events')

# # --- Bottom: TDEs ---
# H_tde, _, _ = np.histogram2d(
#     gal_obj.lgMBH_mass[mbh_mask],
#     z_grid[nucleation_indices][mbh_mask],
#     bins=[mass_bins, z_bins],
#     weights=generated_TDEs[mbh_mask]
# )
# # H_tde = gaussian_filter(H_tde, sigma=1.5)

# pcm2 = axes[1].pcolormesh(X, Y, H_tde, cmap='plasma', shading='auto')

# axes[1].set_title(f'TDEs = {np.sum(generated_TDEs[mbh_mask])}')
# axes[1].set_xlabel(r'$\log_{10}(M_{\mathrm{MBH}} / M_\odot)$')
# axes[1].set_ylabel('Redshift (z)')

# # Colorbar for TDEs
# cbar_tde = fig.colorbar(pcm2, ax=axes[1], label='Number of TDE events')

# plt.tight_layout()
# plt.savefig(f'{loc}/generated_objects_clustering_smooth.png', dpi=300)
# plt.close()


# # Convert spherical coordinates (RA, Dec, z) to Cartesian for true 3D shells
# # RA in radians, Dec in radians
# ra_rad = np.radians(ra)
# dec_rad = np.radians(dec)
# z_filtered = z_grid[nucleation_indices][mbh_mask]  # for color coding in polar plots

# # Cartesian coordinates
# x = z_filtered * np.cos(dec_rad) * np.cos(ra_rad)
# y = z_filtered * np.cos(dec_rad) * np.sin(ra_rad)
# z = z_filtered * np.sin(dec_rad)

# # --- Polar plot RA vs z ---
# plt.figure(figsize=(8,6), facecolor='white')
# ax = plt.subplot(projection='polar')
# ax.set_facecolor('#f9f9f9')  # light background
# ax.scatter(ra_rad, z_filtered, c=colors, cmap='plasma', marker='o', s=20, alpha=0.6)
# ax.set_rlabel_position(240)  # radial labels at bottom
# ax.grid(True, color='gray', linestyle='--', alpha=0.3)
# plt.title('Galaxy Distribution: RA vs Redshift', fontsize=14)
# plt.savefig(f'{loc}/RA_vs_redshift_clustering.png', dpi=300)
# # plt.show()
# plt.close()

# # --- Polar plot Dec vs z ---
# # Since Dec is not circular, we shift it to 0-360 deg for polar visualization
# dec_shifted = dec + 90  # from [-90,90] -> [0,180]
# plt.figure(figsize=(8,6), facecolor='white')
# ax = plt.subplot(projection='polar')
# ax.set_facecolor('#f9f9f9')
# ax.scatter(np.radians(dec_shifted), z_filtered, c=colors, cmap='plasma', marker='o', s=20, alpha=0.6)
# ax.set_rlabel_position(240)  # radial labels at bottom
# ax.grid(True, color='gray', linestyle='--', alpha=0.3)
# plt.title('Galaxy Distribution: Dec vs Redshift', fontsize=14)
# plt.savefig(f'{loc}/Dec_vs_redshift_clustering.png', dpi=300)
# # plt.show()
# plt.close()


# # 3D scatter plot with concentric shells
# fig = plt.figure(figsize=(10,8), facecolor='white')
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(x, y, z, c=z_filtered, s=15, cmap='plasma')
# cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
# cbar.set_label('Redshift', fontsize=12)
# ax.set_xlabel('X [$z$]')
# ax.set_ylabel('Y [$z$]')
# ax.set_zlabel('Z [$z$]')
# ax.set_title('3D Galaxy Distribution (Concentric Shells)', fontsize=14)
# ax.grid(False)
# ax.set_box_aspect([1,1,1])

# plt.savefig(f'{loc}/3D_galaxy_distribution_clustering.png', dpi=300)
# # plt.show()


#  SHOW ANGULAR CLUSTERING

zmin, zmax_slice = 0.8, 1.0  # THIN slice
mask = (z_grid[nucleation_indices][mbh_mask] > zmin) & (z_grid[nucleation_indices][mbh_mask] < zmax_slice)

from scipy.ndimage import gaussian_filter

H, xedges, yedges = np.histogram2d(ra[mask], dec[mask], bins=(400,200))
Hs = gaussian_filter(H, sigma=2.0)

fig, ax = plt.subplots(figsize=(7,4))
im = ax.imshow(Hs.T, origin='lower', aspect='auto',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap='magma')
ax.set_xlabel("RA [deg]"); ax.set_ylabel("Dec [deg]")
ax.set_title("Angular density (smoothed)")
plt.colorbar(im, ax=ax, label="Smoothed counts")
plt.tight_layout()
plt.savefig(f'{loc}/angular_density_slice_clustering.png', dpi=300)
plt.close()


# Convert to radians and wrap RA to [-pi, pi]
# ra_rad  = np.deg2rad(ra[mask])
# dec_rad = np.deg2rad(dec[mask])
# ra_wrap = (ra_rad + np.pi) % (2*np.pi) - np.pi

# fig = plt.figure(figsize=(8,4.5))
# ax = fig.add_subplot(111, projection="mollweide")
# ax.scatter(-ra_wrap, dec_rad, s=0.5, alpha=0.4, rasterized=True)  # minus flips RA like sky maps
# ax.grid(True, alpha=0.3)
# ax.set_title(f"${zmin} \leq z \leq {zmax_slice}$")
# plt.tight_layout()
# plt.savefig(f'{loc}/mollweide_sky_map_{zmin}_{zmax_slice}_clustering.png', dpi=300)
# plt.close()

# choose a thin Dec slice
# dec0, ddec = 0.0, 2.0
# w = (np.abs(dec - dec0) < ddec)  # degrees
# fig, ax = plt.subplots(figsize=(9,4))
# ax.scatter(ra[w], r[w], s=0.3, alpha=0.3, rasterized=True)
# ax.set_xlabel("RA [deg]"); ax.set_ylabel("Comoving distance r [Mpc]")
# ax.set_title(f"Wedge plot |Dec-{dec0}|<{ddec}°")
# plt.tight_layout()
# plt.savefig(f'{loc}/wedge_plot_{zmin}_{zmax_slice}_clustering.png', dpi=300)
# plt.close()

# fig, axs = plt.subplots(1,3, figsize=(15,4), sharex=True, sharey=True)
# sw = 30.0
# axs[0].scatter(y[np.abs(x)<sw], z[np.abs(x)<sw], s=0.3, alpha=0.3); axs[0].set_title("|x|<sw : (y,z)")
# axs[1].scatter(x[np.abs(y)<sw], z[np.abs(y)<sw], s=0.3, alpha=0.3); axs[1].set_title("|y|<sw : (x,z)")
# axs[2].scatter(x[np.abs(z)<sw], y[np.abs(z)<sw], s=0.3, alpha=0.3); axs[2].set_title("|z|<sw : (x,y)")
# for ax in axs: ax.set_aspect("equal"); ax.grid(alpha=0.2)
# plt.tight_layout()
# plt.show()
# plt.close()

# from scipy.spatial import cKDTree

# def xi_dd_only(xyz, rbins):
#     tree = cKDTree(xyz)
#     # cumulative counts within r
#     counts = np.array([tree.count_neighbors(tree, r) for r in rbins])
#     DD_shell = np.diff(counts)  # pairs per shell (double-counted)
#     return DD_shell

# # Use a subsample if huge:
# sub = xyz[rng.choice(len(xyz), size=min(len(xyz), 200_000), replace=False)]
# rbins = np.linspace(5, 150, 30)
# DD = xi_dd_only(sub, rbins)
# rcent = 0.5*(rbins[1:]+rbins[:-1])

# plt.figure()
# plt.loglog(rcent, DD + 1e-12)
# plt.xlabel("r [Mpc]"); plt.ylabel("DD(r) (arb.)")
# plt.title("Pair counts vs separation (diagnostic)")
# plt.savefig(f'{loc}/pair_counts_vs_separation_clustering.png', dpi=300)
# plt.close()