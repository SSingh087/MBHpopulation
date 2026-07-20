import numpy as np


# Unit Conversions

# Length
pc_to_cm = 3.085677581491367e18      # parsec -> cm
kpc_to_cm = 1e3 * pc_to_cm           # kiloparsec -> cm

# Time
sec_per_year = 3.154e7               # s / yr  (consistent with your usage)

# Mass
Msun_to_grams = 1.98847e33           # solar mass -> grams

# Speed of light conversions
c_cgs = 2.99792458e10                # cm/s
c_pc_per_year = c_cgs * sec_per_year / pc_to_cm   # pc / yr

# Gravitational Constant in Multiple Unit Systems

G_cgs = 6.67430e-8                    # cm^3 g^-1 s^-2

# G in pc^3 / (Msun * yr^2)
G_pc3_per_Msun_yr2 = G_cgs * (sec_per_year**2) / (pc_to_cm**3 / Msun_to_grams)

# M-sigma relation defaults
MBH_A_sigma = 7.87
MBH_B_sigma = 4.55
MBH_scatter_sigma = 0.53

MBH_A_LGMGAL = 7.43
MBH_B_LGMGAL = 1.61
MBH_scatter_LGMGAL = 0.81

MBH_sigma0 = 160.0  # km/s arXiv:2205.06277 uses sigma_0 = 70 which will have
                     # consequence of the Gamma_E and the t_EMRI. t_EMRI will
                     # go up and Gamma_E lower

MGAL0 = 3.0e10  # Msun https://www.annualreviews.org/docserver/fulltext/astro/58/1/aa58_greene_supmat.pdf?expires=1784470050&id=id&accname=guest&checksum=836916D30A41C751799F2E61AF43390B


# Broggi et al. 2022 (arXiv:2205.06277) Eq. 15 -- the M-sigma normalization
# THEIR simulation grid was built around. Independent of this project's
# own M-sigma relation (MBH_A/MBH_B/MBH_sigma0 above) -- needed specifically
# to evaluate sigma/sigma_0 in the rate fits, since those were calibrated
# against Broggi's grid, not against this population's relation.
BROGGI_SIGMA0    = 70.0      # km/s
BROGGI_MBH_PIVOT = 1.53e6    # Msun
BROGGI_INV_SLOPE = 1.0 / 4.24

# Dehnen profile default slope
gamma_dehnen_initial = 1.5

# Coulomb logarithm used in t_relax
lnLambda = 15.0

# Cusp regrowth fraction
kappa_cusp = 0.25

# Miscellaneous constants
Rsun = 6.957e10                       # cm (used in tidal radius)
Msun = 1.0                            # Msun units (placeholder for consistency)


# Random number generator
# this is solely for reproducibility and testing
# the main code should use this function to get RNGs
# instead of np.random directly

def default_rng(seed=None):
    return np.random.default_rng(seed)