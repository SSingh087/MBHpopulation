"""
config.py

Central location for physical constants, unit conversions,
and default model parameters used throughout the NSC+Galaxy+Cosmology codebase.

This file preserves exactly the constants and behaviors implied
in your original code, without changing the physics.
"""

import numpy as np


# Unit Conversions

# Length
pc_to_cm = 3.085677581491367e18      # parsec → cm
kpc_to_cm = 1e3 * pc_to_cm           # kiloparsec → cm

# Time
sec_per_year = 3.154e7               # s / yr  (consistent with your usage)

# Mass
Msun_to_grams = 1.98847e33           # solar mass → grams

# Speed of light conversions
c_cgs = 2.99792458e10                # cm/s
c_pc_per_year = c_cgs * sec_per_year / pc_to_cm   # pc / yr

# Gravitational Constant in Multiple Unit Systems

G_cgs = 6.67430e-8                    # cm^3 g^-1 s^-2

# G in pc^3 / (Msun * yr^2
G_pc3_per_Msun_yr2 = G_cgs * (sec_per_year**2) / (pc_to_cm**3 / Msun_to_grams)

# M–sigma relation defaults
MBH_A = 7.87
MBH_B = 4.55
MBH_sigma0 = 160.0
MBH_scatter = 0.53

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