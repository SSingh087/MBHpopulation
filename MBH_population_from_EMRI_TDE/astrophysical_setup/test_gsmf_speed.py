#!/usr/bin/env python
"""Benchmark for GSMF optimization"""
import numpy as np
import time
from cosmology import GalaxyStellarMassFunction

# Initialize
gsmf = GalaxyStellarMassFunction()

# Test case: similar to gen_events.py
N_objs = 1000
z_grid = np.random.uniform(0.1, 5.0, N_objs)

print(f"Testing with {N_objs} galaxies across different redshifts...")
print(f"Mass grid size: 1000 points")
print()

# Test the optimized sample_gsmf
start = time.time()
lgMgal_samples = gsmf.sample_gsmf(z_gal=z_grid, size=1000)
elapsed = time.time() - start

print(f"✓ sample_gsmf() completed in {elapsed:.4f} seconds")
print(f"  Output shape: {lgMgal_samples.shape}")
print(f"  Sample values (first 10): {lgMgal_samples[:10]}")
print()

# Verify the samples are in the valid range
valid_range = (gsmf.lgMgal_data.min(), gsmf.lgMgal_data.max())
in_range = np.all((lgMgal_samples >= valid_range[0]) & (lgMgal_samples <= valid_range[1]))
print(f"✓ All samples in valid range {valid_range}: {in_range}")
print()

# Test get_gsmf still works
start = time.time()
lgMgal_grid, phi = gsmf.get_gsmf(z_grid[:10], n_points_mass=1000)
elapsed = time.time() - start

print(f"✓ get_gsmf() for 10 redshifts completed in {elapsed:.4f} seconds")
print(f"  Grid shape: {lgMgal_grid.shape}")
print(f"  Phi shape: {phi.shape}")
