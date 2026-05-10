#!/usr/bin/env python
"""Test to verify GSMF sampling is unbiased (not skewed towards low masses)"""
import numpy as np
import matplotlib.pyplot as plt
from cosmology import GalaxyStellarMassFunction

# Initialize
gsmf = GalaxyStellarMassFunction()

# Test at a single redshift to check the distribution
z_test = 0.5
N_samples = 10000

# Draw many samples at same redshift
z_grid = np.full(N_samples, z_test)
lgMgal_samples = gsmf.sample_gsmf(z_gal=z_grid, size=1000)

# Get the true GSMF at this redshift for comparison
lgMgal_grid, phi = gsmf.get_gsmf(np.array([z_test]), n_points_mass=1000)
phi_true = phi[0]  # (1000,)

print(f"Testing sampling bias at z = {z_test}")
print(f"Drew {N_samples} samples")
print()
print(f"Sample statistics:")
print(f"  Mean: {np.mean(lgMgal_samples):.4f}")
print(f"  Std:  {np.std(lgMgal_samples):.4f}")
print(f"  Min:  {np.min(lgMgal_samples):.4f}")
print(f"  Max:  {np.max(lgMgal_samples):.4f}")
print()

# Create histogram of samples
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: histogram of samples vs true GSMF
ax = axes[0]
counts, bins, _ = ax.hist(lgMgal_samples, bins=50, density=True, alpha=0.7, label='Samples', color='blue')
ax.plot(lgMgal_grid, 10**phi_true / np.sum(10**phi_true), 'r-', linewidth=2, label='True GSMF (normalized)')
ax.set_xlabel('log10(Mgal) [Msun]')
ax.set_ylabel('Probability density')
ax.set_title(f'Sampling Distribution vs True GSMF (z={z_test})')
ax.legend()
ax.grid(True, alpha=0.3)

# Right: log-scale comparison
ax = axes[1]
ax.semilogy(lgMgal_grid, 10**phi_true, 'r-', linewidth=2, label='True GSMF')
counts_normalized = counts * (bins[1] - bins[0])  # normalize by bin width
bin_centers = (bins[:-1] + bins[1:]) / 2
ax.semilogy(bin_centers, counts_normalized, 'b-', linewidth=2, label='Samples (histogram)')
ax.set_xlabel('log10(Mgal) [Msun]')
ax.set_ylabel('Phi (log scale)')
ax.set_title(f'GSMF Comparison (log scale)')
ax.legend()
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('gsmf_sampling_test.png', dpi=150)
print(f"✓ Saved plot to gsmf_sampling_test.png")
print()

# Quantitative check: KL divergence
# Compute histogram PDF using consistent binning
n_bins = 50
hist_counts, bin_edges = np.histogram(lgMgal_samples, bins=n_bins, range=(lgMgal_grid.min(), lgMgal_grid.max()))
hist_pdf = hist_counts / np.sum(hist_counts)  # normalize to probability

# Interpolate true PDF at bin centers for comparison
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
true_pdf_interp = np.interp(bin_centers, lgMgal_grid, 10**phi_true)
true_pdf_interp = true_pdf_interp / np.sum(true_pdf_interp)  # normalize

# Avoid log(0)
eps = 1e-10
hist_pdf = np.clip(hist_pdf, eps, 1.0)
true_pdf_interp = np.clip(true_pdf_interp, eps, 1.0)

kl_div = np.sum(hist_pdf * (np.log(hist_pdf) - np.log(true_pdf_interp)))
print(f"KL divergence (samples || true): {kl_div:.6f}")
print(f"  (values close to 0 indicate good agreement)")
print()

# Check for low-mass bias: compare median of samples vs median of true distribution
from scipy.interpolate import interp1d

# Compute CDF of true distribution
true_pdf = 10**phi_true / np.sum(10**phi_true)
cdf_true = np.cumsum(true_pdf)
cdf_true = cdf_true / cdf_true[-1]
median_true_idx = np.argmin(np.abs(cdf_true - 0.5))
median_true = lgMgal_grid[median_true_idx]

median_samples = np.median(lgMgal_samples)

print(f"Median comparison:")
print(f"  True GSMF median:    {median_true:.4f}")
print(f"  Sample median:       {median_samples:.4f}")
print(f"  Difference:          {median_samples - median_true:.4f}")
if abs(median_samples - median_true) > 0.1:
    print(f"  ⚠ WARNING: Median shifted by {abs(median_samples - median_true):.4f} dex (possible bias)")
else:
    print(f"  ✓ Good agreement (< 0.1 dex)")
