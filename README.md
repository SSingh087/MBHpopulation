

# Overview 
This repository contains comprehensive analyses of supermassive black hole (SMBH) populations constrained through Extreme Mass Ratio Inspirals (EMRIs) and Tidal Disruption Events (TDEs) and mergers of supermassive black holes (MBHB). The work combines astrophysical population modeling with statistical inference to characterize black hole mass distributions and detection rates.

## Project Structure

### 1. **MBH_population_from_EMRI/**

This directory contains analyses focusing on constraining SMBH populations using EMRI observations from gravitational wave detectors like LISA.

**Key Components:**
- **`PLOTS/`**: Visualization outputs including posterior distributions, parameter distributions, and inference results
- **`scripts/plotting_scripts/`**: Python scripts for generating publication-quality figures
  - `combine_plots_*_posterior_PPD.py`: Scripts combining posterior and predictive distributions for different population models

**Scientific Purpose:** 
Uses EMRI observations to infer the properties and distribution of supermassive black holes across cosmic time, leveraging the sensitivity of LISA to extreme mass-ratio systems.

---

### 2. **MBH_population_from_EMRI_TDE/** *(Currently Private Research Code)*

This is the main research directory implementing a complete pipeline for modeling SMBH populations and their detectable signatures through both EMRI and TDE channels. .

**What to Expect Upon Public Release:**

This comprehensive pipeline produces several key outputs:

1. **Synthetic SMBH Population Datasets**
   - 1+ million simulated galaxies with SMBH properties across cosmic history
   - Full parameter sets including black hole masses, spins, stellar velocities, and merger histories
   - HDF5 and NumPy formats for easy integration with machine learning and statistical analyses

2. **Event Rate Predictions**
   - Detectable EMRI rates as functions of SMBH mass, redshift, and observation time
   - TDE rates from various stellar environments (ZTF, LSST sensitivity)
   - Event rate evolution across cosmic time accounting for galaxy mergers and dynamical evolution

3. **Trained Neural Network Models**
   - SNR prediction models for parameter estimation (initial orbital parameters p₀, eccentricity, etc.)
   - TDE detectability classification models
   - Models trained on 10⁶ synthetic events with optimized architectures

4. **Fisher Information Analysis**
   - Parameter measurement uncertainties for EMRI/TDE observations
   - Sky localization and luminosity distance constraints
   - Fundamental limits on SMBH population parameter inference

5. **Posterior Distributions & Inference Results**
   - Hierarchical Bayesian posteriors for population parameters
   - Comparisons between different population models (Models A, B, MIX)
   - EMRI-only, TDE-only, and joint EMRI+TDE inference results
   - Corner plots and posterior predictive distributions

6. **Diagnostic Plots & Validation**
   - Galaxy stellar mass functions vs. SMBH mass functions
   - Relaxation timescale evolution
   - Merger clustering analysis
   - Population distribution validation


---

## Citation

If you use this work, please cite:

```bibtex
@article{Singh:2025pzh,
    author = "Singh, Shashwat and Chapman-Bird, Christian E. A. and Berry, Christopher P. L. and Veitch, John",
    title = "{Constraints on the extreme mass-ratio inspiral population from LISA data}",
    eprint = "2508.16399",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "8",
    year = "2025"
}
```

See [`MBH_population_from_EMRI/CITATION.bib`](./MBH_population_from_EMRI/CITATION.bib) for the complete bibliography and related references.

---

## Data Availability

Posterior samples from the inference analysis are available at:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18197326.svg)](https://doi.org/10.5281/zenodo.18197326)

---
