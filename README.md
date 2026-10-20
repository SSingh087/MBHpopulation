# Overview

This repository contains analyses of the supermassive black hole (MBH) population constrained through extreme mass-ratio inspirals (EMRIs) observed by LISA and tidal disruption events (TDEs) observed by ZTF and LSST.

## Science goal

MBHs in galactic nuclei are the source of both EMRIs and TDEs, so the two channels trace the same underlying population and probe it in complementary ways: EMRIs through the MBH mass, spin, the eccentricity of the orbit at plunge and the redshift distribution, and TDEs through the MBH mass and the redshift distribution. We ask what the detected events can tell us about the MBH population, and how much is gained by combining both channels.

## Approach

1. **Mock universes.** Galaxies are drawn from the cosmological mass functions, each with a nuclear star cluster and an MBH, for different MBH mass relations, nucleation and occupation fractions, and galaxy merger-rate models. The cusp evolution gives the EMRI and TDE events each galaxy produces.
2. **Detectability.** Neural-network emulators give the LISA signal-to-noise ratio of each EMRI and the probability that ZTF or LSST detects each TDE. The universe that best represents the expected number of detections across all scenarios is selected.
3. **Measurement.** A Fisher analysis of every detected EMRI and TDE gives its posterior on the MBH mass, redshift and, for EMRIs, spin and plunge eccentricity.
4. **Population inference.** A hierarchical Bayesian analysis infers the MBH population from these posteriors for two parametrised population models, using EMRIs only, TDEs only, and both together. The selection function of each channel, emulated by a neural network, enters the likelihood.

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
