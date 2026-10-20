

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

### 2. **MBH_population_from_EMRI_TDE/**

Pipeline that builds a mock galaxy population with nuclear star clusters (NSCs), predicts the EMRI and TDE events each galaxy produces, decides which are detectable by LISA (EMRIs) and ZTF/LSST (TDEs), and infers the MBH population hierarchically from the detected events.

Scripts are written to be run from the `MBH_population_from_EMRI_TDE/` directory. Generated data is expected under `./data` (not tracked).

| Directory / file | Content |
|---|---|
| `astrophysical_setup/` | Cosmology, mass functions, NSC, cusp evolution, EMRI/TDE rate and merger models |
| `utils.py` | Shared helpers and data paths for the generation pipeline |
| `data_generation/` | Event generation (`sources_num.py`), EMRI/TDE parameters and detection, ZTF/LSST footprints |
| `data_gen_realistic/` | Event counts for realistic (thinned) observing conditions |
| `training_EMRI_SNR/` | Neural-network emulator of the EMRI SNR |
| `training_TDE_detectability/` | Neural-network classifier for TDE detectability per survey and light-curve model |
| `fisher_analysis/` | Measurement uncertainties (Fisher) for EMRIs and TDEs |
| `hierarchical_inference/` | Hierarchical Bayesian inference (populations A and B), with `plots/` for posteriors, violin plots and PPDs |
| `hierarchical_inference_altered/` | Variant of the inference with a different scenario setup, same layout |
| `tests/` | Validation: `TEST_merger_rates`, `TEST_time_average`, `PPS_VERIFICATION`, `beta_test` |
| `plots/` | Diagnostic plots and `plots/paper/` figure scripts |
| `EXECUTABLES/`, `SUBMIT_FILES/` | Shell wrappers (`config.sh` holds the shared settings) and HTCondor submit files |
| `make_data_gen_dag.py`, `make_fisher_TDE_dag.py`, `report_universes.py` | Generate the DAGs and pick the best universe across scenarios |

**Typical workflow**

1. Train the surrogates in `training_EMRI_SNR/` and `training_TDE_detectability/`.
2. `python make_data_gen_dag.py` writes the multi-universe DAG (events, parameters, detection, footprints, `report_universes.py`); submit it with `condor_submit_dag`.
3. `make_fisher_TDE_dag.py` and `fisher_analysis/` give measurement errors for the detected events.
4. In `hierarchical_inference/`, `gen_dag.py` writes the inference and plotting DAG.

Dependencies include numpy, scipy, h5py, matplotlib, torch, [FastEMRIWaveforms](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms), [redback](https://github.com/nikhil-sarin/redback) and `poplar`.

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
