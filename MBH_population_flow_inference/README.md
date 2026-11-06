# MBH population from EMRI and TDE sources: flow-based inference

Learn the distribution of EMRI and TDE source parameters, p(x | Lambda), with a conditional normalizing
flow written from scratch in PyTorch, where Lambda is the MBH-population physics vector: M-sigma
normalisation `A`, slope `B`, intrinsic scatter, and the occupation-fraction parameters `logM0`, `delta`,
`theta`. The nucleation fraction is fixed. The flow is intended to replace the phenomenological
population-shape fit in the hierarchical inference of `../MBH_population_from_EMRI_TDE`.

Project notes (settings, priors, decisions, findings from the old code) are kept outside the repo in `/home/2673888s/EMRI_TDE_flow_inference/{CONFIG,PLAN}.md`.

## Layout

| Folder | Content |
|---|---|
| `src/` | `source_sim.py`: `simulate_sources(Lambda, seed)` returns the EMRI and TDE source populations of one universe (reuses `../MBH_population_from_EMRI_TDE/astrophysical_setup`, imported unchanged) |
| `sources/` | smoke test, validation against the existing pipeline's catalogs, HTCondor DAG writer and worker for generating many universes |
| `flow/` | spline-flow library (`flows.py`), tests, dataset builder, training, hyperparameter-sweep DAG, selection and acceptance test (see below) |

## Simulator modes

`simulate_sources(..., legacy=False)` (default) draws each galaxy's sigma and M_BH once and carries them through.
`legacy=True` reproduces two quirks of the existing pipeline (a second, independent sigma draw inside
`Galaxy.lgMBH`, and a redraw of every selected event followed by the mass cut), and matches its catalogs
(expected counts within a few percent, marginals KS-consistent).

## Generating the training set

```
python sources/make_dag.py --n_universes 10000 --per_job 50
cd $DATA/dag/sources_full && condor_submit_dag sources_full.dag
```

Data, models, plots and logs live outside the repository (`/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference/`).

## Training the flows

```
python flow/build_dataset.py && python flow/plot_training_data.py      # consolidate the universes, overview plots
python flow/make_sweep_dag.py --n_configs 12                           # writes the sweep DAG (nothing is submitted)
cd $DATA/dag/flow_sweep && condor_submit_dag flow_sweep.dag
```
The DAG trains `n_configs` random-search configurations per population (config 00 is the baseline) on one GPU each,
then `select_best.py` ranks them by validation NLL, copies the winner to `models/flow_<pop>_best/` and plots its
diagnostics, and finally `acceptance.py` compares the winners with freshly simulated held-out universes at five
Lambda values (corner overlays, marginals, KS statistics). Plots: `plots/sweep_<pop>/` (comparison of all configs,
plus each config's loss curve), `plots/flow_<pop>_best/`, `plots/acceptance_best/`.
