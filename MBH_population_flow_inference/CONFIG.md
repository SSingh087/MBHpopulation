# CONFIG.md — EMRI_TDE_flow_inference (v1)

Source: your plan doc Config table (`chat_output.txt`) + Part 0 findings (PLAN.md). All later parts read this file.

## Paths

| Setting | Value |
|---|---|
| Existing project (read-only) | `/home/2673888s/EMRI_TDE_population` (confirmed) |
| Scripts dir | `/home/2673888s/EMRI_TDE_flow_inference` (papers, chat) and the git repo folder `MBHpopulation/MBH_population_flow_inference` |
| Data/models/plots/logs dir | `/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference/` with `simulations/ models/ plots/ logs/ dag/` (created) |
| Conda env | `redback_few` (python 3.12, torch 2.4.0, CUDA available). No flow package needed: flow is written from scratch in PyTorch |
| Papers | `/home/2673888s/EMRI_TDE_flow_inference/papers/` |
| Scheduler | HTCondor DAG |

## Scope and likelihood

| Setting | Value |
|---|---|
| v1 scope | EMRI only |
| Flow library | **none: conditional flow implemented from scratch in plain PyTorch** (changed from zuko, per user 2026-11-06). No zuko, no sbi, no SNPE/SNRE |
| Catalog embedding | DeepSets sum-pool + explicit event count N |
| Likelihood (later parts) | Poisson(N_obs \| mu(Lambda)) × Π p_flow(x_i \| Lambda) |
| **Current priority** | **Stage 1 only: p_flow(x \| Lambda). No mu(Lambda)/likelihood yet** |

## Lambda (inferred): fiducial ± quoted uncertainty

M–σ relation: log10(M/Msun) = A + B log10(σ/160 km/s) + ε, ε ~ N(0, scatter). Greene, Strader & Ho 2020, Supp. Table 5, row "All, limits"
(the paper gives 10 distinct relations; this is the one the code uses).

| Param | Fiducial | Uncertainty | Source |
|---|---|---|---|
| A | 7.87 | ± 0.06 | Greene+2020 Supp. T5 |
| B | 4.55 | ± 0.23 | same |
| scatter | 0.55 dex | ± 0.04 | same (paper.tex's 0.53 is the "no limits" row; not used) |
| logM0 | 9.46 | +0.32 / −0.51 | Zou+2025 arXiv:2510.05252 |
| delta | 3.91 | +4.66 / −2.53 | same |
| theta | 0.80 | +0.55 / −0.30 | same |

Occupation form: f_occ = (1 + δ exp(−θ δ (lgMgal − logM0)))^(−1/δ).

Alternative M–M* relation (recorded, NOT used in v1): A 7.43 ± 0.09, B 1.61 ± 0.12, scatter 0.81 ± 0.06 (pivot M0 = 3e10 Msun).

## Lambda prior box (v1, uniform unless stated)

| Param | Range |
|---|---|
| A | [7.57, 8.17] |
| B | [3.25, 5.85] |
| scatter | [0.35, 0.75] dex |
| logM0 | [8.5, 10.5] |
| delta | [0.5, 20], **log-uniform** |
| theta | [0.2, 2.0] |

## Fixed in v1 (not inferred)

| Setting | Value |
|---|---|
| Nucleation method | `hannah`, functional form as-is: `exp(-(abs(lgMgal-7.39)**6.7)/3051.82)` (galaxy.py; fit to eyeballed points, not digitised; no uncertainty) |
| mbh_mass_method | `use_sigma` |
| Detector | **analytic SNR > 20** (logic of `calculate_SNR_EMRI_pseudo.py`), NOT the trained NN (lgM≥6 training coverage vs catalog 4–7; ambiguous input columns) |
| Merger model | `oleary21` |
| T_OBS_EMRI | 6 yr |
| Z_MAX | 8.0 |
| MBH window | 4 < lgMBH ≤ 7 |
| sigma_0 | 160 km/s |
| Size–mass | log10 Re = 0.82 + 0.24(lgMgal−10.7), 0.20 dex scatter; kvir=1, M_enc=0.5 M* |
| Compact object | sBH 10 Msun |
| Y0 range | 0.1–0.7; T_SIGNAL ∈ [1, T_OBS_EMRI] yr |
| MC galaxy pool (GALAXIES) | 100000 (production value; tune per draw in Part 1) |
| Known bug to fix in new wrapper | existing generator redraws lgMBH + M–σ scatter after PPS resampling → draw lgMBH once and carry it through |
| RNG | single seeded `np.random.Generator` throughout (old code mixes in global `np.random`) |

## Flow & training settings

| Setting | Value |
|---|---|
| x per detected event | log10(MBH), spin a, redshift z, eccentricity at plunge, log10(SNR) |
| Embedding | per-event MLP [5→64→64], sum-pool, concat log(N+1), post-pool MLP [65→64→32] → context vector |
| Flow | from-scratch PyTorch conditional neural spline flow (rational-quadratic splines, autoregressive/coupling layers written by hand), ~6 conditional transforms, hidden width 64, 8 bins |
| Lambda transforms before flow | logit for [0,1]-type params; log for scatter and delta (theta/occupation-shape handled per plan; see note) |
| Smoke test (Part 1) | ~30 Lambda draws |
| Full training set (Part 2) | ~5,000–10,000 draws |
| Train/val | 90/10 |
| Optimiser | Adam, lr 1e-3, ReduceLROnPlateau; batch 64–256 catalogs/step; early stop patience ~20; max 200 epochs |
| Stage-1 acceptance | 4–5 Lambda values (fiducial + near edges); held-out simulated catalog vs flow samples; overlay marginal/corner plots (mass, spin, z, ecc) |
| Resource rule | small test first; auto-proceed only if full run < ~30 min compute and a few GB disk; else report estimate + command and stop |

Notes / open details for Part 1 (not blockers):
- The plan says "logit-transform occupation-fraction-shape params": none of logM0/delta/theta is natively in [0,1]. Interpretation used unless you say otherwise: Lambda is rescaled to the prior box (unit interval), then logit; delta and scatter are log-transformed first.
- **Cost warning for Part 1 timing:** the analytic SNR needs p0 and e0, which `gen_EMRI_params.py` gets from a FEW backward trajectory integration per event. That loop is the expected bottleneck; Part 1 will time it per draw, and per-draw event count is ~Lambda_EMRI_tot (before detection).
- With analytic SNR the x-vector's log10(SNR) is included as specified; p0/e0 are not part of x.

## Part structure (re-scoped by user, 2026-11-06; supersedes the plan doc's Part 1-5 numbering)

| Part | Content |
|---|---|
| Part 1 | Source generation only: physics (Lambda -> galaxies -> nucleation/occupation -> M-sigma -> EMRI source population). Get the *source* distribution (mass, spin, z, eccentricity) correct and validated. No SNR/detectability |
| Part 2 | Detectability (analytic SNR > 20, p0/e0 handling) applied on top of Part 1 sources |
| Part 3 | Inference (flow p(x \| Lambda), then likelihood) |

Consequence: Part 1 smoke test saves the full (undetected) source catalog per Lambda draw with the expected count Lambda_EMRI_tot; the FEW p0/e0 integration cost is deferred to Part 2.

## Scope update (2026-11-06, user): EMRI + TDE sources, no detectability

- Generate EMRI and TDE **source** populations together (same galaxies, same Lambda, same cusp state), many seeded universes; flow learns p(x | Lambda) of the sources. Detectability is NOT part of this stage.
- EMRI x = (lgMBH, spin, z, e_plunge); TDE x = (lgMBH, z) (+ lgMgal, sigma stored); TDE spin not modelled. T_OBS_EMRI = 6 yr, T_OBS_TDE = 17.25 yr (as old code).
- Stored per universe: Lambda, exact expected counts (Lambda_tot, Lambda_tot_tde), all EMRI sources, random subsample <= 20000 TDE sources.
- Simulator: `source_sim.py` (clean mode default; `legacy=True` reproduces the old pipeline's double-sigma draw and post-PPS redraw).
- Part 1 validated: legacy mode matches old catalogs (EMRI & TDE counts within 1-4%, marginals KS-consistent); clean mode shifts lgMBH up ~0.8 dex and EMRI count +20% (TDE -6%) by removing those quirks. Smoke test: 30 draws, 5.4 s/draw, 165 KB/draw, 0 crashes, 0 empty EMRI catalogues (min N_EMRI = 6).

## Folder layout (keep new work in these places)

Code (git repo `SSingh087/MBHpopulation`, folder `MBH_population_flow_inference/`, branch `flow-inference`;
`/home/2673888s/EMRI_TDE_flow_inference/` keeps `papers/` and `chat*.txt`, which are NOT committed, plus symlinks to CONFIG.md/PLAN.md):
- `src/` importable library code (`source_sim.py`: EMRI + TDE source simulator)
- `sources/` source generation: `run_smoke_test.py`, `validate_against_old.py`, `run_chunk.py` (condor worker), `make_dag.py` (DAG writer)
- `flow/` from-scratch PyTorch conditional spline flow: `flows.py`, `test_flow.py`, `common.py`, `build_dataset.py`, `plot_training_data.py`, `train_flow.py`, `diagnostics.py`, `make_sweep_dag.py` (hyperparameter-sweep DAG), `select_best.py`, `acceptance.py`
- `CONFIG.md`, `PLAN.md`

Data `/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference/`:
- `simulations/smoke_test/`, `simulations/full_run/chunk_XXXX/univ_XXXXXX.npz`
- `dag/<name>/` (DAG, .sub, wrapper), `logs/<name>/` (condor logs; validation/smoke JSON in `logs/`), `plots/`, `models/`

Full source run: `python sources/make_dag.py --n_universes 10000 --per_job 50` (master seed 2026; universe g fully determined by (master_seed, g); resumable).

## Flow training plan (hyperparameter sweep)

`flow/make_sweep_dag.py --n_configs 12` writes a DAG (not submitted): per population, 12 random-search configurations
(n_layers {4,6,8,10}, hidden {32,64,128,256}, spline bins {6,8,12,16}, context width {32,64,128}, batch {2048,4096,8192}, lr log-uniform 3e-4..3e-3;
config 00 = baseline 6/64/8/64/4096/1e-3) -> `select_best.py` (lowest validation NLL; ranking + comparison plots; winner copied to `models/flow_<pop>_best/`)
-> `acceptance.py --tag _best` after 5 CPU jobs simulate the held-out universes. Validation = universes with id % 10 == 0.
Flow conditions on Lambda directly (the DeepSets catalog embedding is not needed for p(x | Lambda); it returns for the inference stage).

## Part 2: detectability models (copied 2026-11-06)

Models: `models/detectability/` (data dir); plots: `plots/detectability/`.

**EMRI SNR network** (`emri_snr/model.pth`, = `LISA_selection_function/lisa_snr_package/model/model.pth`, d3_w256, L1 loss, wd 1e-5, 2000 ep; held-out median error 0.083 dex in log10 SNR, p90 0.44 dex).
Inputs (raw): M, mu, a, e0, p0, Y0, t_back (+ z via Planck18 D_L). Network predicts SNR*D_L. Interface: `lisa_snr_package/snr_model.py`.
Training coverage: lgM 3.5-7, mu 3-120, a 0-1, e0 0-1, p0 3.4-260, Y0 -1..1, t_back 1-10 yr, **z 0.001-7**.
Versus our sources: lgMBH 4-7 OK, sBH 10 OK, t_back = T_SIGNAL in [1,6] OK, z>7 is 0.6% of EMRIs (treat as undetected). **e0 and p0 are not in the source catalogue**: need the FEW backward integration (or an e_plunge -> (p0,e0) emulator) at Part 2.

**TDE detectability classifiers** (8 BCE MLPs, `tde/<lightcurve>_<neurons>_<layers>_10_<SURVEY>/`, 4 light-curve models x ZTF/LSST).
Trained lgM 4-7 (matches ours), **z 0-0.5 for ZTF and 0-4 for LSST**. Our TDE sources go to z<8: 98.6% are above z=0.5 and 11.7% above z=4.
Detection fraction in the last training bins is small (ZTF z 0.4-0.5: 0.04-0.18; LSST z 3.5-4: 0.003-0.22), so events beyond the training z can be treated as undetected, with the tde_analytical / stream_stream_tde models (detection fraction flat ~0.2 out to z=4) the caveat.
Inputs besides lgMBH and z are nuisance parameters (ra, dec, eta, alpha, beta, t0, light-curve extras) drawn from priors in the old `gen_TDE_params.py`; survey windows ZTF MJD 58200-61396, LSST 60827-64423.

**Decision (user, 2026-11-06): TDE light-curve model = `tde_fallback`** (LSST `tde_fallback_128_4_10_LSST`, ZTF `tde_fallback_512_4_10_ZTF`). Its detection fraction is ~0 at the top of the trained z range (LSST z>3: <=0.7%; ZTF z 0.4-0.5: ~4%), so events beyond z=4 (LSST) / z=0.5 (ZTF) are simply treated as undetected.
