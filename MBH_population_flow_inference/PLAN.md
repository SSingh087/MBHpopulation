# PLAN.md — Part 0 findings

Existing project (read-only): `/home/2673888s/EMRI_TDE_population`. Nothing there was modified.

## Blockers / things you must decide (read first)

1. **Plan doc not found.** CONFIG.md rows depending on your Config table are marked. Please paste it or point me to it.
2. **The production EMRI detectability is NOT the trained NN.** `make_data_gen_dag.py` → `prepare_EMRIs.sh` →
   `data_generation/calculate_SNR_EMRI_pseudo.py` uses an *analytic* SNR (Moore-Cole-Berry h_c, Robson+19 PSD,
   Keplerian f_gw from p0) with threshold **SNR > 20**. The trained net exists but nothing in the current
   pipeline calls it. Which should the flow be trained on? (Both interfaces documented below.)
3. **Nucleation choice** (§3) — recommendation given, needs your OK.
4. **No callable generator exists.** Catalog generation is a CLI script with module-level code, and A/B/scatter are
   not arguments anywhere (§1). Part 1 must write a wrapper; it cannot just call a function.
5. **Superseded: user chose NOT to use zuko; the flow will be written from scratch in PyTorch.** (For the record, zuko is NOT installed in `redback_few` (verified: torch 2.4.0, CUDA available, `import zuko` fails). Nothing to install.)
6. **Latent inconsistency in the existing generator** (§1.4): the MBH mass used for the rate weights is not the one saved in the catalog.

## 1. EMRI catalog generation (existing code)

### 1.1 Pipeline (per universe, driven by `make_data_gen_dag.py` / `EXECUTABLES/*.sh`)
1. `data_generation/sources_num.py` (CLI, argparse; no importable function). Args:
   `--UNIVERSE_MODEL --GALAXIES --Z_MAX --T_OBS_EMRI --T_OBS_TDE --mbh_mass_method {use_sigma,use_lgMgal}
   --nucleation_method {hannah,neumayer} --occupation_method {uniform,varying} --merger_model {mundy17,oleary21,duan24,baseline} --SEED`.
   Draws z ∝ dVc/dz, lgMgal uniform over a pilot-derived band, importance weights `mc_weight_i`
   (GSMF φ × V_tot × width / N_sim), then `Galaxy.check_nucleation` → `Galaxy(...)` → NSC → rate;
   Gamma_EMRI_i = rate·T_obs·active/(1+z) masked to 4<lgMBH<=7 & occupied; N ~ Poisson(Lambda_tot);
   PPS resample by `mc_weight × Gamma`. Writes `galaxy_catalog_data_{mbh}_{nuc}_{occ}_{merger}_{U}.h5`
   (`EMRI_catalog/{lgMgal,z_gal,lgMBH,sigma_km_s,ra_deg,dec_deg,initial_MBHspin,eccentricities_at_plunge,...}`;
   attrs `Lambda_EMRI_tot`, `n_eff_EMRI`, `N_gal_true`).
2. `data_generation/gen_EMRI_params.py` (CLI): adds Y0 ~ U(Y0 range), T_SIGNAL, angles, phases; backward-integrates
   (FEW `EMRIInspiral`, pn5) to get e0, p0 (fallback e0~U(0.05,0.5), p0=NaN → never detected). Writes `all_EMRI_params_data_*.h5`
   (`lgMBH_mass, distance_Gpc, z_gal, MBHspin, qS, phiS, sBH_mass, e0, p0, eccentricities_at_plunge, Y0, qK, phiK, Phi_*, T_SIGNAL_duration_years`).
3. `data_generation/calculate_SNR_EMRI_pseudo.py` (CLI, `--SNR_THRESHOLD` default 20): writes `EMRI_SNRs_data_*.h5`
   = only SNR-passing rows + `SNR`, `f_gw_det_Hz`.

### 1.2 Key classes (importable, from `astrophysical_setup/`, needs `sys.path` insert)
- `Galaxy(z_gal, ra_deg=None, dec_deg=None, rng=None, lgMgal=None, lgMBH=None, sigma_pc_yr=None, nucleation_occurs=True, mbh_mass_method='use_sigma', nucleation_method='hannah', occupation_method='uniform', occupation_occurs=True, occupation_value=1.0)`
- `Galaxy.check_nucleation(lgMgal, nucleation_method='hannah', rng=None)` → bool mask (static)
- `Galaxy.check_occupation(lgMgal, method='uniform'|'varying', value=1.0, rng=None)` → bool mask (static)
- `NSC(galaxy)`, `MBH_properties(nsc)` (spin ~ Beta, mass-dependent), `CompactObject(...)`, `RateModel`, `ConstantRate`, `CuspEvolution`, `LastMajorMerger(cosmo, merger_model=)`.
- Eccentricity: `emri_parameters(M)`/`sample_emri_eccentricity(M)` are **defined inside sources_num.py** (not importable without running the script) — Beta mixture, mass-dependent weight.

### 1.3 How Lambda enters — what must be patched
- **A, B, scatter**: read from `config.py` module constants (`MBH_A_sigma=7.87, MBH_B_sigma=4.55, MBH_scatter_sigma=0.55`),
  bound at import via `from config import *` into the `galaxy` namespace, then copied to `self.MBH_A/B/MBH_scatter`
  in `Galaxy.__init__`. **`MBH_properties(A=,B=,MBH_scatter=)` args are ignored.** Part 1 options:
  (a) set `gal.MBH_A/B/MBH_scatter` is too late (lgMBH is computed in `__init__`), so
  (b) monkeypatch `galaxy.MBH_A_sigma` etc. before constructing `Galaxy` (works, but global state), or
  (c) cleanest: add optional args to a *copy* of Galaxy in the new folder. Recommend (c); do not edit the old project.
- **Occupation**: `_f_occupation_varying_(lgMgal, logM0=9.46, delta=3.91, theta=0.80)` has the right parameters but
  `check_occupation(method='varying')` calls it with defaults only → need a wrapper passing (logM0, delta, theta).
- **Nucleation**: fixed in v1, so no change.
- sigma_0=160 km/s is a config constant.

### 1.4 Issues found in the existing generator (not fixed, flag only)
- `sources_num.build_event_catalog` rebuilds `Galaxy(lgMgal=lgMgal_event, ...)` for the PPS-selected events, which **re-draws
  Re, sigma and the M-sigma scatter**. Rate weights were computed with the *first* draw of lgMBH, but the saved catalog lgMBH is a
  fresh draw. The saved (mass, z) distribution therefore is not exactly the weighted one. For a flow that learns p(x|Lambda) this
  matters (it changes the effective scatter in the learned mass distribution). Part 1 should draw lgMBH once and carry it through.
- `paper.tex` says scatter 0.53; `config.py` 0.55; `nsc.MBH_properties` default 0.53 (unused). Use Table 5 value 0.55.
- `np.random.poisson`, `np.random.normal`, `np.random.beta` (global state) are used alongside the seeded `rng` → runs are not fully
  reproducible from `--SEED`.
- Spin prior in catalog is a Beta with lgMBH-dependent alpha; training range in PARAMS_RANGE.txt: spin 0.1–0.9, log M 4–10, z 0–8.

## 2. Trained SNR detectability net (inspected directly)

Path: `/data/wiay/postgrads/shashwat/EMRI_data/trained_models/MODEL_theta_SNR_config_MSE_no_rescaler_2048_4_500/best_final_model/model/model.pth`
- Library: `poplar` (`/home/2673888s/poplar`, `poplar.nn.networks.LinearModel`). **Whole-object pickle**, load with
  `from poplar.nn.networks import load_model; m = load_model(path, device)`; needs `PYTHONPATH=/home/2673888s/poplar`. Loads fine in `redback_few` (verified).
- Architecture: MLP 11 → 2048 ×4 hidden (SiLU, dropout 0.2) → 1. Rescaler: `IdentityRescaler` (no input normalisation).
- Run: `m.run_on_dataset(x_tensor)` (eval mode, no_grad).
- **Output** = min-max-scaled log10(SNR / D_Gpc): `y = (log10(SNR/D) − min)/(max − min)` with
  min = −7.109823, max = 2.549535 (from `snrs_dist_log_scaled_COMBINED.npy`; these constants are not stored with the model).
  Invert: `SNR = D_Gpc · 10**(y·(max−min) + min)`.
- **Inputs (11)**: the training array `injection_params_dist_REMOVED_COMBINED.npy` (distance removed). Empirical ranges of the 12 pre-removal columns:
  c0 lgM 6–10; c1 1–2; c2 0.1–0.7; c3 6.3–27.2; c4 0.068–0.7; c5 0.096–0.7; c6 dist 1–8 Gpc (removed); c7 0.1–3.11; c8 0.1–6.22; c9 0.1–3.11; c10 0.1–6.22; c11 1–6 (T_signal).
  **Column semantics/order could not be pinned down**: the scripts disagree with each other and with the saved arrays
  (`gen_theta_ex_p0.py` says lgM 5–10, a 0.1–0.9, Y0 0.3–0.99; the saved array has lgM≥6, c2≤0.7). Likely
  (lgM, ?, a, p0, e0, Y0, qS, phiS, qK, phiK, T) but c1 (range 1–2) is unidentified. Must be resolved by reading `gen_theta_ex_p0.py`/`eval_p0.py` in Part 1 before use.
- **Coverage problem**: training lgM ≥ 6 whereas the EMRI catalog is 4 < lgMBH ≤ 7, so the net would extrapolate for most low-mass events. Another reason the analytic SNR is what production uses.
- The net takes p0, e0 (requires the FEW backward integration), which the flow will not have for a generic catalog — it is not cheap either.

Recommendation for v1: use the same detector as the published pipeline (analytic SNR>20, `calculate_SNR_EMRI_pseudo.py` logic, vectorised and importable) unless you say the NN must be used.

## 2b. Other infrastructure
Detected-count scale from the pipeline: `GALAXIES=100000`, `T_OBS_EMRI=6 yr`, `N_UNIVERSES=100`, HTCondor DAGs, outputs under `/data/wiay/postgrads/shashwat/EMRI_TDE_data/astrophysical_data/<GALAXIES>/`.
Old PRE_TRAIN_DATA under `EMRI_TDE_data` is empty; the real SNR training data is in `EMRI_data/PRE_TRAIN_DATA/1E6_events`.

## 3. Nucleation choice (not a free parameter)

Pipeline supports `hannah` (arXiv:2407.10911) and `neumayer` (arXiv:2001.03626); the project sweeps both. No fiducial is declared in code or paper.tex.
- Hannah: f_nucl ≈ 1 down to lowest sampled masses → the occupation fraction alone sets the low-mass hosting probability. **This makes the sample most informative about (logM0, delta, theta)**, which are Lambda parameters.
- Neumayer: f_nucl ≲ 0.2 at low mass → occupation signal is suppressed and expected EMRI counts are small (the paper's `lgMgal/neumayer` scenarios have very low counts).
- **Recommendation: `hannah`, form as-is, fixed.** Caveat: both parametrisations in galaxy.py are "fit to eyeballed points, NOT digitized". **Awaiting your confirmation.**

## 4. Scaling relations — Greene, Strader & Ho 2020 (ARA&A 58:257) supplement, Table 5 (read from `papers/aa58_greene_supmat.pdf`)

Fit: log MBH = α + β log(σ/160 km/s) + ε ; log MBH = α + β log(M*/3e10 Msun) + ε. **Ten distinct relations; there is no single relation with one error bar.**

| Fit | Sample | α | β | ε |
|---|---|---|---|---|
| MBH–σ | All, no limits | 7.88 ± 0.05 | 4.34 ± 0.24 | 0.53 ± 0.04 |
| MBH–σ | **All, limits** (used by code) | **7.87 ± 0.06** | **4.55 ± 0.23** | **0.55 ± 0.04** |
| MBH–σ | Early | 8.03 ± 0.06 | 4.24 ± 0.25 | 0.43 ± 0.04 |
| MBH–σ | Late, no limits | 7.40 ± 0.10 | 2.54 ± 0.50 | 0.50 ± 0.07 |
| MBH–σ | Late, limits | 7.44 ± 0.12 | 3.61 ± 0.50 | 0.58 ± 0.09 |
| MBH–M* | All, no limits | 7.56 ± 0.09 | 1.39 ± 0.13 | 0.79 ± 0.05 |
| MBH–M* | **All, limits** (used by code, `use_lgMgal`) | **7.43 ± 0.09** | **1.61 ± 0.12** | **0.81 ± 0.06** |
| MBH–M* | Early | 7.89 ± 0.09 | 1.33 ± 0.12 | 0.65 ± 0.05 |
| MBH–M* | Late, no limits | 6.94 ± 0.13 | 0.98 ± 0.27 | 0.60 ± 0.08 |
| MBH–M* | Late, limits | 6.70 ± 0.13 | 1.61 ± 0.24 | 0.65 ± 0.09 |

Choice made (matches the code): **MBH–σ, "All, limits"**: A=7.87±0.06, B=4.55±0.23, ε=0.55±0.04.
Quoted errors are 1σ-type symmetric; correlations between α and β are not given (they are correlated at fixed pivot only weakly, but unquantified here).
Note the α/β error bars are fit uncertainties; the "late"/"early" rows differ by far more than these errors, so the choice of relation is a systematic larger than the statistical error.

Occupation (Zou+2025) values/uncertainties are not in `papers/` as a table I verified; they come from `paper.tex` (logM0 = 9.46 +0.32/−0.51, delta = 3.91 +4.66/−2.53, theta = 0.80 +0.55/−0.30). Verify against the Zou PDF (`Central Massive Black Holes Are Not Ubiquitous in Local Low-Mass Galaxies.pdf`) in Part 1 — I did not read it.

## 5. Proposed next steps (Part 1, pending your confirmation)
1. Write the conditional spline flow from scratch in PyTorch (no zuko).
2. Write an importable `simulate_catalog(Lambda, seed)` in the new folder (copy needed pieces of astrophysical_setup; add A/B/scatter and occupation-parameter arguments; draw lgMBH once; seeded RNG throughout), returning detected-EMRI features + the per-Lambda expected count mu(Lambda).
3. Detector = vectorised analytic SNR>20 (or the NN, per your decision).
4. Validate against the old pipeline at Lambda = fiducial (count and (M, z, spin, e) marginals).
