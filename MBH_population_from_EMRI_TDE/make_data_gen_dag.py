#!/usr/bin/env python
"""
Generates the multi-universe DAG: for each of the (mbh_mass_method,
nucleation_method) scenarios CROSSED WITH each occupation_method in
OCCUPATION_METHODS, run N_UNIVERSES fully independent realizations
(same scenario, different --UNIVERSE_MODEL) IN PARALLEL, each carried
through EMRI/TDE SNR/detection:

    gen_events(universe=U)
        -> gen_EMRI_params(universe=U) -> gen_EMRI_SNR(universe=U)
        -> gen_TDE_params(universe=U, model=M) -> gen_TDE_SNR_ZTF(universe=U, model=M)
                                                -> gen_TDE_SNR_LSST(universe=U, model=M)
                                                -> FOOTPRINT(universe=U, model=M)
           (for EVERY M in TDE_MODELS, in parallel)

TDE model is a per-JOB axis, not a per-event one: gen_TDE_params,
gen_TDE_SNR (both surveys), and FOOTPRINT each fan out into one job PER
(universe, TDE_MODEL) pair -- every model gets the SAME full source
TDE_catalog from gen_events (comparing models means running each one
independently on identical events, not splitting the catalog across
them). EMRI generation has no model axis and is completely unaffected
-- gen_events/gen_EMRI_params/gen_EMRI_SNR stay exactly as before, one
job per (scenario, universe).

DAG STRUCTURE:

  ALL gen_events/EMRI/TDE-params/TDE-SNR/FOOTPRINT jobs, across EVERY
  (mbh, nuc, occ, U, TDE_MODEL) combination, run in parallel and are
  ALL parents of ONE SINGLE REPORT_UNIVERSES job.

  REPORT_UNIVERSES is combined, not per-occupation-method: picking the
  winning UNIVERSE_MODEL only makes sense as a single maximin scan
  across every (scenario x occupation x channel) cell together. Model
  is diagnostic-only there -- report_universes.py sums each channel's
  passed-count across all 4 TDE_MODELS before running the existing
  (scenario x channel) maximin selection, and separately reports a
  per-model breakdown for the winning universe. Two separate report
  runs (one per occupation method, or one per model) could each pick a
  DIFFERENT winning universe, which defeats the entire point of "same
  universe used for every scenario" -- that guarantee has to span
  occupation method too, not just (mbh, nuc), and must stay independent
  of which TDE model is being compared.

  PREPARE_DATA_FOR_INFERENCE jobs (one per (mbh, nuc, occ) scenario)
  run ONLY AFTER REPORT_UNIVERSES completes -- they are its CHILDREN,
  not parallel siblings -- because they consume the winning universe's
  max_detected_*.h5 files, which REPORT_UNIVERSES is what produces.

  PULL_REAL_TDE_CATALOG and COMPARE_TDE_DETECTION run at the very end,
  also as children of REPORT_UNIVERSES: COMPARE_TDE_DETECTION reads
  every (scenario, universe, TDE_MODEL, survey) TDE_SNRs file plus the
  real-catalog overlay that PULL_REAL_TDE_CATALOG produces, so it must
  come after both.

MERGER-MODEL AXIS (galaxy major-merger-rate model, --merger_model in the scripts):
  The whole per-universe chain above is additionally CROSSED with every merger model in
  --merger_models (default: mundy17 oleary21 duan24; "baseline" = the original 0.25(1+z)^2 is
  opt-in). Every job name, VARS block ($(MERGER_MODELS) in the .sub files) and every output
  filename carries the merger-model name right after the occupation method. There is ONE
  REPORT_UNIVERSES_<mm> and ONE COMPARE_TDE_DETECTION_<mm> job PER merger model (winning
  universes only need to be consistent within a merger model, since a single model is chosen
  before inference). PREPARE_DATA_FOR_INFERENCE is deliberately NOT generated here: the
  merger-model decision is made beforehand, so nothing downstream of it (inference data prep,
  inference) sees this axis.

Every job's VARS and every output filename carries an
OCCUPATION_METHODS suffix after the existing mbh/nuc suffix, so that
runs with different occupation methods don't collide on disk, and
(for TDE-stage jobs) a TDE_MODEL suffix so the 4 models don't collide
either. Filename pattern:
    EMRI_SNRs_data_{mbh}_{nuc}_{occupation_method}_{merger_model}_{U}.h5
    TDE_SNRs_{survey}_{mbh}_{nuc}_{occupation_method}_{merger_model}_{tde_model}_{U}.h5
(see report_universes.py's emri_snr_path()/tde_snr_path() for the read
side of this).

VARS naming matches the existing submit-file convention exactly
(MBH_MASS_METHODS, NUCLEATION_METHODS, OCCUPATION_METHODS, TDE_MODEL --
all passed straight through as $(OCCUPATION_METHODS) etc. in
arguments= in the .sub files), e.g.:

    executable = ./EXECUTABLES/gen_events.sh
    arguments  = $(UNIVERSE_MODEL) $(SEED) $(MBH_MASS_METHODS) $(NUCLEATION_METHODS) $(OCCUPATION_METHODS) $(MERGER_MODELS)

Run:
    python make_data_gen_dag.py > run_universes.dag
    python make_data_gen_dag.py --merger_models mundy17 duan24 > run_universes.dag      # subset
    python make_data_gen_dag.py --merger_models mundy17 oleary21 duan24 baseline > run_universes.dag
"""

import argparse
import os, sys
sys.path.insert(0, os.path.abspath('./'))
from utils import MERGER_MODEL_CHOICES, DEFAULT_MERGER_MODELS

SCENARIOS = [
    ("use_lgMgal", "hannah"),
    ("use_sigma",  "hannah"),
    ("use_lgMgal", "neumayer"),
    ("use_sigma",  "neumayer"),
]

SURVEYS = ["ZTF", "LSST"]

# Must match data_generation/gen_TDE_params.py's MODELS list exactly.
TDE_MODELS = [
    # "cooling_envelope",
    # "stream_stream_tde",
    # "tde_analytical",
    "tde_fallback",
]

# Galaxy merger-rate models swept (crossed with everything else). Set on the command line with
# --merger_models; default = the three literature models (baseline is opt-in).
MERGER_MODELS = list(DEFAULT_MERGER_MODELS)

# Occupation methods to sweep, crossed with every (mbh, nuc) scenario
# above. Matches Galaxy.check_occupation's occupation_method argument
# in galaxy.py exactly -- "uniform" (occupation_value defaults to 1.0
# inside the pipeline, reproducing pre-occupation-fraction behavior)
# or "varying" (Zou et al. 2025 mass-dependent occupation fraction,
# arXiv:2510.05252).
OCCUPATION_METHODS = ["uniform", "varying"]

# Number of independent universe realizations per scenario. 50 or 100
# both work fine with DAGMan -- just note the total job count printed
# below before submitting.
N_UNIVERSES = 100


def scenario_tag(mbh, nuc, occ, mm):
    return f"{mbh}_{nuc}_{occ}_{mm}"


def gen_events_name(mbh, nuc, occ, mm, u):
    return f"gen_events_{scenario_tag(mbh, nuc, occ, mm)}_u{u}"


def gen_emri_params_name(mbh, nuc, occ, mm, u):
    return f"gen_EMRI_params_{scenario_tag(mbh, nuc, occ, mm)}_u{u}"


def gen_emri_snr_name(mbh, nuc, occ, mm, u):
    return f"gen_EMRI_SNR_{scenario_tag(mbh, nuc, occ, mm)}_u{u}"


def gen_tde_params_name(mbh, nuc, occ, mm, model, u):
    return f"gen_TDE_params_{scenario_tag(mbh, nuc, occ, mm)}_{model}_u{u}"


def gen_tde_snr_name(survey, mbh, nuc, occ, mm, model, u):
    return f"gen_TDE_SNR_{survey}_{scenario_tag(mbh, nuc, occ, mm)}_{model}_u{u}"


def footprint_name(mbh, nuc, occ, mm, model, u):
    return f"FOOTPRINT_{scenario_tag(mbh, nuc, occ, mm)}_{model}_u{u}"


def validate_occupation_methods():
    if not OCCUPATION_METHODS:
        raise ValueError("OCCUPATION_METHODS must not be empty.")
    seen = set()
    for occ in OCCUPATION_METHODS:
        if occ not in ("uniform", "varying"):
            raise ValueError(f"OCCUPATION_METHODS entries must be 'uniform' or 'varying', got {occ!r}")
        if occ in seen:
            raise ValueError(f"Duplicate entry {occ!r} in OCCUPATION_METHODS.")
        seen.add(occ)


def validate_merger_models():
    if not MERGER_MODELS:
        raise ValueError("MERGER_MODELS must not be empty.")
    if len(set(MERGER_MODELS)) != len(MERGER_MODELS):
        raise ValueError(f"Duplicate entry in MERGER_MODELS: {MERGER_MODELS}")
    for mm in MERGER_MODELS:
        if mm not in MERGER_MODEL_CHOICES:
            raise ValueError(f"Unknown merger model {mm!r}; choose from {MERGER_MODEL_CHOICES}")


def main():
    global MERGER_MODELS, N_UNIVERSES
    parser = argparse.ArgumentParser(description="Generate the multi-universe DAG (scenario x occupation x "
                                     "merger model x universe x TDE model). Prints the DAG to stdout.")
    parser.add_argument("--merger_models", nargs="+", default=MERGER_MODELS, choices=MERGER_MODEL_CHOICES,
                        help=f"Galaxy merger-rate models to sweep (default: {' '.join(DEFAULT_MERGER_MODELS)}; "
                             f"'baseline' = original 0.25(1+z)^2 is opt-in).")
    parser.add_argument("--n_universes", type=int, default=N_UNIVERSES, help="Independent universes per scenario.")
    cli = parser.parse_args()
    MERGER_MODELS = list(cli.merger_models)
    N_UNIVERSES = cli.n_universes

    validate_occupation_methods()
    validate_merger_models()

    lines = []
    terminal_jobs = {mm: [] for mm in MERGER_MODELS}  # every TDE-SNR job, per merger model

    n_scenario_combos = len(SCENARIOS) * len(OCCUPATION_METHODS)
    n_mm = len(MERGER_MODELS)
    n_models = len(TDE_MODELS)
    # JOB lines per universe: gen_events + gen_EMRI_params (prepare_EMRIs.sub; the EMRI SNR step runs inside
    # it, not as its own JOB) = 2, plus per TDE model: gen_TDE_params + gen_TDE_SNR_ZTF + gen_TDE_SNR_LSST = 3.
    # (FOOTPRINT / BUILD_FOOTPRINT_MAPS jobs are commented out below, so they are not counted.)
    n_per_universe = 2 + (1 + len(SURVEYS)) * n_models
    total_jobs = (n_mm * n_scenario_combos * N_UNIVERSES * n_per_universe
                  + n_mm            # REPORT_UNIVERSES_<mm>
                  + 1 + n_mm)       # PULL_REAL_TDE_CATALOG, COMPARE_TDE_DETECTION_<mm>

    lines.append("# ============================================================")
    lines.append(f"# Multi-universe DAG: {n_mm} merger models ({', '.join(MERGER_MODELS)}) "
                 f"x {len(SCENARIOS)} (mbh, nucleation) scenarios "
                 f"x {len(OCCUPATION_METHODS)} occupation methods "
                 f"x {N_UNIVERSES} universes each, x {n_models} TDE models "
                 f"(TDE-stage jobs only), run independently.")
    lines.append(f"# Merger models swept: {', '.join(MERGER_MODELS)}")
    lines.append(f"# Occupation methods swept: {', '.join(OCCUPATION_METHODS)}")
    lines.append(f"# TDE models swept (per-JOB, not per-event -- every model sees the SAME "
                 f"source events): {', '.join(TDE_MODELS)}")
    lines.append(f"# ALL {n_mm * n_scenario_combos * N_UNIVERSES} (merger model x scenario x occupation x universe) "
                 f"realizations, each with {n_models} parallel TDE-model branches, run in parallel; "
                 f"each merger model feeds its OWN REPORT_UNIVERSES_<merger_model> job.")
    lines.append(f"# PULL_REAL_TDE_CATALOG and COMPARE_TDE_DETECTION_<merger_model> run AFTER the reports. "
                 f"PREPARE_DATA_FOR_INFERENCE is NOT part of this DAG (merger model is chosen beforehand).")
    lines.append(f"# Total jobs: {total_jobs}")
    lines.append("# ============================================================")
    lines.append("")

    for mm in MERGER_MODELS:
      for occ in OCCUPATION_METHODS:
        for mbh, nuc in SCENARIOS:
            tag = scenario_tag(mbh, nuc, occ, mm)
            lines.append(f"# ---------------- scenario: {tag} ({N_UNIVERSES} universes x {n_models} TDE models) ----------------")

            for u in range(N_UNIVERSES):
                ev = gen_events_name(mbh, nuc, occ, mm, u)
                lines.append(f"JOB {ev} SUBMIT_FILES/gen_events.sub")
                lines.append(f'VARS {ev} MBH_MASS_METHODS="{mbh}" NUCLEATION_METHODS="{nuc}" '
                             f'OCCUPATION_METHODS="{occ}" MERGER_MODELS="{mm}" UNIVERSE_MODEL="{u}" SEED="{42 + u}"')

                emri_p = gen_emri_params_name(mbh, nuc, occ, mm, u)
                lines.append(f"JOB {emri_p} SUBMIT_FILES/prepare_EMRIs.sub")
                lines.append(f'VARS {emri_p} MBH_MASS_METHODS="{mbh}" NUCLEATION_METHODS="{nuc}" '
                             f'OCCUPATION_METHODS="{occ}" MERGER_MODELS="{mm}" UNIVERSE_MODEL="{u}" SEED="{42 + u}"')

                # One gen_TDE_params job PER TDE MODEL, all reading the SAME
                # gen_events output (full source catalog, no splitting).
                tde_p_names = []
                for model in TDE_MODELS:
                    tde_p = gen_tde_params_name(mbh, nuc, occ, mm, model, u)
                    tde_p_names.append(tde_p)
                    lines.append(f"JOB {tde_p} SUBMIT_FILES/gen_TDE_params.sub")
                    lines.append(f'VARS {tde_p} MBH_MASS_METHODS="{mbh}" NUCLEATION_METHODS="{nuc}" '
                                 f'OCCUPATION_METHODS="{occ}" MERGER_MODELS="{mm}" UNIVERSE_MODEL="{u}" '
                                 f'SEED="{42 + u}" TDE_MODEL="{model}"')

                lines.append(f"PARENT {ev} CHILD {emri_p} {' '.join(tde_p_names)}")

                # EMRI SNR jobs are already wired to their params job in prepare_EMRIs.sub, so no need to add them here.

                for model in TDE_MODELS:
                    tde_p = gen_tde_params_name(mbh, nuc, occ, mm, model, u)

                    tde_snr_names = []
                    for survey in SURVEYS:
                        snr = gen_tde_snr_name(survey, mbh, nuc, occ, mm, model, u)
                        tde_snr_names.append(snr)
                        lines.append(f"JOB {snr} SUBMIT_FILES/gen_TDE_SNRs.sub")
                        lines.append(f'VARS {snr} SURVEY="{survey}" MBH_MASS_METHODS="{mbh}" '
                                     f'NUCLEATION_METHODS="{nuc}" OCCUPATION_METHODS="{occ}" '
                                     f'MERGER_MODELS="{mm}" UNIVERSE_MODEL="{u}" TDE_MODEL="{model}"')
                    lines.append(f"PARENT {tde_p} CHILD {' '.join(tde_snr_names)}")

                    terminal_jobs[mm].extend(tde_snr_names)

                    # fp = footprint_name(mbh, nuc, occ, mm, model, u)
                    # lines.append(f"JOB {fp} SUBMIT_FILES/footprint.sub")
                    # lines.append(f'VARS {fp} MBH_MASS_METHODS="{mbh}" NUCLEATION_METHODS="{nuc}" '
                    #              f'OCCUPATION_METHODS="{occ}" MERGER_MODELS="{mm}" UNIVERSE_MODEL="{u}" TDE_MODEL="{model}"')
                    # lines.append(f"PARENT {tde_p} BUILD_FOOTPRINT_MAPS CHILD {fp}")

                lines.append("")

    # ---------------- ONE combined report PER merger model, over every scenario x occupation x TDE model x universe ----------------
    lines.append("# ---------------- per-merger-model combined report (scenarios x occupation methods x TDE models x universes) ----------------")
    report_jobs = []
    for mm in MERGER_MODELS:
        rep_job = f"REPORT_UNIVERSES_{mm}"
        report_jobs.append(rep_job)
        lines.append(f"JOB {rep_job} SUBMIT_FILES/report_universes.sub")
        lines.append(f'VARS {rep_job} N_UNIVERSES="{N_UNIVERSES}" MERGER_MODELS="{mm}"')
        lines.append(f"PARENT {' '.join(terminal_jobs[mm])} CHILD {rep_job}")
    lines.append("")

    # ---------------- real-catalog comparison: one pull, one comparison per merger model ----------------
    lines.append("# ---------------- real TDE catalog pull + model-vs-real comparison (after the reports) ----------------")
    lines.append("JOB PULL_REAL_TDE_CATALOG SUBMIT_FILES/pull_real_tde_catalog.sub")
    lines.append(f"PARENT {' '.join(report_jobs)} CHILD PULL_REAL_TDE_CATALOG")
    for mm in MERGER_MODELS:
        cmp_job = f"COMPARE_TDE_DETECTION_{mm}"
        lines.append(f"JOB {cmp_job} SUBMIT_FILES/compare_TDE_detection.sub")
        lines.append(f'VARS {cmp_job} N_UNIVERSES="{N_UNIVERSES}" MERGER_MODELS="{mm}"')
        lines.append(f"PARENT PULL_REAL_TDE_CATALOG CHILD {cmp_job}")

    lines.append("# ============================================================")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
