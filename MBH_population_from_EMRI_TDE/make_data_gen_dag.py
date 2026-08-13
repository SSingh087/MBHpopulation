#!/usr/bin/env python
"""
Generates the multi-universe DAG: for each of the (mbh_mass_method,
nucleation_method) scenarios CROSSED WITH each occupation_method in
OCCUPATION_METHODS, run N_UNIVERSES fully independent realizations
(same scenario, different --UNIVERSE_MODEL) IN PARALLEL, each carried
through EMRI/TDE SNR/detection:

    gen_events(universe=U)
        -> gen_EMRI_params(universe=U) -> gen_EMRI_SNR(universe=U)
        -> gen_TDE_params(universe=U)  -> gen_TDE_SNR_ZTF(universe=U)
                                        -> gen_TDE_SNR_LSST(universe=U)

DAG STRUCTURE (this is the part that was wrong before):

  ALL gen_events/EMRI/TDE/SNR jobs, across EVERY (mbh, nuc, occ, U)
  combination, run in parallel and are ALL parents of ONE SINGLE
  REPORT_UNIVERSES job.

  REPORT_UNIVERSES is combined, not per-occupation-method: picking the
  winning UNIVERSE_MODEL only makes sense as a single maximin scan
  across every (scenario x occupation x channel) cell together. Two
  separate report runs (one per occupation method) could each pick a
  DIFFERENT winning universe, which defeats the entire point of "same
  universe used for every scenario" -- that guarantee has to span
  occupation method too, not just (mbh, nuc).

  PREPARE_DATA_FOR_INFERENCE jobs (one per (mbh, nuc, occ) scenario)
  run ONLY AFTER REPORT_UNIVERSES completes -- they are its CHILDREN,
  not parallel siblings -- because they consume the winning universe's
  max_detected_*.h5 files, which REPORT_UNIVERSES is what produces.

Every job's VARS and every output filename carries an
OCCUPATION_METHODS suffix after the existing mbh/nuc suffix, so that
runs with different occupation methods don't collide on disk. Filename
pattern:
    EMRI_SNRs_data_{mbh}_{nuc}_{occupation_method}_{U}.h5
(see report_universes.py's emri_snr_path()/tde_snr_path() for the read
side of this).

VARS naming matches the existing submit-file convention exactly
(MBH_MASS_METHODS, NUCLEATION_METHODS, OCCUPATION_METHODS -- all
plural, all passed straight through as $(OCCUPATION_METHODS) etc. in
arguments= in the .sub files), e.g.:

    executable = ./EXECUTABLES/gen_events.sh
    arguments  = $(UNIVERSE_MODEL) $(SEED) $(MBH_MASS_METHODS) $(NUCLEATION_METHODS) $(OCCUPATION_METHODS)

Run:
    python make_data_gen_dag.py > run_universes.dag
"""

SCENARIOS = [
    ("use_lgMgal", "hannah"),
    ("use_sigma",  "hannah"),
    ("use_lgMgal", "neumayer"),
    ("use_sigma",  "neumayer"),
]

SURVEYS = ["ZTF", "LSST"]

# Occupation methods to sweep, crossed with every (mbh, nuc) scenario
# above. Matches Galaxy.check_occupation's occupation_method argument
# in galaxy.py exactly -- "uniform" (occupation_value defaults to 1.0
# inside the pipeline, reproducing pre-occupation-fraction behavior)
# or "varying" (Zou et al. 2025 mass-dependent occupation fraction,
# arXiv:2510.05252).
OCCUPATION_METHODS = ["uniform", "varying"]

# Number of independent universe realizations per scenario. 50 or 100
# both work fine with DAGMan -- just note the total job count printed
# below before submitting
# (len(SCENARIOS) x len(OCCUPATION_METHODS) x N_UNIVERSES x 6 jobs + 1 report
#  + len(SCENARIOS) x len(OCCUPATION_METHODS) prepare-for-inference jobs).
N_UNIVERSES = 100


def scenario_tag(mbh, nuc, occ):
    return f"{mbh}_{nuc}_{occ}"


def gen_events_name(mbh, nuc, occ, u):
    return f"gen_events_{scenario_tag(mbh, nuc, occ)}_u{u}"


def gen_emri_params_name(mbh, nuc, occ, u):
    return f"gen_EMRI_params_{scenario_tag(mbh, nuc, occ)}_u{u}"


def gen_emri_snr_name(mbh, nuc, occ, u):
    return f"gen_EMRI_SNR_{scenario_tag(mbh, nuc, occ)}_u{u}"


def gen_tde_params_name(mbh, nuc, occ, u):
    return f"gen_TDE_params_{scenario_tag(mbh, nuc, occ)}_u{u}"


def gen_tde_snr_name(survey, mbh, nuc, occ, u):
    return f"gen_TDE_SNR_{survey}_{scenario_tag(mbh, nuc, occ)}_u{u}"


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


def main():
    validate_occupation_methods()

    lines = []
    all_terminal_jobs = []  # every SNR job, across ALL scenarios x occupation methods x universes

    n_scenario_combos = len(SCENARIOS) * len(OCCUPATION_METHODS)
    n_per_universe = 6  # events, emri_params, emri_snr, tde_params, tde_snr_ztf, tde_snr_lsst
    n_prepare_jobs = n_scenario_combos
    total_jobs = n_scenario_combos * N_UNIVERSES * n_per_universe + 1 + n_prepare_jobs

    lines.append("# ============================================================")
    lines.append(f"# Multi-universe DAG: {len(SCENARIOS)} (mbh, nucleation) scenarios "
                 f"x {len(OCCUPATION_METHODS)} occupation methods "
                 f"x {N_UNIVERSES} universes each, run independently.")
    lines.append(f"# Occupation methods swept: {', '.join(OCCUPATION_METHODS)}")
    lines.append(f"# ALL {n_scenario_combos * N_UNIVERSES} (scenario x occupation x universe) "
                 f"realizations run in parallel and feed into ONE combined REPORT_UNIVERSES job.")
    lines.append(f"# PREPARE_DATA_FOR_INFERENCE jobs run AFTER REPORT_UNIVERSES (they consume "
                 f"its max_detected_*.h5 output), not in parallel with it.")
    lines.append(f"# Total jobs: {n_scenario_combos} x {N_UNIVERSES} x {n_per_universe} "
                 f"+ 1 (single combined report) + {n_prepare_jobs} (prepare-for-inference) "
                 f"= {total_jobs}")
    lines.append("# ============================================================")
    lines.append("")

    for occ in OCCUPATION_METHODS:
        for mbh, nuc in SCENARIOS:
            tag = scenario_tag(mbh, nuc, occ)
            lines.append(f"# ---------------- scenario: {tag} ({N_UNIVERSES} universes) ----------------")

            for u in range(N_UNIVERSES):
                ev = gen_events_name(mbh, nuc, occ, u)
                lines.append(f"JOB {ev} SUBMIT_FILES/gen_events.sub")
                lines.append(f'VARS {ev} MBH_MASS_METHODS="{mbh}" NUCLEATION_METHODS="{nuc}" '
                             f'OCCUPATION_METHODS="{occ}" UNIVERSE_MODEL="{u}" SEED="{42 + u}"')

                emri_p = gen_emri_params_name(mbh, nuc, occ, u)
                lines.append(f"JOB {emri_p} SUBMIT_FILES/prepare_EMRIs.sub")
                lines.append(f'VARS {emri_p} MBH_MASS_METHODS="{mbh}" NUCLEATION_METHODS="{nuc}" '
                             f'OCCUPATION_METHODS="{occ}" UNIVERSE_MODEL="{u}" SEED="{42 + u}"')

                tde_p = gen_tde_params_name(mbh, nuc, occ, u)
                lines.append(f"JOB {tde_p} SUBMIT_FILES/gen_TDE_params.sub")
                lines.append(f'VARS {tde_p} MBH_MASS_METHODS="{mbh}" NUCLEATION_METHODS="{nuc}" '
                             f'OCCUPATION_METHODS="{occ}" UNIVERSE_MODEL="{u}" SEED="{42 + u}"')

                lines.append(f"PARENT {ev} CHILD {emri_p} {tde_p}")

                # EMRI SNR jobs are already wired to their params job in prepare_EMRIs.sub, so no need to add them here.

                tde_snr_names = []
                for survey in SURVEYS:
                    snr = gen_tde_snr_name(survey, mbh, nuc, occ, u)
                    tde_snr_names.append(snr)
                    lines.append(f"JOB {snr} SUBMIT_FILES/gen_TDE_SNRs.sub")
                    lines.append(f'VARS {snr} SURVEY="{survey}" MBH_MASS_METHODS="{mbh}" '
                                 f'NUCLEATION_METHODS="{nuc}" OCCUPATION_METHODS="{occ}" UNIVERSE_MODEL="{u}"')
                lines.append(f"PARENT {tde_p} CHILD {' '.join(tde_snr_names)}")

                all_terminal_jobs.extend(tde_snr_names)
                lines.append("")

    # ---------------- ONE combined report, over EVERY scenario x occupation x universe ----------------
    lines.append("# ---------------- final combined report over ALL scenarios x occupation methods x universes ----------------")
    lines.append("JOB REPORT_UNIVERSES SUBMIT_FILES/report_universes.sub")
    lines.append(f'VARS REPORT_UNIVERSES N_UNIVERSES="{N_UNIVERSES}"')
    lines.append(f"PARENT {' '.join(all_terminal_jobs)} CHILD REPORT_UNIVERSES")
    lines.append("")

    # ---------------- PREPARE_DATA_FOR_INFERENCE runs AFTER the report, as its children ----------------
    lines.append("# ---------------- prepare data for inference (runs only after REPORT_UNIVERSES) ----------------")
    prepare_job_names = []
    for occ in OCCUPATION_METHODS:
        for mbh, nuc in SCENARIOS:
            tag = scenario_tag(mbh, nuc, occ)
            job = f"PREPARE_DATA_FOR_INFERENCE_{tag}"
            prepare_job_names.append(job)
            lines.append(f"JOB {job} SUBMIT_FILES/prepare_data_for_inference.sub")
            lines.append(f'VARS {job} MBH_MASS_METHODS="{mbh}" NUCLEATION_METHODS="{nuc}" '
                         f'OCCUPATION_METHODS="{occ}"')
    lines.append(f"PARENT REPORT_UNIVERSES CHILD {' '.join(prepare_job_names)}")

    lines.append("# ============================================================")
    print("\n".join(lines))


if __name__ == "__main__":
    main()