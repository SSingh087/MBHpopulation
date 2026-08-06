#!/usr/bin/env python
"""
Generates the multi-universe DAG: for each of the 4
(mbh_mass_method, nucleation_method) scenarios, run N_UNIVERSES fully
independent realizations (same scenario, different --UNIVERSE_MODEL)
IN PARALLEL, each carried through EMRI/TDE SNR/detection:

    gen_events(universe=U)
        -> gen_EMRI_params(universe=U) -> gen_EMRI_SNR(universe=U)
        -> gen_TDE_params(universe=U)  -> gen_TDE_SNR_ZTF(universe=U)
                                        -> gen_TDE_SNR_LSST(universe=U)

REPORT_UNIVERSES runs last, after every universe's SNR jobs finish. It
scans all of them, and copies the winning universe's files with a
"max_detected_" prefix -- no second DAG layer, no dynamic job
generation. Nothing downstream needs wiring beyond REPORT_UNIVERSES.

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

# Number of independent universe realizations per scenario. 50 or 100
# both work fine with DAGMan -- just note the total job count printed
# below before submitting (4 scenarios x N_UNIVERSES x 6 jobs + 1).
N_UNIVERSES = 100


def scenario_tag(mbh, nuc):
    return f"{mbh}_{nuc}"


def gen_events_name(mbh, nuc, u):
    return f"gen_events_{scenario_tag(mbh, nuc)}_u{u}"


def gen_emri_params_name(mbh, nuc, u):
    return f"gen_EMRI_params_{scenario_tag(mbh, nuc)}_u{u}"


def gen_emri_snr_name(mbh, nuc, u):
    return f"gen_EMRI_SNR_{scenario_tag(mbh, nuc)}_u{u}"


def gen_tde_params_name(mbh, nuc, u):
    return f"gen_TDE_params_{scenario_tag(mbh, nuc)}_u{u}"


def gen_tde_snr_name(survey, mbh, nuc, u):
    return f"gen_TDE_SNR_{survey}_{scenario_tag(mbh, nuc)}_u{u}"


def main():
    lines = []
    all_terminal_jobs = []

    lines.append("# ============================================================")
    lines.append(f"# Multi-universe DAG: {len(SCENARIOS)} scenarios x {N_UNIVERSES} "
                 f"universes each, run independently.")
    n_per_universe = 6  # events, emri_params, emri_snr, tde_params, tde_snr_ztf, tde_snr_lsst
    total_jobs = len(SCENARIOS) * N_UNIVERSES * n_per_universe + 1
    lines.append(f"# Total jobs: {len(SCENARIOS)} x {N_UNIVERSES} x {n_per_universe} + 1 (report) "
                 f"= {total_jobs}")
    lines.append("# ============================================================")
    lines.append("")

    for mbh, nuc in SCENARIOS:
        tag = scenario_tag(mbh, nuc)
        lines.append(f"# ---------------- scenario: {tag} ({N_UNIVERSES} universes) ----------------")

        for u in range(N_UNIVERSES):
            ev = gen_events_name(mbh, nuc, u)
            lines.append(f"JOB {ev} SUBMIT_FILES/gen_events.sub")
            lines.append(f'VARS {ev} MBH_MASS_METHODS="{mbh}" NUCLEATION_METHODS="{nuc}" UNIVERSE_MODEL="{u}" SEED="{42 + u}"')

            emri_p = gen_emri_params_name(mbh, nuc, u)
            lines.append(f"JOB {emri_p} SUBMIT_FILES/prepare_EMRIs.sub")
            lines.append(f'VARS {emri_p} MBH_MASS_METHODS="{mbh}" NUCLEATION_METHODS="{nuc}" UNIVERSE_MODEL="{u}" SEED="{42 + u}"')

            tde_p = gen_tde_params_name(mbh, nuc, u)
            lines.append(f"JOB {tde_p} SUBMIT_FILES/gen_TDE_params.sub")
            lines.append(f'VARS {tde_p} MBH_MASS_METHODS="{mbh}" NUCLEATION_METHODS="{nuc}" UNIVERSE_MODEL="{u}" SEED="{42 + u}"')

            lines.append(f"PARENT {ev} CHILD {emri_p} {tde_p}")

            # EMRI SNR jobs are already wired to their params job in prepare_EMRIs.sub, so no need to add them here.

            tde_snr_names = []
            for survey in SURVEYS:
                snr = gen_tde_snr_name(survey, mbh, nuc, u)
                tde_snr_names.append(snr)
                lines.append(f"JOB {snr} SUBMIT_FILES/gen_TDE_SNRs.sub")
                lines.append(f'VARS {snr} SURVEY="{survey}" MBH_MASS_METHODS="{mbh}" '
                             f'NUCLEATION_METHODS="{nuc}" UNIVERSE_MODEL="{u}"')
            lines.append(f"PARENT {tde_p} CHILD {' '.join(tde_snr_names)}")

            all_terminal_jobs.extend(tde_snr_names)
            lines.append("")

    lines.append("# ---------------- final report over all universes ----------------")
    lines.append("JOB REPORT_UNIVERSES SUBMIT_FILES/report_universes.sub")
    lines.append(f'VARS REPORT_UNIVERSES N_UNIVERSES="{N_UNIVERSES}"')
    lines.append(f"PARENT {' '.join(all_terminal_jobs)} CHILD REPORT_UNIVERSES")

    lines.append("# ============================================================")

    for mbh, nuc in SCENARIOS:
        tag = scenario_tag(mbh, nuc)
        lines.append("JOB PREPARE_DATA_FOR_INFERENCE SUBMIT_FILES/prepare_data_for_inference.sub")
        lines.append(f'VARS PREPARE_DATA_FOR_INFERENCE MBH_MASS_METHODS="{mbh}" NUCLEATION_METHODS="{nuc}"')

    lines.append("# ============================================================")
    print("\n".join(lines))


if __name__ == "__main__":
    main()