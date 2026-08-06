#!/usr/bin/env python
"""
Generates the condor DAG for the full run: 2 populations x 4 scenarios x
7 source/survey combos = 56 inference jobs, plus 8 oneD-posterior jobs
and 8 PPD jobs (one of each per population x scenario).

Run this whenever SOURCE_SURVEY_COMBOS (utils.py) or the scenario list
below changes -- don't hand-edit the .dag file, regenerate it, so the
job list, PARENT/CHILD wiring, and log/error/output paths can never
drift out of sync with each other.

Usage:
    python generate_dag.py > run_all.dag
"""

# Must match utils.SOURCE_SURVEY_COMBOS (source, survey) pairs.
SOURCE_SURVEY_COMBOS = [
    ("EMRI",     "None"),
    ("TDE",      "ZTF"),
    ("TDE",      "LSST"),
    ("TDE",      "ALL"),
    ("EMRI_TDE", "ZTF"),
    ("EMRI_TDE", "LSST"),
    ("EMRI_TDE", "ALL"),
]

# Must match utils.SCENARIO_KEY (mbh_mass_method, nucleation_method) pairs.
SCENARIOS = [
    ("use_lgMgal", "hannah"),
    ("use_sigma",  "hannah"),
    ("use_lgMgal", "neumayer"),
    ("use_sigma",  "neumayer"),
]

POPULATIONS = ["A", "B"]


def data_job_name(population, mbh, nuc):
    return f"make_pseudo_data_{population}_{mbh}_{nuc}"


def infer_job_name(source, population, survey, mbh, nuc):
    return f"infer_{source}_{population}_{survey}_{mbh}_{nuc}"


def oneD_job_name(population, mbh, nuc):
    return f"oneD_{population}_{mbh}_{nuc}"


def ppd_job_name(population, mbh, nuc):
    return f"ppd_{population}_{mbh}_{nuc}"


def main():
    lines = []

    lines.append("# ============= MAKE DATA (one per population x scenario) =========")
    lines.append("# make_pseudo_data.py's output filenames are population+scenario")
    lines.append("# specific (true_data_EMRI_pseudo_data_{POP}_{mbh}_{nuc}.h5)")
    data_names_by_pop_scenario = {}  # (pop, mbh, nuc) -> job_name
    for population in POPULATIONS:
        for mbh, nuc in SCENARIOS:
            name = data_job_name(population, mbh, nuc)
            data_names_by_pop_scenario[(population, mbh, nuc)] = name
            lines.append(f"JOB {name} ./pseudo_runs/make_pseudo_data.sub")
            lines.append(
                f'VARS {name} POPULATION="{population}" MBH_MASS_METHODS="{mbh}" NUCLEATION_METHODS="{nuc}" INCLUDE_LSST="1"'
            )
    lines.append("")

    lines.append("# ================ INFERENCE JOBS ==========================")
    infer_names_by_pop_scenario = {}  # (pop, mbh, nuc) -> [job_name, ...]

    for population in POPULATIONS:
        for mbh, nuc in SCENARIOS:
            names = []
            for source, survey in SOURCE_SURVEY_COMBOS:
                name = infer_job_name(source, population, survey, mbh, nuc)
                names.append(name)
                lines.append(f"JOB {name} ./pseudo_runs/run_inf_pseudo.sub")
                lines.append(
                    f'VARS {name} SOURCE="{source}" POPULATION="{population}" '
                    f'SURVEY="{survey}" MBH_MASS_METHODS="{mbh}" NUCLEATION_METHODS="{nuc}"'
                )
            infer_names_by_pop_scenario[(population, mbh, nuc)] = names
    lines.append("")

    lines.append("# ----------------- 1D posterior plotting jobs -----------------")
    for population in POPULATIONS:
        for mbh, nuc in SCENARIOS:
            name = oneD_job_name(population, mbh, nuc)
            lines.append(f"JOB {name} ./pseudo_runs/make_posterior_plots.sub")
            lines.append(
                f'VARS {name} POPULATION="{population}" '
                f'MBH_MASS_METHOD="{mbh}" NUCLEATION_METHOD="{nuc}"'
            )
    lines.append("")

    lines.append("# ----------------- PPD plotting jobs -----------------")
    for population in POPULATIONS:
        for mbh, nuc in SCENARIOS:
            name = ppd_job_name(population, mbh, nuc)
            lines.append(f"JOB {name} ./pseudo_runs/make_PPD.sub")
            lines.append(
                f'VARS {name} POPULATION="{population}" '
                f'MBH_MASS_METHOD="{mbh}" NUCLEATION_METHOD="{nuc}"'
            )
    lines.append("")

    lines.append("# ----------------- DAG structure -----------------")
    for population in POPULATIONS:
        for mbh, nuc in SCENARIOS:
            data_name = data_names_by_pop_scenario[(population, mbh, nuc)]
            infer_names = infer_names_by_pop_scenario[(population, mbh, nuc)]
            lines.append(f"PARENT {data_name} CHILD {' '.join(infer_names)}")

    for population in POPULATIONS:
        for mbh, nuc in SCENARIOS:
            names = infer_names_by_pop_scenario[(population, mbh, nuc)]
            oneD = oneD_job_name(population, mbh, nuc)
            ppd = ppd_job_name(population, mbh, nuc)
            lines.append(f"PARENT {' '.join(names)} CHILD {oneD} {ppd}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()