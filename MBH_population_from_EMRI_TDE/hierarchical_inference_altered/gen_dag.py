#!/usr/bin/env python
"""
Generates the condor DAG for the full run: 2 populations x 8 scenarios x
7 source/survey combos = 112 inference jobs, plus one posterior-plot job
(1D+2D combined) per population x scenario, one per-scenario violin job,
a single grand combined-violin job, and one PPD job per population x
scenario.

Run this whenever SOURCE_SURVEY_COMBOS/SCENARIO_KEY (utils.py) change --
don't hand-edit the .dag file, regenerate it, so the job list,
PARENT/CHILD wiring, and log/error/output paths can never drift out of
sync with each other.

Per-scenario sequence:
    infer  ->  posterior (1D+2D)  ->  violin (per-scenario)
then, once every scenario's violin job has finished:
    violin_all (grand combined across all scenarios)
then, last of all:
    PPD (per population x scenario)

PPD is deliberately wired to run dead last -- it's the most expensive
plotting step (many catalogue draws per posterior sample), so nothing
else waits on it.

LSST-involving inference jobs (TDE-LSST, TDE-ALL, EMRI_TDE-LSST,
EMRI_TDE-ALL) carry a lower DAGMan PRIORITY than the ZTF-only /
EMRI-only jobs, since their data volume makes them the slow stragglers.
This only affects scheduling order when multiple jobs are ready and
competing for the same idle slots -- it does not cap what a running
LSST job can use, so it won't shrink its memory/runtime footprint by
itself.

Usage:
    python gen_dag.py > run_inference.dag
"""

import argparse

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

# Surveys whose data volume makes them slow -- deprioritized relative to
# ZTF/EMRI-only jobs (see module docstring).
LOW_PRIORITY_SURVEYS = {"LSST", "ALL"}
HIGH_PRIORITY = 5
LOW_PRIORITY = -5

# Must match utils.SCENARIO_KEY (mbh_mass_method, nucleation_method, occupation_method) keys.
SCENARIOS = [
    ("use_lgMgal", "hannah", "varying"),
    ("use_sigma",  "hannah", "varying"),
    ("use_lgMgal", "neumayer", "varying"),
    ("use_sigma",  "neumayer", "varying"),
    ("use_lgMgal", "hannah", "uniform"),
    ("use_sigma",  "hannah", "uniform"),
    ("use_lgMgal", "neumayer", "uniform"),
    ("use_sigma",  "neumayer", "uniform"),
]

POPULATIONS = ["A", "B"]


FISHER_SOURCES = ["EMRI", "TDE_ZTF", "TDE_LSST"]


def fisher_job_name(source, mbh, nuc, occ):
    """Condor job name of the Fisher sub-DAG."""
    return f"fisher_{source}_{mbh}_{nuc}_{occ}"


def infer_job_name(source, population, survey, mbh, nuc, occ):
    """Condor job name for an inference run."""
    return f"infer_{source}_{population}_{survey}_{mbh}_{nuc}_{occ}"


def posterior_job_name(population, mbh, nuc, occ):
    """Condor job name for posterior plots."""
    return f"posterior_{population}_{mbh}_{nuc}_{occ}"


def violin_scenario_job_name(mbh, nuc, occ):
    """Condor job name for a per-scenario violin plot."""
    return f"violin_{mbh}_{nuc}_{occ}"


def violin_all_job_name():
    """Condor job name for the all-scenario violin plot."""
    return "violin_all"


def ppd_job_name(population, mbh, nuc, occ):
    """Condor job name for the PPD plot."""
    return f"ppd_{population}_{mbh}_{nuc}_{occ}"


def main():
    """Build and write the DAG."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--galaxies", type=int, default=100000)
    cli = ap.parse_args()
    lines = []

    lines.append("# ============= FISHER POSTERIORS (one sub-DAG per scenario and source) =========")
    lines.append("# fisher_analysis builds the per-event posteriors of the detected events; the sub-DAG is")
    lines.append("# written by make_fisher_dag.py (PRE script) once the detected catalogue exists.")
    fisher_names_by_scenario = {}  # (mbh, nuc, occ) -> [node, ...]
    for mbh, nuc, occ in SCENARIOS:
        scen = f"{mbh}_{nuc}_{occ}"
        fisher_names_by_scenario[(mbh, nuc, occ)] = []
        for src in FISHER_SOURCES:
            name = fisher_job_name(src, mbh, nuc, occ)
            fisher_names_by_scenario[(mbh, nuc, occ)].append(name)
            lines.append(f"SUBDAG EXTERNAL {name} fisher_{src}_{scen}.dag DIR ..")
            lines.append(f"SCRIPT PRE {name} python make_fisher_dag.py --GALAXIES {cli.galaxies} --SOURCE {src} --SCENARIO {scen}")
    lines.append("")
    lines.append("# ================ INFERENCE JOBS ==========================")
    infer_names_by_pop_scenario = {}  # (pop, mbh, nuc, occ) -> [job_name, ...]

    for population in POPULATIONS:
        for mbh, nuc, occ in SCENARIOS:
            names = []
            for source, survey in SOURCE_SURVEY_COMBOS:
                name = infer_job_name(source, population, survey, mbh, nuc, occ)
                names.append(name)
                lines.append(f"JOB {name} ./runs/run_inference.sub")
                lines.append(
                    f'VARS {name} SOURCE="{source}" POPULATION="{population}" '
                    f'SURVEY="{survey}" MBH_MASS_METHODS="{mbh}" NUCLEATION_METHODS="{nuc}" OCCUPATION_METHODS="{occ}"'
                )
                priority = LOW_PRIORITY if survey in LOW_PRIORITY_SURVEYS else HIGH_PRIORITY
                lines.append(f"PRIORITY {name} {priority}")
            infer_names_by_pop_scenario[(population, mbh, nuc, occ)] = names
    lines.append("")

    lines.append("# ----------------- posterior plotting jobs (1D+2D combined) -----------------")
    posterior_names_by_pop_scenario = {}
    for population in POPULATIONS:
        for mbh, nuc, occ in SCENARIOS:
            name = posterior_job_name(population, mbh, nuc, occ)
            posterior_names_by_pop_scenario[(population, mbh, nuc, occ)] = name
            lines.append(f"JOB {name} ./runs/make_posterior_plots.sub")
            lines.append(
                f'VARS {name} POPULATION="{population}" '
                f'MBH_MASS_METHODS="{mbh}" NUCLEATION_METHODS="{nuc}" OCCUPATION_METHODS="{occ}"'
            )
    lines.append("")

    lines.append("# ----------------- per-scenario violin plotting jobs -----------------")
    violin_scenario_names = {}
    for mbh, nuc, occ in SCENARIOS:
        name = violin_scenario_job_name(mbh, nuc, occ)
        violin_scenario_names[(mbh, nuc, occ)] = name
        lines.append(f"JOB {name} ./runs/make_violin_scenario.sub")
        lines.append(
            f'VARS {name} MBH_MASS_METHODS="{mbh}" NUCLEATION_METHODS="{nuc}" OCCUPATION_METHODS="{occ}"'
        )
    lines.append("")

    lines.append("# ----------------- grand combined (all-scenarios) violin job -----------------")
    violin_all_name = violin_all_job_name()
    lines.append(f"JOB {violin_all_name} ./runs/make_violin_all.sub")
    lines.append("")

    lines.append("# ----------------- PPD plotting jobs (run last) -----------------")
    ppd_names_by_pop_scenario = {}
    for population in POPULATIONS:
        for mbh, nuc, occ in SCENARIOS:
            name = ppd_job_name(population, mbh, nuc, occ)
            ppd_names_by_pop_scenario[(population, mbh, nuc, occ)] = name
            lines.append(f"JOB {name} ./runs/make_PPD.sub")
            lines.append(
                f'VARS {name} POPULATION="{population}" '
                f'MBH_MASS_METHODS="{mbh}" NUCLEATION_METHODS="{nuc}" OCCUPATION_METHODS="{occ}"'
            )
    lines.append("")

    lines.append("# ----------------- DAG structure -----------------")

    # Fisher -> inference
    for population in POPULATIONS:
        for mbh, nuc, occ in SCENARIOS:
            fisher_names = fisher_names_by_scenario[(mbh, nuc, occ)]
            infer_names = infer_names_by_pop_scenario[(population, mbh, nuc, occ)]
            lines.append(f"PARENT {' '.join(fisher_names)} CHILD {' '.join(infer_names)}")

    # inference -> posterior (1D+2D), per population x scenario
    for population in POPULATIONS:
        for mbh, nuc, occ in SCENARIOS:
            infer_names = infer_names_by_pop_scenario[(population, mbh, nuc, occ)]
            posterior_name = posterior_names_by_pop_scenario[(population, mbh, nuc, occ)]
            lines.append(f"PARENT {' '.join(infer_names)} CHILD {posterior_name}")

    # posterior (both populations of a scenario) -> per-scenario violin
    for mbh, nuc, occ in SCENARIOS:
        posterior_names = [posterior_names_by_pop_scenario[(pop, mbh, nuc, occ)] for pop in POPULATIONS]
        violin_name = violin_scenario_names[(mbh, nuc, occ)]
        lines.append(f"PARENT {' '.join(posterior_names)} CHILD {violin_name}")

    # every per-scenario violin -> grand combined violin
    all_violin_names = [violin_scenario_names[s] for s in SCENARIOS]
    lines.append(f"PARENT {' '.join(all_violin_names)} CHILD {violin_all_name}")

    # grand combined violin -> PPD (dead last)
    all_ppd_names = [ppd_names_by_pop_scenario[(population, mbh, nuc, occ)]
                      for population in POPULATIONS for mbh, nuc, occ in SCENARIOS]
    lines.append(f"PARENT {violin_all_name} CHILD {' '.join(all_ppd_names)}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
