#!/usr/bin/env python
"""
Generates the condor DAG for the full run: 2 populations x 8 scenarios x
7 source/survey combos = 112 inference jobs, plus one posterior-plot job
(1D+2D combined) per population x scenario, one violin job per
population (overlaying all 8 scenarios), and one PPD job per population
x scenario.

Run this whenever SOURCE_SURVEY_COMBOS/SCENARIO_KEY/MODEL_LABELS
(utils.py) change -- don't hand-edit the .dag file, regenerate it, so
the job list, PARENT/CHILD wiring, and log/error/output paths can never
drift out of sync with each other.

Structure (deliberately flat, not per-scenario-chained -- posterior
plots are scenario-specific by nature and only wait on their own
scenario's inference; violin and PPD wait on ALL inference to finish):

    data(pop, scenario) -> infer(pop, scenario, case)   [112 jobs]
    infer(pop, scenario, *)  -> posterior(pop, scenario) [1D+2D, 16 jobs]
    ALL infer(pop, *, *)     -> violin(pop)              [2 jobs, one per population --
                                                            population A and B are never
                                                            mixed in one violin figure,
                                                            see plot_violin.py]
    ALL posterior + BOTH violin -> PPD(pop, scenario)    [16 jobs, run dead last]

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

Optional PREPARE stage (--with_prepare): prep_pseudo_data.py reads true_detected_data_*.h5, which
prepare_data_MEDIAN.py writes (once per mbh/nuc/occ scenario, shared by populations A and B). With the
flag, 8 prepare jobs are added as PARENTs of that scenario's two make-data jobs. They run from the repo
root (DIR ..) with SUBMIT_FILES/prepare_data_for_inference.sub. Skip the flag if the files already exist.

Usage:
    python gen_dag.py > run_inference.dag
    python gen_dag.py --with_prepare --tde_model tde_fallback --merger_model mundy17 > run_inference.dag
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


def data_job_name(population, mbh, nuc, occ):
    """Condor job name for data preparation."""
    return f"make_pseudo_data_{population}_{mbh}_{nuc}_{occ}"


def infer_job_name(source, population, survey, mbh, nuc, occ):
    """Condor job name for an inference run."""
    return f"infer_{source}_{population}_{survey}_{mbh}_{nuc}_{occ}"


def posterior_job_name(population, mbh, nuc, occ):
    """Condor job name for posterior plots."""
    return f"posterior_{population}_{mbh}_{nuc}_{occ}"


def violin_job_name(population):
    """Condor job name for the violin plot."""
    return f"violin_{population}"


def ppd_job_name(population, mbh, nuc, occ):
    """Condor job name for the PPD plot."""
    return f"ppd_{population}_{mbh}_{nuc}_{occ}"


def prepare_job_name(mbh, nuc, occ):
    """Condor job name for inference-data preparation."""
    return f"prepare_data_{mbh}_{nuc}_{occ}"


def main():
    """Build and write the DAG."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--with_prepare", action="store_true", help="add the one-off prepare_data_MEDIAN.py stage")
    ap.add_argument("--tde_model", default="tde_fallback")
    ap.add_argument("--merger_model", default="", help="empty = legacy untagged file names")
    ap.add_argument("--best_universe", default="auto")
    cli = ap.parse_args()
    lines = []

    if cli.with_prepare:
        lines.append("# ============= PREPARE DATA (prepare_data_MEDIAN.py, once per scenario) =========")
        for mbh, nuc, occ in SCENARIOS:
            name = prepare_job_name(mbh, nuc, occ)
            lines.append(f"JOB {name} ../SUBMIT_FILES/prepare_data_for_inference.sub DIR ..")
            lines.append(f'VARS {name} TDE_MODEL="{cli.tde_model}" BEST_UNIVERSE="{cli.best_universe}" '
                         f'MBH_MASS_METHODS="{mbh}" NUCLEATION_METHODS="{nuc}" OCCUPATION_METHODS="{occ}" '
                         f'MERGER_MODELS="{cli.merger_model}"')
        lines.append("")

    lines.append("# ============= MAKE DATA (one per population x scenario) =========")
    lines.append("# make_pseudo_data.py's output filenames are population+scenario")
    lines.append("# specific (true_data_EMRI_pseudo_data_{POP}_{mbh}_{nuc}_{occ}.h5)")
    data_names_by_pop_scenario = {}  # (pop, mbh, nuc, occ) -> job_name
    for population in POPULATIONS:
        for mbh, nuc, occ in SCENARIOS:
            name = data_job_name(population, mbh, nuc, occ)
            data_names_by_pop_scenario[(population, mbh, nuc, occ)] = name
            lines.append(f"JOB {name} ./pseudo_runs/make_pseudo_data.sub")
            lines.append(
                f'VARS {name} POPULATION="{population}" MBH_MASS_METHODS="{mbh}" NUCLEATION_METHODS="{nuc}" INCLUDE_LSST="1" OCCUPATION_METHODS="{occ}"'
            )
    lines.append("")

    lines.append("# ================ INFERENCE JOBS ==========================")
    infer_names_by_pop_scenario = {}  # (pop, mbh, nuc, occ) -> [job_name, ...]
    infer_names_by_pop = {pop: [] for pop in POPULATIONS}  # pop -> [job_name, ...] (every scenario/case)

    for population in POPULATIONS:
        for mbh, nuc, occ in SCENARIOS:
            names = []
            for source, survey in SOURCE_SURVEY_COMBOS:
                name = infer_job_name(source, population, survey, mbh, nuc, occ)
                names.append(name)
                lines.append(f"JOB {name} ./pseudo_runs/run_inf_pseudo.sub")
                lines.append(
                    f'VARS {name} SOURCE="{source}" POPULATION="{population}" '
                    f'SURVEY="{survey}" MBH_MASS_METHODS="{mbh}" NUCLEATION_METHODS="{nuc}" OCCUPATION_METHODS="{occ}"'
                )
                priority = LOW_PRIORITY if survey in LOW_PRIORITY_SURVEYS else HIGH_PRIORITY
                lines.append(f"PRIORITY {name} {priority}")
            infer_names_by_pop_scenario[(population, mbh, nuc, occ)] = names
            infer_names_by_pop[population].extend(names)
    lines.append("")

    lines.append("# ----------------- posterior plotting jobs (1D+2D combined) -----------------")
    posterior_names_by_pop_scenario = {}
    all_posterior_names = []
    for population in POPULATIONS:
        for mbh, nuc, occ in SCENARIOS:
            name = posterior_job_name(population, mbh, nuc, occ)
            posterior_names_by_pop_scenario[(population, mbh, nuc, occ)] = name
            all_posterior_names.append(name)
            lines.append(f"JOB {name} ./pseudo_runs/make_posterior_plots.sub")
            lines.append(
                f'VARS {name} POPULATION="{population}" '
                f'MBH_MASS_METHODS="{mbh}" NUCLEATION_METHODS="{nuc}" OCCUPATION_METHODS="{occ}"'
            )
    lines.append("")

    lines.append("# ----------------- violin plotting jobs (one per population, all 8 scenarios overlaid) -----------------")
    violin_names_by_pop = {}
    for population in POPULATIONS:
        name = violin_job_name(population)
        violin_names_by_pop[population] = name
        lines.append(f"JOB {name} ./pseudo_runs/make_violin.sub")
        lines.append(f'VARS {name} POPULATION="{population}"')
    lines.append("")

    lines.append("# ----------------- PPD plotting jobs (run last) -----------------")
    all_ppd_names = []
    for population in POPULATIONS:
        for mbh, nuc, occ in SCENARIOS:
            name = ppd_job_name(population, mbh, nuc, occ)
            all_ppd_names.append(name)
            lines.append(f"JOB {name} ./pseudo_runs/make_PPD.sub")
            lines.append(
                f'VARS {name} POPULATION="{population}" '
                f'MBH_MASS_METHODS="{mbh}" NUCLEATION_METHODS="{nuc}" OCCUPATION_METHODS="{occ}"'
            )
    lines.append("")

    lines.append("# ----------------- DAG structure -----------------")

    # prepare -> data (both populations of a scenario)
    if cli.with_prepare:
        for mbh, nuc, occ in SCENARIOS:
            kids = " ".join(data_names_by_pop_scenario[(pop, mbh, nuc, occ)] for pop in POPULATIONS)
            lines.append(f"PARENT {prepare_job_name(mbh, nuc, occ)} CHILD {kids}")

    # data -> inference
    for population in POPULATIONS:
        for mbh, nuc, occ in SCENARIOS:
            data_name = data_names_by_pop_scenario[(population, mbh, nuc, occ)]
            infer_names = infer_names_by_pop_scenario[(population, mbh, nuc, occ)]
            lines.append(f"PARENT {data_name} CHILD {' '.join(infer_names)}")

    # inference -> posterior (1D+2D), per population x scenario (only waits
    # on its OWN scenario's inference jobs, not every scenario)
    for population in POPULATIONS:
        for mbh, nuc, occ in SCENARIOS:
            infer_names = infer_names_by_pop_scenario[(population, mbh, nuc, occ)]
            posterior_name = posterior_names_by_pop_scenario[(population, mbh, nuc, occ)]
            lines.append(f"PARENT {' '.join(infer_names)} CHILD {posterior_name}")

    # ALL inference for a population (every scenario x case) -> that
    # population's violin job (it overlays all 8 scenarios itself)
    for population in POPULATIONS:
        lines.append(f"PARENT {' '.join(infer_names_by_pop[population])} CHILD {violin_names_by_pop[population]}")

    # every posterior job + both violin jobs -> PPD (dead last)
    gate_names = all_posterior_names + [violin_names_by_pop[pop] for pop in POPULATIONS]
    lines.append(f"PARENT {' '.join(gate_names)} CHILD {' '.join(all_ppd_names)}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
