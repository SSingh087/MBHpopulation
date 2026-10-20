#!/bin/sh

. ./EXECUTABLES/config.sh

# One (TDE model, mbh, nuc, occ) scenario per invocation -- the DAG
# (make_data_gen_dag.py's PREPARE_TDE_MODELS loop) creates one job per
# combination, so no internal looping over scenarios happens here.
#
# BEST_UNIVERSE may be an explicit integer, or "auto" (the DAG's default)
# to have prepare_data_MEDIAN.py resolve the winning universe itself at
# run time from PLOTS/universe_report/best_universe_<TDE_MODEL>.txt
# (written by REPORT_UNIVERSES, which this job runs after).
TDE_MODEL=$1
BEST_UNIVERSE=$2
MBH_MASS_METHOD=$3
NUCLEATION_METHOD=$4
OCCUPATION_METHOD=$5
MERGER_MODEL=$6   # optional; empty = legacy untagged file names
MM_ARG=""; [ -n "$MERGER_MODEL" ] && MM_ARG="--merger_model $MERGER_MODEL"

echo "Preparing data for inference for $GALAXIES galaxies:"
echo "  TDE model: $TDE_MODEL"
echo "  best universe: $BEST_UNIVERSE"
echo "  MBH mass method: $MBH_MASS_METHOD"
echo "  nucleation method: $NUCLEATION_METHOD"
echo "  occupation method: $OCCUPATION_METHOD"

python ./hierarchical_inference/prepare_data_MEDIAN.py \
    --GALAXIES "$GALAXIES" \
    --mbh_mass_method "$MBH_MASS_METHOD" \
    --nucleation_method "$NUCLEATION_METHOD" \
    --occupation_method "$OCCUPATION_METHOD" \
    --tde_model "$TDE_MODEL" \
    --best_universe_model "$BEST_UNIVERSE" $MM_ARG