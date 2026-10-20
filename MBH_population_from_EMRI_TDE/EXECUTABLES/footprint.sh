#!/bin/sh

# Plots the ZTF/LSST footprint + completeness gradient against the REAL
# simulated TDE population for ONE (model, universe) combo. Requires
# build_footprint_maps.sh to have already produced coverage_maps_nohealpy.npz
# (a DAG parent job -- see make_data_gen_dag.py), and gen_TDE_params.py to
# have already produced this combo's TDE catalog.

. EXECUTABLES/config.sh

UNIVERSE_MODEL=$1
MBH_MASS_METHODS=$2 # "use_sigma" "use_lgMgal"
NUCLEATION_METHODS=$3 # "hannah" "neumayer"
OCCUPATION_METHODS=$4 # "uniform" "varying"
TDE_MODEL=$5 # "cooling_envelope" "stream_stream_tde" "tde_analytical" "tde_fallback"
MERGER_MODELS=$6 # "mundy17" "oleary21" "duan24" "baseline"

echo "==========================================================================================="
echo "Footprint: GALAXIES=$GALAXIES universe=$UNIVERSE_MODEL mbh=$MBH_MASS_METHODS nucleation=$NUCLEATION_METHODS occupation=$OCCUPATION_METHODS merger=$MERGER_MODELS TDE_model=$TDE_MODEL"
echo "==========================================================================================="

python ./data_generation/footprint_LSST_ZTF.py \
    --UNIVERSE_MODEL "$UNIVERSE_MODEL" \
    --GALAXIES "$GALAXIES" \
    --TDE_model "$TDE_MODEL" \
    --mbh_mass_method "$MBH_MASS_METHODS" \
    --nucleation_method "$NUCLEATION_METHODS" \
    --occupation_method "$OCCUPATION_METHODS" \
    --merger_model "$MERGER_MODELS"
