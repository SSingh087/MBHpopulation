#!/usr/bin/env bash


# export GALAXIES=100000
# export UNIVERSE_MODEL=0
# export Z_MAX=8.0
# export SEED=42
# export IS_CLUSTERING=0

# POPULATION=$1

# echo "=== Generating posterior plots for ${POPULATION} ==="

# python posterior_1D.py --GALAXIES "$GALAXIES" --POPULATION "$POPULATION"
# # python hierarchical_inference/INF_PLOT_SCRIPTS/posterior_2D.py     --GALAXIES "$GALAXIES" --POPULATION "$POPULATION"



export GALAXIES=100000
export UNIVERSE_MODEL=0
export Z_MAX=8.0
export SEED=42
export IS_CLUSTERING=0

POPULATION=$1
MBH_MASS_METHOD=$2
NUCLEATION_METHOD=$3

echo "=== Generating posterior plots for ${POPULATION} (${MBH_MASS_METHOD}/${NUCLEATION_METHOD}) ==="

python posterior_1D.py --GALAXIES "$GALAXIES" --POPULATION "$POPULATION" \
    --mbh_mass_method "$MBH_MASS_METHOD" --nucleation_method "$NUCLEATION_METHOD"
# python hierarchical_inference/INF_PLOT_SCRIPTS/posterior_2D.py \
#     --GALAXIES "$GALAXIES" --POPULATION "$POPULATION" \
#     --mbh_mass_method "$MBH_MASS_METHOD" --nucleation_method "$NUCLEATION_METHOD"