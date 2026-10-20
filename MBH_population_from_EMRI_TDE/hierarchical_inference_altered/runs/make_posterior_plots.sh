#!/usr/bin/env bash

export GALAXIES=100000
export UNIVERSE_MODEL=0
export Z_MAX=8.0
export SEED=42
export IS_CLUSTERING=0

POPULATION=$1
MBH_MASS_METHOD=$2
NUCLEATION_METHOD=$3
OCCUPATION_METHOD=$4

echo "=== Generating 1D+2D posterior plots for ${POPULATION} (${MBH_MASS_METHOD}/${NUCLEATION_METHOD}/${OCCUPATION_METHOD}) ==="

python plots/posterior_plots.py --GALAXIES "$GALAXIES" --POPULATION "$POPULATION" \
    --mbh_mass_method "$MBH_MASS_METHOD" --nucleation_method "$NUCLEATION_METHOD" \
    --occupation_method "$OCCUPATION_METHOD"
