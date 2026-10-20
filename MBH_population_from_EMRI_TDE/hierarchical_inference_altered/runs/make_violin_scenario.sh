#!/usr/bin/env bash

export GALAXIES=100000
export UNIVERSE_MODEL=0
export Z_MAX=8.0
export SEED=42
export IS_CLUSTERING=0

MBH_MASS_METHOD=$1
NUCLEATION_METHOD=$2
OCCUPATION_METHOD=$3

echo "=== Generating per-scenario violin plots (${MBH_MASS_METHOD}/${NUCLEATION_METHOD}/${OCCUPATION_METHOD}) ==="

for POPULATION in A B; do
    python plots/plot_violin.py --MODE single --GALAXIES "$GALAXIES" --POPULATION "$POPULATION" \
        --mbh_mass_method "$MBH_MASS_METHOD" --nucleation_method "$NUCLEATION_METHOD" \
        --occupation_method "$OCCUPATION_METHOD"
done

python plots/plot_violin.py --MODE grouped --GALAXIES "$GALAXIES" \
    --mbh_mass_method "$MBH_MASS_METHOD" --nucleation_method "$NUCLEATION_METHOD" \
    --occupation_method "$OCCUPATION_METHOD"
