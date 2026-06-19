#!/usr/bin/env bash


export GALAXIES=1000
export UNIVERSE_MODEL=0
export Z_MAX=10.0
export SEED=42
export IS_CLUSTERING=0

POPULATION=$1

echo "=== Generating posterior plots for ${POPULATION} ==="

python posterior_1D.py --GALAXIES "$GALAXIES" --POPULATION "$POPULATION"
# python hierarchical_inference/INF_PLOT_SCRIPTS/posterior_2D.py     --GALAXIES "$GALAXIES" --POPULATION "$POPULATION"
