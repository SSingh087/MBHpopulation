#!/usr/bin/env bash

export GALAXIES=100000
export UNIVERSE_MODEL=0
export Z_MAX=8.0
export SEED=42
export IS_CLUSTERING=0

POPULATION=$1

echo "=== Generating violin plot across all scenarios for population ${POPULATION} ==="

python plots/plot_violin.py --GALAXIES "$GALAXIES" --POPULATION "$POPULATION"

echo "=== Generating violin plot with asinh zoom panel for population ${POPULATION} ==="

python plots/plot_violin_zoom.py --GALAXIES "$GALAXIES" --POPULATION "$POPULATION" --ASINH 1
