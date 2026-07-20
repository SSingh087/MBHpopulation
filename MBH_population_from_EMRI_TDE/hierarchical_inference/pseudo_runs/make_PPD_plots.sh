#!/usr/bin/env bash


export GALAXIES=100000
export UNIVERSE_MODEL=0
export Z_MAX=8.0
export SEED=42
export IS_CLUSTERING=0

POPULATION=$1

echo "=== Generating PPD plots for ${POPULATION} ==="

python PPD.py --GALAXIES "$GALAXIES" --POPULATION "$POPULATION"
