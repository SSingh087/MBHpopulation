#!/usr/bin/env bash


export GALAXIES=1000
export UNIVERSE_MODEL=0
export Z_MAX=10.0
export SEED=42
export IS_CLUSTERING=0

POPULATION=$1

echo "=== Generating PPD plots for ${POPULATION} ==="

python PPD.py --GALAXIES "$GALAXIES" --POPULATION "$POPULATION"
