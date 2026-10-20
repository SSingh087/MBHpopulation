#!/usr/bin/env bash

export GALAXIES=100000
export UNIVERSE_MODEL=0
export Z_MAX=8.0
export SEED=42
export IS_CLUSTERING=0

echo "=== Generating grand combined (all-scenarios) violin plot ==="

python plots/plot_violin.py --MODE all --GALAXIES "$GALAXIES"
