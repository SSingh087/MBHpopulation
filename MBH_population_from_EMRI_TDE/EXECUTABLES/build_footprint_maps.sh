#!/bin/sh

# One-time build of the ZTF/Rubin coverage maps (coverage_maps_nohealpy.npz).
# Combo-independent -- runs ONCE in the DAG, as a parent of every
# footprint.sh job, instead of being rebuilt per (model, universe) combo.

. EXECUTABLES/config.sh

echo "==========================================================================================="
echo "Building ZTF/LSST coverage maps (one-time, combo-independent)"
echo "==========================================================================================="

python ./data_generation/build_footprint_maps.py
