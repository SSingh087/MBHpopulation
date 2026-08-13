#!/usr/bin/env bash

. EXECUTABLES/config.sh

N_UNIVERSES=$1
echo "=== report_universes: scanning ${N_UNIVERSES} universes per scenario ==="

python report_universes.py \
    --GALAXIES "$GALAXIES" \
    --N_UNIVERSES "$N_UNIVERSES"