#!/bin/bash
# Merge the Fisher batches: merge_fisher.sh <EMRI|TDE_ZTF|TDE_LSST> <scenario>
. EXECUTABLES/config.sh

SOURCE=$1
SCENARIO=$2

if [ "$SOURCE" = "EMRI" ]; then
    python fisher_analysis/merge_fisher_batch_EMRI.py --GALAXIES "${GALAXIES}" --SCENARIO "${SCENARIO}"
else
    python fisher_analysis/merge_fisher_batch_TDE.py --GALAXIES "${GALAXIES}" --SURVEY "${SOURCE#TDE_}" --SCENARIO "${SCENARIO}"
fi
