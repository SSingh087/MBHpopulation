#!/bin/bash
# Train one selection network: train_selection_function.sh <A|B> <EMRI|TDE_ZTF|TDE_LSST> <tde_model>
. ./EXECUTABLES/config.sh

python ./selection_function/train_selection_function.py --POPULATION "$1" --CHANNEL "$2" --TDE_MODEL "$3"
