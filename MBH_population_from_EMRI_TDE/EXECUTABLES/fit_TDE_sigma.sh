#!/bin/bash
. EXECUTABLES/config.sh
python ./fisher_analysis/fit_TDE_sigma.py --GALAXIES "${GALAXIES}"
