#!/bin/sh
. EXECUTABLES/config.sh
python merge_TDE_SNRs.py --events $EVENTS --SURVEY $1 --n_chunks $2 --models $MODELS
