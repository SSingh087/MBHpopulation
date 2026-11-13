#!/bin/bash
# run ONCE on the cognition login node: downloads a small Debian 12 image (~50 MB)
R=${EMRI_ROOT:-$HOME}/emri_run
apptainer pull "$R/debian12.sif" docker://debian:12-slim
