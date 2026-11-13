#!/bin/bash
# usage: bash ctr.sh <command...>   runs the command inside a Debian-12 container (glibc 2.36) with GPU + the redback_few env.
R=${EMRI_ROOT:-$HOME}/emri_run
E=${EMRI_ROOT:-$HOME}/envs/redback_few
exec apptainer exec --nv -B /users "$R/debian12.sif" bash -c '
export PATH='"$E"'/bin:$PATH CONDA_PREFIX='"$E"'
source '"$R"'/env.sh
exec "$@"' _ "$@"
