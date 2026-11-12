#!/bin/bash
# RUN ON THE CLUSTER LOGIN NODE.  The three tarballs are in $TB (default $HOME); everything is installed under $EMRI_ROOT (default $HOME).
#   EMRI_ROOT=/path/with/space TB=$HOME bash setup_on_cluster.sh
# after it: rm $TB/*.tar.gz  (frees ~4.5 GB)
set -e; ROOT=${EMRI_ROOT:-$HOME}; TB=${TB:-$HOME}; mkdir -p $ROOT/emri_run $ROOT/envs/redback_few; cd $ROOT
tar xzf $TB/code.tar.gz -C emri_run; tar xzf $TB/emri_data.tar.gz -C emri_run
tar xzf $TB/redback_few.tar.gz -C envs/redback_few; source envs/redback_few/bin/activate; conda-unpack
cp emri_run/EMRI_TDE_flow_inference/MBHpopulation/MBH_population_flow_inference/cluster/* emri_run/; mkdir -p $HOME/.local/share; cp -r emri_run/few_files $HOME/.local/share/few        # FEW data files
# the editable stableemrifisher install is replaced by PYTHONPATH (set in the sbatch scripts)
echo "env ok:"; python -c "import numpy, scipy, torch; print('torch', torch.__version__, 'cuda built', torch.version.cuda)"
