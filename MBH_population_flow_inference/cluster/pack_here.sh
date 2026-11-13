#!/bin/bash
# RUN ON THE OLD MACHINE (wiay). Builds /data/wiay/postgrads/shashwat/cluster_bundle/ (~4.5 GB).
# The three pieces are built IN PARALLEL, compression uses pigz (all cores), each has its own log.
OUT=/data/wiay/postgrads/shashwat/cluster_bundle; D=/data/wiay/postgrads/shashwat/EMRI_TDE_flow_inference
mkdir -p $OUT/logs $OUT/emri_data/{models/detectability,simulations/consolidated}
Z="pigz -p 24"

env_pack() {  # 1. conda env (uncompressed tar from conda-pack, then pigz)
  set -e
  python3 -m venv $OUT/.cp && $OUT/.cp/bin/pip install conda-pack
  $OUT/.cp/bin/conda-pack -p /home/2673888s/.conda/envs/redback_few -o $OUT/redback_few.tar \
      --ignore-editable-packages --ignore-missing-files -f
  $Z -f $OUT/redback_few.tar
  rm -rf $OUT/.cp
}
code_pack() {  # 2. code, local source packages, poplar, StableEMRIFisher
  set -e
  tar cf - --exclude='__pycache__' --exclude='.git' -C /home/2673888s \
      EMRI_TDE_flow_inference/MBHpopulation LISA_selection_function/lisa_snr_package poplar \
      -C /data/wiay/postgrads/shashwat/conda_envs StableEMRIFisher | $Z > $OUT/code.tar.gz
}
data_pack() {  # 3. EMRI data + FEW files
  set -e
  cp -r $D/simulations/p0e0_traj $OUT/emri_data/simulations/
  cp $D/simulations/consolidated/emri.npz $OUT/emri_data/simulations/consolidated/
  cp -r $D/models/detectability/emri_snr $OUT/emri_data/models/detectability/
  cp -r /home/2673888s/.local/share/few $OUT/few_files
  tar cf - -C $OUT emri_data few_files | $Z > $OUT/emri_data.tar.gz
  rm -rf $OUT/emri_data $OUT/few_files
}

env_pack  > $OUT/logs/env.log  2>&1 & P1=$!
code_pack > $OUT/logs/code.log 2>&1 & P2=$!
data_pack > $OUT/logs/data.log 2>&1 & P3=$!
fail=0
wait $P1 || { echo "FAILED: env   (see $OUT/logs/env.log)";  fail=1; }
wait $P2 || { echo "FAILED: code  (see $OUT/logs/code.log)"; fail=1; }
wait $P3 || { echo "FAILED: data  (see $OUT/logs/data.log)"; fail=1; }
ls -la $OUT
[ $fail = 0 ] && echo "ALL OK. copy to the cluster:  rsync -avP $OUT/{redback_few.tar.gz,code.tar.gz,emri_data.tar.gz} <user>@<login-host>:~/"
