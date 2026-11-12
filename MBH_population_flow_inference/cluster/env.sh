# sourced by the sbatch scripts (cluster); adjust R if you unpacked somewhere else
R=${EMRI_ROOT:-$HOME}/emri_run
export EMRI_DATA=$R/emri_data
export LISA_SNR_PKG=$R/LISA_selection_function/lisa_snr_package
export POPLAR_PATH=$R/poplar
export PYTHONPATH=$R/poplar:$R/StableEMRIFisher:$PYTHONPATH
export MPLBACKEND=Agg
source ${EMRI_ROOT:-$HOME}/envs/redback_few/bin/activate
CODE=$R/EMRI_TDE_flow_inference/MBHpopulation/MBH_population_flow_inference
