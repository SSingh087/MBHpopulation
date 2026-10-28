#!/bin/bash
# ============================================================
# run_verification_grid.sh
#
# Re-runs the ACTUAL pipeline (sources_num.py -> gen_TDE_params.py ->
# calculate_SNR_TDE.py) for ONE universe realization per UNIVERSE
# scenario (all 8 mbh_mass_method x nucleation_method x occupation_method
# combos), using the config.py values AS THEY CURRENTLY STAND (i.e.
# AFTER the MBH_scatter_sigma 0.53->0.55 fix).
#
# Writes to a brand-new data directory (GALAXIES=$VERIFY_GALAXIES,
# UNIVERSE_MODEL=$VERIFY_UNIVERSE), which is a different subfolder from
# the production GALAXIES=100000 grid -- nothing under
# astrophysical_data/100000/ is read or modified.
#
# Purpose: confirm/refute that PPS resampling (already code-audited and
# found correct) inflates final EMRI/TDE detection counts, and get a
# fresh post-scatter-fix sanity check of the intrinsic Lambda_TDE_tot.
#
# By default only runs the fiducial (TDE_model=tde_fallback, SURVEY=ZTF)
# combo through the full detection classifier per scenario, since that
# is what report_universes.py's Table 3 uses -- set RUN_ALL_MODELS=1 /
# RUN_ALL_SURVEYS=1 to sweep every TDE model / both surveys too (much
# slower: ~1e6-row catalogs per scenario).
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

PYTHON="${PYTHON:-python}"

. ./EXECUTABLES/config.sh   # for T_OBS_*, Z_MAX, TDE_ARCH_* tables

VERIFY_GALAXIES="${VERIFY_GALAXIES:-20000}"     # MC sample size (NOT catalog size -- see note below)
VERIFY_UNIVERSE="${VERIFY_UNIVERSE:-9001}"      # distinct index, avoids any collision even if it ever landed in the same folder
VERIFY_SEED="${VERIFY_SEED:-42}"
MERGER_MODEL="${MERGER_MODEL:-baseline}"     # mundy17 | oleary21 | duan24 | baseline (original 0.25(1+z)^2)

RUN_ALL_MODELS="${RUN_ALL_MODELS:-0}"
RUN_ALL_SURVEYS="${RUN_ALL_SURVEYS:-0}"

SCENARIOS=(
  "use_lgMgal hannah   uniform"
  "use_sigma  hannah   uniform"
  "use_lgMgal neumayer uniform"
  "use_sigma  neumayer uniform"
  "use_lgMgal hannah   varying"
  "use_sigma  hannah   varying"
  "use_lgMgal neumayer varying"
  "use_sigma  neumayer varying"
)

ALL_TDE_MODELS=(cooling_envelope stream_stream_tde tde_analytical tde_fallback)
ALL_SURVEYS=(ZTF LSST)

if [ "$RUN_ALL_MODELS" = "1" ]; then TDE_MODELS=("${ALL_TDE_MODELS[@]}"); else TDE_MODELS=(tde_fallback); fi
if [ "$RUN_ALL_SURVEYS" = "1" ]; then SURVEYS=("${ALL_SURVEYS[@]}"); else SURVEYS=(ZTF); fi

echo "############################################################"
echo "# Verification grid run"
echo "# GALAXIES(verify)=$VERIFY_GALAXIES  UNIVERSE_MODEL=$VERIFY_UNIVERSE  SEED=$VERIFY_SEED  MERGER_MODEL=$MERGER_MODEL"
echo "# TDE models: ${TDE_MODELS[*]}   Surveys: ${SURVEYS[*]}"
echo "# Output dir: astrophysical_data/$VERIFY_GALAXIES/  (separate from astrophysical_data/100000/)"
echo "############################################################"

for scen in "${SCENARIOS[@]}"; do
  read -r MBH NUC OCC <<< "$scen"
  echo ""
  echo "==================== scenario: $MBH / $NUC / $OCC ===================="

  $PYTHON ./data_generation/sources_num.py \
    --UNIVERSE_MODEL "$VERIFY_UNIVERSE" \
    --GALAXIES "$VERIFY_GALAXIES" \
    --Z_MAX "$Z_MAX" \
    --T_OBS_EMRI "$T_OBS_EMRI" \
    --T_OBS_TDE "$T_OBS_TDE" \
    --mbh_mass_method "$MBH" \
    --nucleation_method "$NUC" \
    --occupation_method "$OCC" --merger_model "$MERGER_MODEL" \
    --SEED "$VERIFY_SEED"

  $PYTHON ./data_generation/gen_EMRI_params.py \
    --UNIVERSE_MODEL "$VERIFY_UNIVERSE" \
    --GALAXIES "$VERIFY_GALAXIES" \
    --Y0 0.1 0.7 \
    --T_SIGNAL 1.0 "$T_OBS_EMRI" \
    --mbh_mass_method "$MBH" --nucleation_method "$NUC" --occupation_method "$OCC" --merger_model "$MERGER_MODEL" \
    --SEED "$VERIFY_SEED"

  $PYTHON ./data_generation/calculate_SNR_EMRI.py \
    --UNIVERSE_MODEL "$VERIFY_UNIVERSE" \
    --GALAXIES "$VERIFY_GALAXIES" \
    --mbh_mass_method "$MBH" --nucleation_method "$NUC" --occupation_method "$OCC" --merger_model "$MERGER_MODEL" \
    --MODEL_PATH "$EMRI_SNR_MODEL_PATH" --SCALING_PATH "$EMRI_SNR_SCALING_PATH" --SNR_THRESHOLD "$EMRI_SNR_THRESHOLD"

  for MODEL in "${TDE_MODELS[@]}"; do
    $PYTHON ./data_generation/gen_TDE_params.py \
      --UNIVERSE_MODEL "$VERIFY_UNIVERSE" \
      --GALAXIES "$VERIFY_GALAXIES" \
      --TDE_model "$MODEL" \
      --eta 0.001 0.1 --alpha 0.01 0.1 --beta 1.001 30 \
      --mbh_mass_method "$MBH" --nucleation_method "$NUC" --occupation_method "$OCC" --merger_model "$MERGER_MODEL" \
      --SEED "$VERIFY_SEED"

    for SURVEY in "${SURVEYS[@]}"; do
      NEURONS_VAR="TDE_ARCH_NEURONS_${MODEL}_${SURVEY}"
      LAYERS_VAR="TDE_ARCH_LAYERS_${MODEL}_${SURVEY}"
      NEURONS="${!NEURONS_VAR}"
      LAYERS="${!LAYERS_VAR}"
      ZMIN_VAR="Z_GAL_MIN_${SURVEY}"; ZMAX_VAR="Z_GAL_MAX_${SURVEY}"

      $PYTHON ./data_generation/calculate_SNR_TDE.py \
        --UNIVERSE_MODEL "$VERIFY_UNIVERSE" \
        --GALAXIES "$VERIFY_GALAXIES" \
        --SURVEY "$SURVEY" \
        --TDE_model "$MODEL" \
        --Z_GAL_MIN "${!ZMIN_VAR}" --Z_GAL_MAX "${!ZMAX_VAR}" \
        --DETECTION_THRESHOLD 0.5 \
        --num_neurons "$NEURONS" --layers "$LAYERS" \
        --mbh_mass_method "$MBH" --nucleation_method "$NUC" --occupation_method "$OCC" --merger_model "$MERGER_MODEL"
    done
  done
done

echo ""
echo "Done. Now run: $PYTHON tests/PPS_VERIFICATION/tabulate_grid_counts.py --GALAXIES $VERIFY_GALAXIES --UNIVERSE_MODEL $VERIFY_UNIVERSE"
