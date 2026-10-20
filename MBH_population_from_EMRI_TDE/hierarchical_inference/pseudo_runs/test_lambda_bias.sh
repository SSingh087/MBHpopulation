#!/usr/bin/env bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
POP=$1; SCEN=$2; SURVEY=$3
echo "=== lambda bias test: ${POP} ${SCEN} ${SURVEY} ==="
python test_lambda_bias.py --POP "$POP" --SCEN "$SCEN" --SURVEY "$SURVEY"
