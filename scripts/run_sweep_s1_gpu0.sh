#!/usr/bin/env bash
# ==========================================================================
# [Stage 1 / GPU 0]  λ 스크리닝 -- EWC 전 구간 + LwF 절반
#
#   seed 1 하나로 여러 λ 를 빠르게 훑는다. 최선 λ 는 Stage 2 에서 확정.
#   EWC: device 는 정상인데 clinical 만 붕괴했으므로 λ 를 키우는 방향이 핵심.
# ==========================================================================
set -euo pipefail

GPU=0
CKPT="${CLINICAL_CKPT:?set to Stage1 pretrained checkpoint path}"
TEST_IDX="${TEST_IDX_PATH:?set to Stage1 test_idx.pt path}"
SAVE_DIR="./results_lambda_sweep"

export ABL_DEVICE_BASE="${DATA_ROOT:?set data root}/device"
export ABL_SNUH_EXCEL="${DATA_ROOT:?set data root}/Z_volume_selection.csv"
export ABL_SNUH_DATA="${DATA_ROOT:?set data root}/rf/converted_csv"
export ABL_DEV_VAL_RATIO=0.2
export ABL_CLIN_VAL_RATIO=0.2

export ABL_NUM_WORKERS=8
export ABL_EVAL_NUM_WORKERS=0
export ABL_PIN_MEMORY=1
export ABL_USE_CACHE=1
export ABL_CACHE_DIR="./_prep_cache_sw0"

mkdir -p logs "${SAVE_DIR}"
echo "[S1/GPU0 START] $(date)"

CUDA_VISIBLE_DEVICES="${GPU}" python -u lambda_sweep.py \
  --stage 1 --seed 1 \
  --ckpt "${CKPT}" --test_idx "${TEST_IDX}" \
  --jobs "ewc:100,1000,10000,100000,1000000|lwf:0.1,1.0" \
  --epochs 150 --lr 1e-3 \
  --save_dir "${SAVE_DIR}" \
  2>&1 | tee "logs/sweep_s1_gpu0.log"

echo "[S1/GPU0 DONE] $(date)"
