#!/usr/bin/env bash
# ==========================================================================
# [Stage 1 / GPU 1]  λ 스크리닝 -- SI 전 구간 + LwF 나머지
#
#   SI 는 device 까지 붕괴했으므로(0.151) λ 를 [줄이는] 방향을 포함한다.
#   LwF 는 clinical 만 붕괴했으므로 λ 를 키우는 방향.
# ==========================================================================
set -euo pipefail

GPU=1
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
export ABL_CACHE_DIR="./_prep_cache_sw1"

mkdir -p logs "${SAVE_DIR}"
echo "[S1/GPU1 START] $(date)"

CUDA_VISIBLE_DEVICES="${GPU}" python -u lambda_sweep.py \
  --stage 1 --seed 1 \
  --ckpt "${CKPT}" --test_idx "${TEST_IDX}" \
  --jobs "si:0.01,0.1,1.0,10.0|lwf:10.0,100.0" \
  --epochs 150 --lr 1e-3 \
  --save_dir "${SAVE_DIR}" \
  2>&1 | tee "logs/sweep_s1_gpu1.log"

echo "[S1/GPU1 DONE] $(date)"
