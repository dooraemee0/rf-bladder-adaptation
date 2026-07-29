#!/usr/bin/env bash
# ==========================================================================
# GPU1: main table 나머지 baseline 절반 (device_only/coral/mmd/lora_conv)
#   + r_derpp(이전 제안, 참고용)도 여기서 함께.
# ==========================================================================
set -euo pipefail

GPU=1
SEEDS="1,2,3,42"

CKPT="${CLINICAL_CKPT:?set to Stage1 pretrained checkpoint path}"
TEST_IDX="${TEST_IDX_PATH:?set to Stage1 test_idx.pt path}"
SAVE_DIR="./results_final_multiseed_valsplit"

export ABL_DEVICE_BASE="${DATA_ROOT:?set data root}/device"
export ABL_SNUH_EXCEL="${DATA_ROOT:?set data root}/Z_volume_selection.csv"
export ABL_SNUH_DATA="${DATA_ROOT:?set data root}/rf/converted_csv"
export ABL_DEV_VAL_RATIO=0.2
export ABL_CLIN_VAL_RATIO=0.2
export ABL_NUM_WORKERS=8
export ABL_EVAL_NUM_WORKERS=0
export ABL_PIN_MEMORY=1
export ABL_USE_CACHE=1
export ABL_CACHE_DIR="./_prep_cache"

mkdir -p logs/final_valsplit "${SAVE_DIR}"
echo "[GPU1 baselines START] $(date)"

CUDA_VISIBLE_DEVICES="${GPU}" python -u ablation_A_adaptation.py \
  --ckpt "${CKPT}" --test_idx "${TEST_IDX}" \
  --strategies device_only,coral,mmd,lora_conv,r_derpp \
  --seeds "${SEEDS}" --epochs 150 --lr 1e-3 --clinical_ratio 1.0 \
  --save_dir "${SAVE_DIR}" \
  2>&1 | tee "logs/final_valsplit/gpu1_baselines.log"

echo "[GPU1 baselines DONE] $(date)"
