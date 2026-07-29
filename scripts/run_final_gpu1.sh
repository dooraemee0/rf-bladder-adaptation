#!/usr/bin/env bash
# ==========================================================================
# GPU1: der++ 와 cfp_derpp 를 seed 2, 42 으로 (validation-selected 프로토콜)
#   - GPU0 스크립트와 동시에 실행. 두 스크립트는 같은 SAVE_DIR에 써도 안전
#     (전략/seed별 하위폴더가 달라 충돌 없음). 단, summary CSV는 각 프로세스가
#     자기 실행분만 집계하므로, 전부 끝난 뒤 aggregate_summary.py로 통합한다.
# ==========================================================================
set -euo pipefail

GPU=1
SEEDS="2,42"

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

echo "[GPU1 START] seeds=${SEEDS}  $(date)"

CUDA_VISIBLE_DEVICES="${GPU}" python -u ablation_A_adaptation.py \
  --ckpt "${CKPT}" --test_idx "${TEST_IDX}" \
  --strategies der++ \
  --seeds "${SEEDS}" --epochs 150 --lr 1e-3 --clinical_ratio 1.0 \
  --der_alpha 0.5 --der_beta 0.5 \
  --save_dir "${SAVE_DIR}" \
  2>&1 | tee "logs/final_valsplit/gpu1_derpp.log"

CUDA_VISIBLE_DEVICES="${GPU}" python -u ablation_A_adaptation.py \
  --ckpt "${CKPT}" --test_idx "${TEST_IDX}" \
  --strategies cfp_derpp \
  --seeds "${SEEDS}" --epochs 150 --lr 1e-3 --clinical_ratio 1.0 \
  --der_alpha 0.5 --der_beta 0.5 \
  --feature_lambda 0.05 --feature_loss_mode cosine \
  --feature_schedule cosine_decay --feature_min_ratio 0.1 --feature_grad_clip 5.0 \
  --save_dir "${SAVE_DIR}" \
  2>&1 | tee "logs/final_valsplit/gpu1_cfp.log"

echo "[GPU1 DONE] $(date)"
