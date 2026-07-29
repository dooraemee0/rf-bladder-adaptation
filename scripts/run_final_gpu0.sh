#!/usr/bin/env bash
# ==========================================================================
# GPU0: der++ 와 cfp_derpp 를 seed 1, 3 으로 (validation-selected 프로토콜)
#   - GPU1 스크립트와 함께 돌리면 4 seed x {der++, cfp} 가 2 GPU에 분산됨.
#   - 나머지 baseline(rehearsal/joint/ewc/... )은 run_baselines_gpu0.sh 참고.
# ==========================================================================
set -euo pipefail

GPU=0
SEEDS="1,3"

CKPT="${CLINICAL_CKPT:?set to Stage1 pretrained checkpoint path}"
TEST_IDX="${TEST_IDX_PATH:?set to Stage1 test_idx.pt path}"
SAVE_DIR="./results_final_multiseed_valsplit"

# 데이터 경로
export ABL_DEVICE_BASE="${DATA_ROOT:?set data root}/device"
export ABL_SNUH_EXCEL="${DATA_ROOT:?set data root}/Z_volume_selection.csv"
export ABL_SNUH_DATA="${DATA_ROOT:?set data root}/rf/converted_csv"

# validation 분리 비율 (기본 0.2). 필요시 조정.
export ABL_DEV_VAL_RATIO=0.2
export ABL_CLIN_VAL_RATIO=0.2

# 로더/캐시
export ABL_NUM_WORKERS=8
export ABL_EVAL_NUM_WORKERS=0
export ABL_PIN_MEMORY=1
export ABL_USE_CACHE=1
export ABL_CACHE_DIR="./_prep_cache"

mkdir -p logs/final_valsplit "${SAVE_DIR}"

echo "[GPU0 START] seeds=${SEEDS}  $(date)"

# der++
CUDA_VISIBLE_DEVICES="${GPU}" python -u ablation_A_adaptation.py \
  --ckpt "${CKPT}" --test_idx "${TEST_IDX}" \
  --strategies der++ \
  --seeds "${SEEDS}" --epochs 150 --lr 1e-3 --clinical_ratio 1.0 \
  --der_alpha 0.5 --der_beta 0.5 \
  --save_dir "${SAVE_DIR}" \
  2>&1 | tee "logs/final_valsplit/gpu0_derpp.log"

# cfp_derpp (cosine_decay schedule, feature_lambda 0.05 — 기존 최종 세팅)
CUDA_VISIBLE_DEVICES="${GPU}" python -u ablation_A_adaptation.py \
  --ckpt "${CKPT}" --test_idx "${TEST_IDX}" \
  --strategies cfp_derpp \
  --seeds "${SEEDS}" --epochs 150 --lr 1e-3 --clinical_ratio 1.0 \
  --der_alpha 0.5 --der_beta 0.5 \
  --feature_lambda 0.05 --feature_loss_mode cosine \
  --feature_schedule cosine_decay --feature_min_ratio 0.1 --feature_grad_clip 5.0 \
  --save_dir "${SAVE_DIR}" \
  2>&1 | tee "logs/final_valsplit/gpu0_cfp.log"

echo "[GPU0 DONE] $(date)"
