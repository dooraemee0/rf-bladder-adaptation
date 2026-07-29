#!/usr/bin/env bash
# ==========================================================================
# Supplementary Table S2 재실행: adaptation-scope ablation
#   - 채택 목적함수(DER++) 아래에서 "어느 파라미터를 풀 것인가"만 바꿔 비교
#   - 본 실험과 동일한 validation-selected + 4 seed 프로토콜
#
# 소요 시간: 3 scope x 4 seed x 150 epoch = DER++ 12회분
#            (GPU 1장 기준 대략 본 실험의 3배)
#   급하면 --seeds 1,42 로 줄여도 되지만 표 caption 에 seed 수를 명시할 것.
# ==========================================================================
set -euo pipefail

GPU=1
SEEDS="1,2,3,42"
SCOPES="head_only,last_conv,all_conv"
SAVE_DIR="./results_scope_ablation"

CKPT="${CLINICAL_CKPT:?set to Stage1 pretrained checkpoint path}"
TEST_IDX="${TEST_IDX_PATH:?set to Stage1 test_idx.pt path}"

# 본 실험과 동일해야 split 이 재현된다
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

mkdir -p logs "${SAVE_DIR}"

echo "[START] scope ablation $(date)"

CUDA_VISIBLE_DEVICES="${GPU}" python -u ablation_scope.py \
  --ckpt "${CKPT}" --test_idx "${TEST_IDX}" \
  --scopes "${SCOPES}" \
  --seeds "${SEEDS}" \
  --epochs 150 --lr 1e-3 \
  --der_alpha 0.5 --der_beta 0.5 \
  --save_dir "${SAVE_DIR}" \
  2>&1 | tee "logs/scope_ablation.log"

echo "[DONE] $(date)"
echo ""
echo "논문에 붙여넣을 표: ${SAVE_DIR}/table_s2.tex"
