#!/usr/bin/env bash
# ==========================================================================
# 채택 모델(DER++)로 in vitro / cross-domain 수치 재생성
#   - 학습 없이 평가만 하므로 GPU 1장이면 충분하고 몇 분이면 끝난다.
#   - 반드시 학습 때와 동일한 데이터 경로/환경변수를 써야 split이 재현된다.
# ==========================================================================
set -euo pipefail

GPU=0
RESULTS_DIR="./results_final_multiseed_valsplit"
STRATEGY="der++"
SEEDS="1,2,3,42"
OUT_DIR="./final_invitro"

TEST_IDX="${TEST_IDX_PATH:?set to Stage1 test_idx.pt path}"

# 학습 때와 동일하게 (split 재현에 필수)
export ABL_DEVICE_BASE="${DATA_ROOT:?set data root}/device"
export ABL_SNUH_EXCEL="${DATA_ROOT:?set data root}/Z_volume_selection.csv"
export ABL_SNUH_DATA="${DATA_ROOT:?set data root}/rf/converted_csv"
export ABL_DEV_VAL_RATIO=0.2
export ABL_CLIN_VAL_RATIO=0.2

export ABL_NUM_WORKERS=4
export ABL_EVAL_NUM_WORKERS=0
export ABL_PIN_MEMORY=1
export ABL_USE_CACHE=1
export ABL_CACHE_DIR="./_prep_cache"

mkdir -p logs "${OUT_DIR}"

echo "[START] in vitro re-evaluation $(date)"

CUDA_VISIBLE_DEVICES="${GPU}" python -u eval_invitro_final.py \
  --results_dir "${RESULTS_DIR}" \
  --strategy "${STRATEGY}" \
  --seeds "${SEEDS}" \
  --test_idx "${TEST_IDX}" \
  --n_boot 2000 \
  --boot_seed 42 \
  --out_dir "${OUT_DIR}" \
  2>&1 | tee "logs/eval_invitro.log"

echo "[DONE] $(date)"
echo ""
echo "논문에 붙여넣을 값: ${OUT_DIR}/paper_values.txt"
