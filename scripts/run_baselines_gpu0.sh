#!/usr/bin/env bash
# ==========================================================================
# GPU0: main table의 나머지 baseline (전부 같은 validation-selected 프로토콜)
#   공정성을 위해 baseline도 der++/cfp와 동일하게 4 seed로 돌린다.
#   무겁다면 seed를 줄여도 되지만, 표 note에 반드시 명시할 것.
#   여기서는 GPU0에 절반(rehearsal/joint/ewc/si/lwf)을 배정.
# ==========================================================================
set -euo pipefail

GPU=0
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
echo "[GPU0 baselines START] $(date)"

CUDA_VISIBLE_DEVICES="${GPU}" python -u ablation_A_adaptation.py \
  --ckpt "${CKPT}" --test_idx "${TEST_IDX}" \
  --strategies rehearsal,joint,ewc,si,lwf \
  --seeds "${SEEDS}" --epochs 150 --lr 1e-3 --clinical_ratio 1.0 \
  --save_dir "${SAVE_DIR}" \
  2>&1 | tee "logs/final_valsplit/gpu0_baselines.log"

echo "[GPU0 baselines DONE] $(date)"
