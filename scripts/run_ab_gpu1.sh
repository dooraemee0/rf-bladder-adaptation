#!/usr/bin/env bash
# ==========================================================================
# [GPU 1]  DER++ ablation:  beta = 0  (ground-truth replay 제거)
#
#   L = L_y(f(x_d), y_d) + 0.5 * ||f(x_c) - f_old(x_c)||^2 + 0 * L_y(f(x_c), y_c)
#
#   -> dark experience 만 남음 (개념적으로 LwF 에 가까움)
#
# GPU 0 스크립트(run_ab_gpu0.sh)와 동시에 실행 가능.
# 저장 폴더가 서로 달라 충돌하지 않는다.
# ==========================================================================
set -euo pipefail

GPU=1
SEEDS="1,2,3,42"
SAVE_DIR="./results_derpp_ablation/beta0"

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
# 두 GPU 를 동시에 돌릴 때 캐시 경합을 피하려면 아래처럼 분리한다.
# (같은 캐시를 공유해도 코드가 try/except 로 방어하므로 동작은 하지만,
#  분리하는 편이 안전하고 디스크만 조금 더 쓴다.)
export ABL_CACHE_DIR="./_prep_cache_beta0"

mkdir -p logs "${SAVE_DIR}"
echo "[GPU1 START] beta=0 (no ground-truth replay)  $(date)"

CUDA_VISIBLE_DEVICES="${GPU}" python -u ablation_A_adaptation.py \
  --ckpt "${CKPT}" --test_idx "${TEST_IDX}" \
  --strategies der++ \
  --seeds "${SEEDS}" --epochs 150 --lr 1e-3 \
  --der_alpha 0.5 --der_beta 0.0 \
  --save_dir "${SAVE_DIR}" \
  2>&1 | tee "logs/derpp_beta0.log"

echo "[GPU1 DONE] $(date)"
