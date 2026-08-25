# Envelope ablation reproducibility record

## Fixed protocol

- Git base commit: `d28c8040fa8d360cb47bc5b60453fe52b6b3709e`
- Dataset split SHA-256: `51a7e0c71fdc038dd9d07a6de978cd5d84e1a1fae265df51bb901ef8b6832668`
- Stage-1 checkpoint SHA-256: `cfb8f0dd43387a59e53c6eba0eb32fa980b883198afee5da77cdba596de1d0d0`
- Python 3.10.16, PyTorch 2.6.0+cu118, NumPy 2.1.2, SciPy 1.15.1
- Training hardware: two NVIDIA GeForce RTX 3090 Ti GPUs
- Final test/direct-transfer evaluation: CPU; predictions use the same deterministic preprocessing and checkpoint states
- Selection rule was written before common-envelope test evaluation. Test metrics were not used for checkpoint or preprocessing selection.

Condition B changed only wearable Hilbert-envelope detection (`axis=1`) before the retained legacy Gaussian smoothing (`axis=0`). Stage 1, clinical preprocessing, TGC, filtering, clipping, crop, channel construction/order, downsampling, standardization, padding, optimizer, learning rate, scheduler, trainable scope, epochs, DER++ coefficients, splits, and seeds remained fixed.

## Checkpoints and logs

| Seed | Selected epoch | Stage-2 SHA-256 | Training time | Training-log SHA-256 |
|---:|---:|---|---:|---|
| 1 | 120 | `790860cbdebb315a1622e6169206b868d7579103a88ae1eaf890d5ba3deba1c6` | 284.318 s | `6c6b06c9520e4a0b08d15d1ed80b34bb6d528a235fb68a9a496a8ddcc6e4ba52` |
| 2 | 150 | `3a2e26fb00f21e3a30346a81fcd357e8087035b1211e80d3834eda5ad43177f0` | 281.503 s | `cc9a7658b951ccb5642eda1c0e8addfd055513177c810230411ec8a89b1ceb14` |
| 3 | 90 | `ac2a535e8a1746218299e7ae118a6c4abd34667fb2f27d8ceb817501841c6d6b` | 265.752 s | `c97543ae6b4f1443c156d6582dcab466ec37b75fae04e8ba05b36c96dd79e4de` |
| 42 | 60 | `7b9d9047ba001b139c57d3e132157836a463128066af34c5f7b9ec419b08f9b8` | 261.718 s | `629b5fb64a520f397954a77b96039217616574176ff39958eeb9a5728761ff84` |

Checkpoints, per-seed logs, validation predictions, validation metrics, split manifests, and preprocessing configurations were stored outside the source tree during the original analysis. They are not part of this public historical release unless listed in the bundle manifest.

## Analysis artifacts

| Artifact | SHA-256 |
|---|---|
| Preprocessing numerical tests | `16f5bb102514183e46fa26f0196451fc32018f1c9534b63ba8d4868ca857aeda` |
| Condition A reproduction | `7b425d3b11308fb75508e80e06b8d0dd394d262ab2f78c7f8bb22abdccb84aa0` |
| Validation decision before test | `c6b8740e9e27dbc8aa32274064d412c4dcc9fdf5360dcc4c990b14625573f191` |
| Final consolidated JSON | `4fea2df7b111568df6bf2c09a8b4b939b3b174933f80d0923ecf329ecc0c3dab` |

The first two artifacts are under `deployment/artifacts/envelope_ablation/`. The final decision, test predictions, direct-transfer predictions, and consolidated JSON are under the ignored `deployment/envelope_ablation_run/` directory.

## Commands

```bash
python deployment/test_envelope_preprocessing.py \
  --device-data-root <WEARABLE_DATA_ROOT> \
  --output deployment/artifacts/envelope_ablation/preprocessing_numerical_tests.json

python deployment/run_envelope_ablation_analysis.py \
  --phase reproduce-a \
  --stage1-checkpoint <WORKSPACE_ROOT>/rf-bladder/checkpoints/best_model.pt \
  --test-index <WORKSPACE_ROOT>/rf-bladder/checkpoints/test_idx.pt \
  --device-root <WEARABLE_DATA_ROOT> \
  --clinical-excel <CLINICAL_TABLE> \
  --clinical-rf-root <CLINICAL_RF_ROOT> \
  --released-checkpoint-root checkpoints/final/der++ \
  --common-envelope-checkpoint-root checkpoints/envelope_ablation/common_envelope \
  --output-root deployment/artifacts/envelope_ablation

CUDA_VISIBLE_DEVICES=0 python deployment/train_envelope_stage2.py \
  --stage1-checkpoint <WORKSPACE_ROOT>/rf-bladder/checkpoints/best_model.pt \
  --test-index <WORKSPACE_ROOT>/rf-bladder/checkpoints/test_idx.pt \
  --device-root <WEARABLE_DATA_ROOT> \
  --clinical-excel <CLINICAL_TABLE> \
  --clinical-rf-root <CLINICAL_RF_ROOT> \
  --output-root checkpoints/envelope_ablation \
  --cache-dir <CACHE_ROOT>/gpu0 \
  --seeds 1,3 --epochs 150 --lr 0.001 --der-alpha 0.5 --der-beta 0.5
```

The GPU-1 worker used the same command with `CUDA_VISIBLE_DEVICES=1`, cache `gpu1`, and seeds `2,42`. The complete final-analysis command is embedded in `deployment/envelope_ablation_run/validation_decision_before_test.json`.

No manuscript TeX, released checkpoint, quantized ONNX model, or Android asset was modified.
