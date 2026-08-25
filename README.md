# Bladder Volume Estimation from RF Ultrasound — Stage 2: Clinical-to-Wearable Adaptation

Code, experiment wrappers, and trained checkpoints for the wearable-adaptation
stage of our work on bladder-volume estimation directly from radio-frequency
(RF) ultrasound. A clinically pretrained per-channel RF regression model is
adapted to wearable phantom acquisitions with three reference volumes. Device
performance and clinical-retention performance are reported separately.

This repository covers **Stage 2 (adaptation)**. Stage 1 (clinical pretraining)
lives in a separate repository:

- **Stage 1 — clinical pretraining:** https://github.com/dooraemee0/rf-bladder
- **Stage 2 — adaptation (this repo):** clinical → wearable, DER++ and variants
- **Trained checkpoints (12):** GitHub Release
  [`v1.0.0`](https://github.com/dooraemee0/rf-bladder-adaptation/releases/tag/v1.0.0)
  (`rf-bladder-adaptation-checkpoints.zip`, SHA256
  `6a1bc47398f893c24a629f645f64ad9075f05a2d7ae27042694cb55ce8502020`)

The two stages connect through the released Stage-1 checkpoint: every reported
Stage-2 run starts from `rf-bladder/checkpoints/best_model.pt` and uses the
recorded clinical evaluation indices in `rf-bladder/checkpoints/test_idx.pt`.

## Method in one paragraph

Device data cover only three discrete phantom volumes (50, 150, 300 mL), which
makes continuous regression from them ill-posed: infinitely many curves pass
through three points. Fine-tuning on device data alone therefore raises the
device score while destroying the clinical volume relationship — memorization
that a high device R² can hide. We treat deployment as a stability–plasticity
problem and adapt with **DER++ (Dark Experience Replay++)**: alongside the device
loss, clinical inputs are replayed and the frozen pretrained model's responses on
them are matched, pinning the input→output map along the clinical volume
manifold. We compare this against regularization, distillation, feature-alignment,
and parameter-efficient fine-tuning families under one multi-seed,
validation-controlled protocol, and against two DER++ variants introduced here
(CFP-DER++, R-DER++).

## Repository layout

```
.
├── model.py                    # RFNet: per-channel 1D-conv encoders + regression head
├── dataloader.py               # RF loading/preprocessing; builds the H3 synthetic channel
├── ablation_common.py          # data splits, RNG-safe eval, validation-based checkpoint selection
├── strategies.py               # DER++ / CFP-DER++ / R-DER++ + baselines (rehearsal, EWC, SI, LwF, CORAL, MMD, LoRA-conv, ...)
├── ablation_A_adaptation.py    # Stage-2 runner: train → restore best-on-val → test once   (Table 3)
├── ablation_scope.py           # adaptation-scope ablation (Supp.)
├── lambda_sweep.py             # EWC/SI/LwF regularization-strength search (Supp.)
├── eval_invitro_final.py       # re-evaluate checkpoints; cross-domain + Fig. 4 CSVs   (Table 4)
├── plot_fig4.py                # redraw Fig. 4a/4b
├── aggregate_summary.py        # merge metrics.json across runs → mean ± seed std
├── summarize_ab_ablation.py    # Table 4 (alpha=0 / beta=0) summary
├── summarize_lambda_sweep.py   # lambda-sweep summary
├── _base.yaml                  # human-readable common experiment settings
├── table3_*.yaml               # human-readable main-strategy settings
├── table4_*.yaml               # human-readable DER++ ablation settings
├── scripts/                    # runnable .sh wrappers (paths via env vars)
├── release/checkpoint_manifest.json  # release ZIP/model hashes and byte sizes
├── deployment/                 # ONNX export, quantization, selection, and validation
├── android_mobile_benchmark/   # Android app, preprocessing module, and benchmarks
├── reproducibility/            # artifact manifests and recorded-result checks
├── checkpoints/final/          # 12 released checkpoints (der++/cfp_derpp/r_derpp × 4 seeds)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # then edit .env with your local paths
```

`.env` provides three paths (see `.env.example`): `DATA_ROOT` (RF datasets),
`CLINICAL_CKPT` and `TEST_IDX_PATH` (the Stage-1 checkpoint and test indices).
The scripts fail fast if any is unset.

## Reproducing the paper

The root-level YAML files record the settings associated with each result.
Executable arguments are supplied by the shell wrappers in `scripts/`; common
optimizer and loader settings are implemented in `strategies.py` and
`ablation_common.py`. `_base.yaml` records AdamW, weight decay 1e-3, cosine
schedule, batch size 16, Huber loss with δ = 10, 150 epochs, and seeds
1/2/3/42 in one place for inspection.

**Table 3 — main comparison.** Adapt from the Stage-1 checkpoint with each
strategy across four seeds:

```bash
source .env
bash scripts/run_final_gpu0.sh      # der++ / cfp_derpp (seeds 1, 3)
bash scripts/run_final_gpu1.sh      # der++ / cfp_derpp (seeds 2, 42)
bash scripts/run_baselines_gpu1.sh  # r_derpp + baselines
python aggregate_summary.py         # mean ± seed std
```

**Table 4 — DER++ objective ablation.** Remove each replay term in turn:

```bash
bash scripts/run_ab_gpu0.sh   # alpha = 0 (no response matching)
bash scripts/run_ab_gpu1.sh   # beta  = 0 (no ground-truth replay)
python summarize_ab_ablation.py
```

**Evaluate the released checkpoints (no retraining).** Score the 12 checkpoints
in `checkpoints/final/` and regenerate the Fig. 4 CSVs:

```bash
bash scripts/run_eval_invitro.sh
python plot_fig4.py
```

Supplementary: `scripts/run_scope_ablation.sh` (adaptation scope) and
`scripts/run_sweep_s1_gpu{0,1}.sh` (EWC/SI/LwF λ search).

## Released checkpoints

`checkpoints/final/` contains the exact models behind Table 3:

```
checkpoints/final/{der++,cfp_derpp,r_derpp}/seed{1,2,3,42}/model.pt   # 12 files
```

Each is a pure `state_dict` (loadable with `weights_only=True`), saved by
`ablation_A_adaptation.py`. They are not committed to the repository; download
them from the GitHub Release above and unzip into place:

```bash
# from the repo root, after downloading the release asset
python scripts/verify_released_checkpoints.py \
  --zip rf-bladder-adaptation-checkpoints.zip
unzip rf-bladder-adaptation-checkpoints.zip
python scripts/verify_released_checkpoints.py \
  --checkpoint-root checkpoints/final
```

The verifier uses only the Python standard library. It checks the release ZIP
hash and size as well as the SHA-256 hash and byte size of every extracted model.
The expected release ZIP SHA-256 is
`6a1bc47398f893c24a629f645f64ad9075f05a2d7ae27042694cb55ce8502020`.

## Android deployment and physical-device validation

The repository includes the wearable RF preprocessing implementation, ONNX
export and U8U8 QDQ quantization scripts, Android application, and Jetpack
Microbenchmark source. The selected deployment artifact is DER++ seed 3:

- checkpoint SHA-256: `ba9e8bd59b8eeb14ea09cbb2fbd613fad4490699064c25e8d882681d04ce2572`
- U8U8 QDQ ONNX SHA-256: `f473b9b8b4a745a9d7579896619af063d2e9cabfaad71fd699728837073a684e`
- ONNX file size: 6,651,648 bytes

Download the ONNX model and processed 60-sample validation tensor from the
GitHub Release, then install verified copies into the Android modules:

```bash
curl -fL -O https://github.com/dooraemee0/rf-bladder-adaptation/releases/download/v1.0.0/derpp_seed3_u8u8_qdq.onnx
curl -fL -O https://github.com/dooraemee0/rf-bladder-adaptation/releases/download/v1.0.0/mobile_test_inputs.bin
python reproducibility/install_release_assets.py \
  --repo-root . \
  --onnx derpp_seed3_u8u8_qdq.onnx \
  --test-inputs mobile_test_inputs.bin
```

Recorded Android outputs are checked without regenerating performance metrics:

```bash
python reproducibility/verify_recorded_android_results.py --repo-root .
python -m unittest reproducibility/test_release.py
```

On a host with JDK 17 and Android SDK 36:

```bash
cd android_mobile_benchmark
./gradlew :pipeline:test
./gradlew :app:assembleBenchmark
./gradlew :benchmark:assembleReleaseAndroidTest
```

The physical-device snapshot used ONNX Runtime Android 1.23.2, one CPU thread,
and batch size 1 on a Galaxy S8+. Android and desktop predictions were identical
for all 60 processed held-out phantom inputs. The three-session
median-of-session-medians was 14.277 ms for smartphone-resident RF preprocessing
and 39.202 ms for preprocessing plus ONNX inference and scalar retrieval. The
paper-facing pooled summaries were 14.33 ms and 39.23 ms, respectively. These
boundaries begin with six-channel RF already in smartphone memory and exclude
live acquisition, BLE transport, networking, and screen rendering; they are not
acquisition-to-readout latency.

See [`deployment/PUBLIC_ANDROID_README.md`](deployment/PUBLIC_ANDROID_README.md)
for the complete input contract, recorded results, and interpretation limits.

## Data availability

Clinical RF data are **not publicly available** owing to patient privacy and IRB
restrictions (SNUH IRB No. H-2107-024-1233). The code references clinical file
and column conventions (e.g. `Z_volume_selection.csv`, `Upright_H`/`Upright_S`)
but contains no patient data. Benchtop phantom procedures are described in the
paper. Re-evaluating the checkpoints or retraining the models requires the RF
data under the stated data-use conditions; the public release permits the code,
configuration, and model artifacts to be inspected and hash-verified without
those data.

The mobile test tensor distributed as a GitHub Release asset contains processed
benchtop-phantom model inputs only. Raw phantom RF recordings are not
distributed in this repository; the raw-RF instrumentation tests can be rerun
after separately authorized assets are supplied in the paths documented by the
Android project.

## Citation

```bibtex
@article{TODO,
  title   = {TODO},
  author  = {TODO},
  journal = {TODO},
  year    = {2026}
}
```

## License

Code is released under the MIT License (see `LICENSE`). Trained checkpoints and
any data are subject to separate terms stated with the release asset and in the
Data availability section above.
