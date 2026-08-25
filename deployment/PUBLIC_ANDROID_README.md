# Android and ONNX public validation snapshot

## Artifact identity

The deployed model is historical DER++ seed 3, static U8U8 QDQ with per-channel weight quantization.

- Checkpoint SHA-256: `ba9e8bd59b8eeb14ea09cbb2fbd613fad4490699064c25e8d882681d04ce2572`
- ONNX SHA-256: `f473b9b8b4a745a9d7579896619af063d2e9cabfaad71fd699728837073a684e`
- ONNX size: `6,651,648` bytes
- ONNX Runtime Android: `1.23.2`
- Jetpack Benchmark: `1.4.1`
- Gradle wrapper: `8.13`

The ONNX file is distributed once as a GitHub Release asset. Download it and the processed model-input tensor using the links in the repository root README, then install hash-verified copies into the app and benchmark modules with `reproducibility/install_release_assets.py` before building.

## Input contract

Six physical traces are ordered `[S1, S2, S3, S4, H_left, H_right]`. The virtual horizontal channel is the sample-wise mean of `S2` and `S3`. Model channels are `[H_left, H_virtual_mean_S2_S3, H_right, S1, S2, S3, S4]`.

The smartphone pipeline performs amplitude scaling, TGC, seventh-order 1.5--4.5 MHz band-pass filtering at 20 MSPS, clipping/cropping, virtual-channel construction, channel-axis Gaussian smoothing, 10x downsampling to 220 samples, independent channel standardization, and padding to 400 samples. Positions 220--399 are filled with the mean of the final 30 standardized samples of that channel.

## Build

Prerequisites are JDK 17, Android SDK 36, Build Tools 35 or newer, and an Android device with API 26 or newer.

```bash
cd android_mobile_benchmark
./gradlew :pipeline:test
./gradlew :app:assembleBenchmark
./gradlew :benchmark:assembleReleaseAndroidTest
```

## Frozen validation results

Recorded physical-device results are under `deployment/public_results/`.

- Android/Python final preprocessing tensor maximum difference on three validation samples: `2.771615982e-6`.
- Sixty held-out phantom samples: Android/desktop prediction maximum difference `0 mL`.
- Held-out seed-3 U8U8 performance: R2 `0.694224`, MAE `43.2399 mL`, RMSE `56.8123 mL`, 200 mL alarm accuracy `90.0%`, F1 `0.850`.
- Three-session median-of-session-medians: raw RF preprocessing `14.277 ms`; raw-RF-to-scalar-volume `39.202 ms`. The paper-facing pooled summaries were `14.33 ms` and `39.23 ms`.
- Repeated in-process initialization median: `136.829 ms`. This is not a process cold-start measurement.
- Largest observed total PSS point: `61,776 kB`. This is a point observation, not a continuously sampled peak.

## Measurement boundaries

`model_only/activity_benchmark_report.json` measures `OrtSession.run` with a prepared `(1,7,400)` tensor and excludes model loading, RF preprocessing, BLE, and UI updates.

`raw_rf_pipeline/session*_benchmarkData.json` measures smartphone-resident processing with six-channel raw RF already in memory. `sixChannelRawRfToModelTensor` covers preprocessing. `sixChannelRawRfToScalarVolume` covers preprocessing, ONNX inference, and scalar retrieval. BLE reception and screen rendering are excluded.

These measurements are not acquisition-to-readout latency and must not be described as fully on-device monitoring of live sensor data.

## Verification

```bash
python reproducibility/check_onnx_artifact.py /path/to/derpp_seed3_u8u8_qdq.onnx
python reproducibility/verify_recorded_android_results.py --repo-root .
```

Raw phantom RF files are not committed to the Git repository pending separate
redistribution approval. This does not affect inspection of the preprocessing
code, the processed 60-sample model-only validation, or verification of the
recorded physical-device JSON results. Repeating the raw-RF instrumentation
tests requires separately authorized raw assets.
