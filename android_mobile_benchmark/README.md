# Android ONNX Runtime benchmark

This project validates one explicitly selected DER++ checkpoint on a physical
Android device. The app and Jetpack Microbenchmark both use ONNX Runtime 1.23.2,
CPU Execution Provider, one intra-op thread, one inter-op thread, batch size 1,
and fixed `(1, 7, 400)` float32 inputs.

The packaged artifact is the U8U8 QDQ per-channel model from seed 3. U8U8 was
selected using aggregate four-seed wearable validation, while seed 3 was
selected separately from wearable and clinical-retention validation metrics.
Neither selection used held-out test performance.

## Install release assets

Download the seed-3 ONNX model and processed validation tensor as described in
the repository root README, then run `reproducibility/install_release_assets.py`.
The script verifies both SHA-256 hashes and installs `model_8bit.onnx` and
`test_inputs.bin` into the app and benchmark asset directories. The associated
`test_manifest.json` files are versioned with the source.

## Build prerequisites

- Android Studio with JDK 17
- Android SDK 36
- A physical Android device with USB debugging enabled
- ONNX Runtime Android 1.23.2
- Jetpack Benchmark 1.4.1

Recorded physical-device outputs and benchmark JSON files are included under
`../deployment/public_results/`. They were generated on the device described in
`DEVICE_METADATA_PUBLIC.json`; a new build is needed only to repeat the tests.

Set the SDK path before building, then verify that a physical device is visible:

```bash
export ANDROID_SDK_ROOT=/absolute/path/to/Android/Sdk
export PATH="$ANDROID_SDK_ROOT/platform-tools:$PATH"
adb devices -l
```

Build and install the non-debuggable app configuration with:

```bash
./gradlew :app:assembleBenchmark
./gradlew :app:installBenchmark
```

## Run prediction concordance and descriptive timing

Open this directory in Android Studio, select the physical device, and run the
`app` configuration. Tap **Run CPU validation**. The app writes:

- `benchmark_report.json`
- `android_predictions.csv`

to its external files directory. Retrieve them with:

```bash
adb pull /sdcard/Android/data/org.dgist.bladderbenchmark/files/benchmark_report.json
adb pull /sdcard/Android/data/org.dgist.bladderbenchmark/files/android_predictions.csv
```

Then run the desktop concordance check:

```bash
python deployment/compare_android_outputs.py \
  --manifest android_mobile_benchmark/app/src/main/assets/test_manifest.json \
  --android-csv android_predictions.csv
```

## Run Jetpack Microbenchmark

Use Android Studio's gutter action for
`OrtModelBenchmark.ortSessionRunBatch1SingleThread`, or run:

```bash
./gradlew :benchmark:connectedReleaseAndroidTest
```

The session and all input tensors are created before `measureRepeated`. The
measured block cycles through held-out tensors and calls only the ORT inference
boundary plus output retrieval. AndroidX writes the authoritative benchmark JSON
under the module's connected-test output directory.

Do not label this result end-to-end latency. Model loading, raw RF preprocessing,
BLE transfer, server communication, and UI updates are excluded.

The additional `RawRfPreprocessingBenchmark` and `RawRfToVolumeBenchmark`
classes start with six-channel RF already in smartphone memory. Their respective
boundaries include preprocessing alone and preprocessing plus ONNX inference.
The raw phantom assets used for the recorded run are not distributed in the Git
repository pending a separate data-distribution review. Supply authorized assets
under the paths named by `local_raw_manifest.json` before repeating those tests.

The paper-facing pooled medians were 14.33 ms for preprocessing and 39.23 ms
for raw-RF-to-scalar-volume. These values exclude live RF acquisition, BLE
transport, network transport, initialization, and screen rendering.
