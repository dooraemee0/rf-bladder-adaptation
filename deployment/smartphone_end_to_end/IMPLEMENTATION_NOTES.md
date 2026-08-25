# Smartphone-resident RF-to-volume implementation notes

> Historical note: this document records the pre-device implementation stage.
> Physical validation was subsequently completed and is reported in
> `../PUBLIC_ANDROID_README.md` and `SMARTPHONE_END_TO_END_RESULTS.md`.

## Scope and isolation

All Android changes are confined to:

`deployment/smartphone_end_to_end/android_mobile_benchmark`

The released checkpoint, FP32 ONNX model, U8U8 QDQ ONNX model, data splits,
test membership, and existing model-only physical benchmark were not modified.
The manuscript TeX was not edited.

## Implemented execution path

The local app path is:

`six physical RF traces -> deterministic Android preprocessing -> (1, 7, 400)
float32 tensor -> frozen U8U8 ONNX model -> scalar volume -> 200 mL alarm -> UI`

No server API is called by this path. The app identifies the mode as
`Local / server-free`, the model as `DER++ seed 3`, and the quantization as
`static U8U8 QDQ`. The existing model-only validation path remains available as
a separate debug action.

## Android preprocessing implementation

The preprocessing code is in the independent `:pipeline` Android library:

- `PreprocessingMetadata.kt`: frozen dimensions, constants, and channel order.
- `TgcProcessor.kt`: two-region exponential TGC.
- `ButterworthFilter.kt`: frozen SciPy 1.15.1 BA coefficients and `lfilter_zi`,
  odd extension, and forward-backward filtering.
- `ChannelConstructor.kt`: physical-to-logical channel conversion and
  `H_mid = mean(S2, S3)`.
- `GaussianChannelSmoother.kt`: sigma-1, radius-4 Gaussian filtering along the
  channel axis, separately within the 3-channel horizontal and 4-channel
  sagittal groups, with reflect boundaries.
- `ChannelStandardizer.kt`: per-channel population standardization and
  non-zero tail-mean padding.
- `WearableRfPreprocessor.kt`: validates the six `(5120,)` inputs and exposes
  all ten intermediate stages for parity analysis.

Malformed shape, NaN, infinity, and invalid filter-length inputs fail
explicitly. Constant channels remain finite because the standard-deviation
denominator includes `1e-8`.

## Validation design

Three samples were selected before Android comparison, one from each wearable
validation volume: `50_30.txt`, `150_30.txt`, and `300_30.txt`. No held-out raw
RF sample or held-out preprocessing golden output is packaged at this stage.
The frozen preprocessed test tensor assets required by the retained legacy
model-only path remain present but are not read by the parity test.
`PreprocessingParityTest` records shape,
dtype, non-finite counts, maximum and mean absolute difference, RMSE, relative
L2 error, and per-channel errors for every stage. It also compares the final
U8U8 prediction. No numerical tolerance was selected in advance; the test only
asserts structural validity and finiteness so that observed differences can be
reported before an acceptance criterion is set.

## Benchmark additions

The existing `OrtModelBenchmark` was retained. Two independent Jetpack
Microbenchmark classes were added:

- `RawRfPreprocessingBenchmark`: six-channel raw RF to `(1, 7, 400)` tensor.
- `RawRfToVolumeBenchmark`: Android preprocessing, tensor construction,
  `OrtSession.run`, output retrieval, and tensor/result closure.

Model, session, and raw asset loading are outside the measured blocks. The raw
benchmark input is a predeclared validation sample and is not used for model or
parameter selection.

## Profiling additions

`DeploymentProfilingTest` writes separate JSON reports for repeated in-process
session initialization and runtime memory point observations. The
initialization report explicitly states that a true fresh-process cold start is
not measured. Memory is sampled at named points; continuous sampling is not
used and no point observation is called a peak.

## Pending physical validation

The Linux host has JDK 17 but no Android SDK, and the Galaxy S8+ is attached to
the Windows host. Therefore Android compilation, instrumentation parity, local
app screenshot capture, and standardized physical benchmarks must be executed
on Windows. The implementation remains unfrozen until validation parity is
reviewed. Held-out raw RF data will only be packaged after that review.
