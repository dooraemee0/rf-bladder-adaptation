# Latency boundaries

All application timestamps use Android's monotonic
`SystemClock.elapsedRealtimeNanos()` clock.

## Local raw-file execution

- `T_input_ready`: the six raw RF arrays have been loaded and validated.
- `T_preprocessing_done`: the `(1, 7, 400)` model input is complete.
- `T_inference_done`: ONNX Runtime has returned and the scalar output has been
  retrieved.
- `T_ui_request`: the UI-thread update has been requested.

Derived intervals:

- preprocessing: `T_preprocessing_done - T_input_ready`
- model inference: `T_inference_done - T_preprocessing_done`
- raw-RF-to-volume: `T_inference_done - T_input_ready`
- UI scheduling request: `T_ui_request - T_inference_done`

These Activity-level values are diagnostic only. Standardized warm latency is
reported from Jetpack Microbenchmark, not from the app screen.

## Jetpack Microbenchmark boundaries

- preprocessing-only includes preprocessing of six in-memory `(5120,)` raw RF
  traces through production of the flat `(7, 400)` float32 tensor.
- inference-only retains the existing official boundary: `OrtSession.run` and
  output retrieval for one pre-created `(1, 7, 400)` tensor.
- combined raw-RF-to-volume includes preprocessing, ONNX tensor creation,
  `OrtSession.run`, scalar output retrieval, and tensor/result closure.

Asset reads, session creation, and validation-sample loading are outside all
warm timed blocks. Batch size is one; CPU EP, intra-op one, inter-op one, and
sequential execution are fixed.

## BLE terminology boundary

BLE acquisition and transfer are not included in the implemented local-file
path. Until a packet-complete callback is implemented and physically measured,
the terms `BLE-reception-to-display`, `request-to-display`, and `end-to-end
latency` must not be used for these results.

If BLE is later completed, the planned timestamps are:

- `T0`: acquisition request write initiated by the phone.
- `T1`: final valid packet accepted and one six-channel measurement assembled.
- `T2`: preprocessing complete.
- `T3`: inference and output retrieval complete.
- `T4`: UI update requested.
- `T5`: rendered frame confirmed, only if frame instrumentation is available.
