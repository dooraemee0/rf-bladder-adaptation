# Public release integration audit

## Scope and lineage

This public snapshot integrates the deployment, Android, and reproducibility
materials from the historical DER++ internal-feasibility lineage. Its recorded
base is commit `d28c8040fa8d360cb47bc5b60453fe52b6b3709e` plus the tracked server
changes listed in `SOURCE_SNAPSHOT.json`. It is distinct from the later clean
Gate C1 lineage; metrics from the two lineages must not be combined.

No model was retrained and no manuscript file was edited during this public
integration. The seed-3 checkpoint and U8U8 QDQ model identities are frozen in
`RELEASE_ARTIFACTS.json`.

## Import decisions

- The source bundle manifest verified all 265 listed payload files before
  integration.
- Generated Gradle caches and build reports found in the source directory were
  excluded from version control.
- Raw phantom RF, raw-RF golden arrays, and validation RF packages were excluded
  pending separate redistribution approval.
- The processed 60-sample `(60, 7, 400)` model-input tensor and the selected
  U8U8 QDQ ONNX model are distributed as GitHub Release assets rather than Git
  blobs. Their byte sizes and SHA-256 hashes are recorded in
  `RELEASE_ARTIFACTS.json`.
- Source code, input manifests, recorded Android JSON/CSV outputs, and scripts
  for hash and concordance verification are versioned in this repository.

## Local verification performed during integration

- Python source compilation: PASS.
- Release unit tests: 4/4 PASS.
- Recorded preprocessing and 60-sample prediction concordance: PASS.
- ONNX full checker: 12/12 artifacts PASS.
- Release artifact size and SHA-256 checks: PASS.
- Android build on macOS with Android Studio JBR 21 and SDK 36:
  `:pipeline:test`, `:app:assembleBenchmark`, and
  `:benchmark:assembleReleaseAndroidTest` PASS.
- Privacy and secret scan of public files: PASS.

The Android build confirms source and dependency compilation; it is not a new
physical-device measurement. The repository preserves and verifies the
previously recorded Galaxy S8+ outputs. Live RF acquisition, BLE transport,
networking, and UI rendering remain outside the reported latency boundaries.

## Public-data boundary

Clinical RF data are restricted by privacy and IRB conditions. Raw phantom RF
is also omitted from this release pending a separate data-distribution review.
Consequently, the public materials support source inspection, artifact identity
checks, processed-input inference, Android/desktop output concordance checks,
and Android compilation. Repeating raw-RF preprocessing instrumentation or
retraining requires separately authorized data.
