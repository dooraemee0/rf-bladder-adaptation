# Physical smartphone raw-RF-to-volume validation

**Status: complete for the local raw-file path.** A frozen Android
implementation performed six-channel RF preprocessing and static U8U8 ONNX
inference without a server on a physical Samsung Galaxy S8+. BLE acquisition
and transfer remain outside the implemented and measured path.

## Device and runtime

| Item | Value |
|---|---|
| Device | Samsung Galaxy S8+ (SM-G955N) |
| Hardware / OS | Exynos 8895; Android 9 (API 28); arm64-v8a |
| Runtime | ONNX Runtime Android 1.23.2, CPU Execution Provider |
| Model | DER++ RFNet seed 3, static U8U8 QDQ |
| Model size | 6,651,648 bytes (6.344 MiB) |
| Raw input | Six physical RF channels, 5,120 float32 samples per channel |
| Model input / output | float32 `(1,7,400)` / scalar volume, batch 1 |
| ORT configuration | intra-op 1; inter-op 1; sequential execution |
| Benchmark | Jetpack Microbenchmark 1.4.1, release/non-debuggable |
| Performance mode | Sustained-performance mode enabled |

## Numerical validation chain

The production preprocessing implementation was evaluated first on three
predeclared wearable validation samples, one at each reference volume. The
maximum Android-Python difference in the final `(7,400)` tensor was
`2.7716e-6`, while all three U8U8 predictions were identical. The Android
implementation was then frozen before held-out RF files were packaged.

The one-time held-out evaluation included 60 wearable phantom measurements.
There were no missing, duplicate, or non-finite samples. The maximum tensor
difference was `2.6599e-6`, and the Android and desktop U8U8 predictions were
identical for every sample.

| Held-out metric | Android raw-RF result |
|---|---:|
| Samples | 60 |
| MAE | 43.2399 mL |
| RMSE | 56.8123 mL |
| R2 | 0.6942241 |
| Within 50 mL | 65.0% |
| 200 mL alarm accuracy | 90.0% |
| Alarm precision / recall / F1 | 0.850 / 0.850 / 0.850 |
| TN / FP / FN / TP | 37 / 3 / 3 / 17 |

These results reproduce the earlier model-only Android performance exactly,
showing that the smartphone preprocessing implementation did not change the
frozen deployment output. Because the held-out set contains only 50, 150, and
300 mL phantom states, these results must not be presented as independent
validation of continuous in-vivo volume regression.

## Standardized warm latency

The preprocessing-only timed block starts with six in-memory `(5120,)` RF
arrays and ends when the flat `(7,400)` tensor is complete. The combined timed
block includes preprocessing, ONNX tensor creation, `OrtSession.run`, scalar
output retrieval, and tensor/result closure. Asset reads, RF file parsing,
session creation, BLE, network communication, and UI work are excluded.

Each of three independent sessions recorded 50 values per scope after Jetpack
warm-up. All sessions used battery level 43%, started at or below 28.8 C,
ended at 28.8 C, and reported zero thermal-throttle sleep. Percentiles use the
nearest-rank definition.

### RF preprocessing

| Session | Temperature (start/end) | Warm-up / repeat | Mean | Median | P90 | P95 | Min-max | CV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 28.6 / 28.8 C | 548 / 6 | 14.1378 | 14.5012 | 15.5096 | 15.8867 | 7.1160-16.1843 | 12.275% |
| 2 | 28.8 / 28.8 C | 544 / 2 | 14.5914 | 14.2771 | 17.2388 | 17.6696 | 12.6628-18.5993 | 11.673% |
| 3 | 28.8 / 28.8 C | 539 / 7 | 14.3148 | 14.2589 | 14.9631 | 15.2365 | 13.5008-15.6715 | 3.282% |

Across 150 values, preprocessing latency had a pooled median of **14.3325
ms**, P90 of **15.6773 ms**, and P95 of **16.9837 ms**. The pooled mean was
14.3480 ms, sample SD was 1.4325 ms, and CV was 9.984%. Despite within-session
outliers, the three session medians were stable at **14.3457 +/- 0.1349 ms**.

### Raw RF to scalar volume

| Session | Temperature (start/end) | Warm-up / repeat | Mean | Median | P90 | P95 | Min-max | CV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 28.6 / 28.8 C | 206 / 2 | 38.6342 | 38.6705 | 39.6348 | 39.7703 | 37.4536-40.4231 | 2.521% |
| 2 | 28.8 / 28.8 C | 207 / 2 | 38.7007 | 39.2019 | 39.5501 | 39.6030 | 37.2382-44.2699 | 3.054% |
| 3 | 28.8 / 28.8 C | 205 / 2 | 39.0183 | 39.6016 | 40.0046 | 40.2300 | 37.8884-40.4090 | 2.395% |

Across 150 values, combined raw-RF-to-volume latency had a pooled median of
**39.2305 ms**, P90 of **39.8459 ms**, and P95 of **40.0613 ms**. The pooled
mean was 38.7844 ms, sample SD was 1.0425 ms, and CV was 2.688%. The three
session medians were **39.1580 +/- 0.4671 ms**.

The prior, separately measured model-only `OrtSession.run` reference had a
pooled median of 24.2771 ms and P90 of 24.6382 ms. It is a consistency
reference rather than an exactly subtractable component: it used pre-created
tensors in a separate benchmark campaign, and its recorded ART compilation
mode was `speed` rather than the `speed-profile` mode recorded here.

## Initialization and memory

Ten fresh `OrtSession` initializations were measured repeatedly within one
instrumentation process. The total median was **136.829 ms**, with model asset
read and session creation medians of 72.490 and 64.347 ms, respectively. The
first iteration included initial ORT environment acquisition and produced a
total of 290.770 ms. This experiment is not a true process cold-start
measurement and must be described as repeated in-process session
initialization.

Memory was captured as stage-specific point observations in a separate
instrumentation process. Baseline total PSS was 14,736 kB (14.39 MiB), and the
maximum observed total PSS was **61,776 kB (60.33 MiB)** after raw RF buffers
were loaded, an increase of 47,040 kB (45.94 MiB) over baseline. PSS after 100
inferences was 48,269 kB (47.14 MiB). These values were not continuously
sampled and must not be called peak memory.

## Local smartphone demonstration

The application processed validation sample `150_30.txt` entirely on the
phone, returned 171.715 mL, and correctly produced a normal state below the
200 mL threshold. Its absolute volume error was 21.715 mL. The Activity timing
was diagnostic only; standardized latency is taken from Jetpack Benchmark.

The returned automated ADB screenshot captured the phone home screen rather
than the result view and is excluded from publication figures. A separately
captured result screenshot should be preserved as a source file before it is
used as photographic or screenshot evidence.

## Interpretation boundary

The completed claim is a **server-free, smartphone-resident raw-RF-to-volume
path** using RF arrays already present in phone memory. It is not yet a complete
sensor-to-display wearable latency measurement because the firmware packet
format lacks the metadata required for an unambiguous Android BLE receiver,
and BLE acquisition/transfer was not implemented or measured.

## Reproducible sources

- analysis script: `analyze_and_plot_smartphone_end_to_end.py`
- recomputed metrics: `analysis/smartphone_end_to_end_metrics.json`
- pooled raw latency table: `analysis/latency_runs.csv`
- returned archive SHA-256:
  `e69e428279a975391727d2e704998479d737f50d96e5b8f1a2a80c19a44af4be`
- physical results: `raw/final_windows_results/results/`
- preprocessing parity: `raw/preprocessing_parity/`
