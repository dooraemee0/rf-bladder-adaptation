# Deployment audit

> Historical note: this audit records the state before the physical Android
> validation was completed. The unsupported preliminary `6.5 MB` and `4.9 ms`
> claims discussed below were subsequently replaced by the artifact-backed
> results and explicit measurement boundaries in `PUBLIC_ANDROID_README.md`.

## Verified model artifacts

- The paper's adopted method is standard DER++, not CFP-DER++ or R-DER++.
- Four released DER++ checkpoints exist at
  `checkpoints/final/der++/seed{1,2,3,42}/model.pt`.
- Each file is a pure state dictionary with 116 entries and loads strictly into
  `model.RFNet` with no missing or unexpected keys.
- The model contains 6,502,401 parameters. Its FP32 state occupies 26,015,092
  bytes before serialization overhead.
- The four released checkpoint hashes match the original files under
  `research/2606/DANN/0717/results_final_multiseed_valsplit/der++`.
- The paper reports mean and standard deviation across four independently
  trained seeds. It does not designate one seed as the single deployment model.
  Export therefore requires an explicit `--checkpoint`; no seed is selected
  implicitly.

## Verified deployment input

- RFNet receives three horizontal tensors followed by four sagittal tensors.
- The raw wearable text loader assumes six physical channels in the order
  `[S1, S2, S3, S4, H_left, H_right]`.
- It creates a virtual horizontal signal as `mean(S2, S3)` and forms the model
  tensor as `[H_left, H_virtual, H_right, S1, S2, S3, S4]`.
- The manuscript also calls the synthesized channel `H3` in places. Because
  that name does not uniquely communicate its tensor index, deployment metadata
  records the construction and index explicitly.
- Device preprocessing returns 220 samples per channel. The mixed-domain model
  input is `(batch, 7, 400)`.
- Padding is not zero padding. Samples 220--399 are filled independently for
  each channel using the mean of that channel's final 30 processed samples.

## Training/evaluation preprocessing used by the released checkpoints

The final Stage-2 loader performs the following wearable operations:

1. Parse six concatenated physical traces and divide amplitudes by 1700.
2. Apply piecewise exponential TGC using coefficients 0.15 and 0.4.
3. Apply a seventh-order 1.5--4.5 MHz Butterworth band-pass filter at 20 MSPS.
4. Clip amplitude to `[-0.1, 0.1]` and crop raw indices `[300:3000]`.
5. Create `mean(S2, S3)` and reorder to the seven logical channels.
6. Apply `gaussian_filter1d(..., sigma=1, axis=0)` exactly as implemented.
7. Downsample by 10 and retain the first 220 samples.
8. Standardize each logical channel independently.
9. Pad 220 to 400 with the final-30-sample channel mean.

The current manuscript differs from this implementation in the stated TGC
coefficient, filter order, clipping description, and Gaussian smoothing axis.
The free-living `infer_24h.py` also uses Gaussian smoothing on axis 1, whereas
the final training/evaluation loader uses axis 0. These discrepancies must be
resolved before raw-RF preprocessing is moved to Android. The first Android
benchmark therefore accepts already preprocessed `(7, 400)` tensors and is
strictly a model-only benchmark.

The full call-path audit, fixed-checkpoint axis sensitivity, and 24-hour output
comparison are recorded in `PREPROCESSING_RECONCILIATION.md`.

## Audit of the published 6.5 MB and 4.9 ms claims

The only supporting code found is
`research/2606/DANN/0702/eval_table4.py`:

- `6.5 MB` is calculated as `6,502,401 parameters x 1 byte`; no quantized model
  is written and its actual serialized size is not measured.
- The script uses PyTorch dynamic quantization. Conv1d static activation
  quantization and held-out accuracy are not validated.
- Latency is measured on an unspecified desktop CPU with one PyTorch thread,
  one random input, ten warm-up calls, and 200 repeated calls.
- The summary prints FP32 latency even when the dynamic-quantization branch is
  enabled. There is no Android device, ONNX Runtime, device metadata, latency
  distribution, or measurement log tied to the manuscript value.

Accordingly, 6.5 MB and 4.9 ms are unsupported as claims about an actual static
int8 Android deployment. They must be replaced only after this pipeline creates
the model and records physical-device results.

The completed four-seed export selected U8U8 QDQ per-channel using aggregate
wearable-validation MAE before loading the held-out test partition. The selected
models serialize to 6.652 MB and their test results are reported in
`RESULTS_ALL_SEEDS.md`. This does not supply the missing Android measurement.
