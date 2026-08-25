# Golden reference report

## Scope

Golden outputs were generated only from the frozen wearable validation
partition. Held-out test raw RF was not loaded or used to choose an algorithm,
coefficient, implementation detail, or tolerance.

## Reproduction command

Run from the project root:

```bash
<USER_HOME>/anaconda3/envs/torch310/bin/python \
  deployment/smartphone_end_to_end/generate_golden_reference.py
```

The generator refuses to overwrite a nonempty output directory unless
`--force` is provided. It verifies the frozen FP32 and U8U8 model hashes before
inference.

## Samples

| Validation sample | Target | FP32 ONNX | U8U8 ONNX |
|---|---:|---:|---:|
| `50_30.txt` | 50 mL | 127.20201111 mL | 124.01663971 mL |
| `150_30.txt` | 150 mL | 179.62091064 mL | 171.71534729 mL |
| `300_30.txt` | 300 mL | 148.83082581 mL | 143.09611511 mL |

These samples were predeclared as the first validation member of each phantom
volume. Their predictions are parity targets, not a selected performance
subset.

## Saved stages

Every sample directory contains little-endian float32 binary data and a JSON
manifest for:

1. parsed `(6,5120)` physical RF
2. amplitude-scaled `(6,5120)` RF
3. TGC output `(6,5120)`
4. band-pass output `(6,5120)`
5. clipped and cropped `(6,2700)` RF
6. logical `(7,2700)` RF
7. Gaussian-processed `(7,2700)` RF
8. downsampled `(7,220)` RF
9. standardized `(7,220)` RF
10. padded `(7,400)` model input

For each stage, the manifest records shape, dtype, value count, minimum,
maximum, mean, population standard deviation, non-finite count, and SHA-256.
`filter_coefficients.json` records the frozen Butterworth coefficients,
`filtfilt` settings, and Gaussian settings.

## Verification

- Independent stage implementation versus authoritative deployment loader:
  48/48 validation samples bitwise equal.
- Maximum final-tensor difference: 0.0.
- Golden archive checksum verification: all entries in
  `golden_reference/SHA256SUMS.txt` passed.
- Recomputed predictions differ from the existing human-readable validation
  CSV only by its decimal serialization, with maximum differences below
  `5e-6` mL for the three golden samples.

## Runtime

- Python 3.10.16
- NumPy 2.1.2
- SciPy 1.15.1
- ONNX Runtime 1.23.2
- CPU execution provider, sequential execution, one intra-op and one inter-op thread
