# DER++ release reproducibility

## Lineage boundary

This directory records the historical Stage 2 internal-feasibility lineage based on repository commit `d28c8040fa8d360cb47bc5b60453fe52b6b3709e` plus the tracked preprocessing-sensitivity and validation-selection changes listed in `SOURCE_SNAPSHOT.json`. It is not the later clean Gate C1 participant-disjoint lineage. Metrics from those lineages must not be combined.

The Stage 2 runs initialize from the checkpoint released by the Stage 1 repository and use its recorded `test_idx.pt`. Clinical RF data are excluded from this repository. Re-executing clinical training or clinical-retention evaluation therefore requires separately authorized access to the clinical dataset.

## Historical DER++ results

`DERPP_HISTORICAL_SEED_METRICS.csv` contains the four frozen seed results copied from the final historical run records. `DERPP_HISTORICAL_AGGREGATE.csv` is a deterministic mean and sample-standard-deviation aggregation of those records. No model was retrained and no new performance result was generated while building this release.

The representative physical-device artifact was selected using wearable-validation R2 plus clinical-retention-validation R2, with no held-out test metric used. Seed 3 was selected. The checkpoint and ONNX hashes are recorded in `RELEASE_ARTIFACTS.json`.

## Stage 2 commands

After downloading and verifying the checkpoint release described in the root README:

```bash
# Requires authorized clinical and wearable source data configured as documented
source .env
bash scripts/run_final_gpu0.sh
bash scripts/run_final_gpu1.sh
python aggregate_summary.py
```

Checkpoint selection is validation-controlled in `ablation_common.py` and `strategies.py`. Quantization-format and representative-seed selection are implemented in `deployment/run_all_seeds_deployment.py` and `deployment/select_deployment_model.py`.
