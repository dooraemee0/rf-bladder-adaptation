# Bladder Volume Estimation from RF Ultrasound — Stage 2: Clinical-to-Wearable Adaptation

Code and trained checkpoints for the wearable-adaptation stage of our work on
bladder volume estimation directly from raw radio-frequency (RF) ultrasound.
A clinically pretrained per-channel RF regression model is adapted to a wearable
device using only a few discrete phantom volumes, while **preserving the
continuous volume relationship learned from clinical data** — which is what makes
the device-side predictions a genuine regression rather than memorization of the
handful of available phantom states.

This repository covers **Stage 2 (adaptation)**. Stage 1 (clinical pretraining)
lives in a separate repository:

- **Stage 1 — clinical pretraining:** https://github.com/dooraemee0/rf-bladder
- **Stage 2 — adaptation (this repo):** clinical → wearable, DER++ and variants
- **Trained checkpoints (12):** archived on Zenodo — DOI: `TODO_ZENODO_DOI`

The two stages connect through the Stage-1 checkpoint: every Stage-2 run starts
from `rf-bladder/checkpoints/best_model.pt` and evaluates clinical retention on
the held-out indices in `rf-bladder/checkpoints/test_idx.pt`.

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
├── configs/                    # hyperparameters for each reported result
├── scripts/                    # runnable .sh wrappers (paths via env vars)
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

Configs in `configs/` fully specify each result; `_base.yaml` holds the settings
that are otherwise hard-coded (AdamW, weight decay 1e-3, cosine schedule, batch
size 16, Huber loss with δ = 10, 150 epochs, seeds 1/2/3/42).

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
`ablation_A_adaptation.py`. Because of their size they are also mirrored on
Zenodo (DOI above); if the copies in this repo are hosted via a release asset or
LFS, follow the download note there.

## Data availability

Clinical RF data are **not publicly available** owing to patient privacy and IRB
restrictions (SNUH IRB No. H-2107-024-1233). The code references clinical file
and column conventions (e.g. `Z_volume_selection.csv`, `Upright_H`/`Upright_S`)
but contains no patient data. Benchtop phantom procedures are described in the
paper; the released checkpoints allow evaluation without access to the raw
clinical dataset.

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
any data are subject to separate terms stated on the Zenodo record and in the
Data availability section above.
