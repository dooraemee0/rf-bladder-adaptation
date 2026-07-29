"""
ablation_A_adaptation.py -- 통합 runner (validation-selected checkpoint, test-only 최종평가)

프로토콜 (논문용 최종본)
------------------------
  Train set       -> 학습
  Validation set  -> epoch별 checkpoint 선택 (dev_va R2 + clin_va R2)
  Test set        -> 선택된 checkpoint로 최종 1회 평가 (dev_te / clin_te)

전략은 strategies.py 안에서 validation으로만 best를 고르고, 이 runner는
학습 종료 후 test set을 딱 한 번 평가한다. test는 model selection에 절대
쓰이지 않는다.

실행 예 (통합 all-strategy):
  python ablation_A_adaptation.py \
      --ckpt /path/best_model.pt --test_idx /path/test_idx.pt \
      --strategies der++,cfp_derpp,rehearsal,device_only,joint,ewc,si,lwf,coral,mmd,lora_conv,r_derpp \
      --seeds 1,2,3,42 --epochs 150 --save_dir ./results_final_multiseed_valsplit

결과:
  save_dir/<strategy>/seed<seed>/model.pt
  save_dir/<strategy>/seed<seed>/metrics.json
  save_dir/<strategy>/seed<seed>/predictions_{device,clinical}.csv
  save_dir/summary_ablationA.csv       (전 전략 x seed)
  save_dir/summary_ablationA_mean.csv  (전략별 seed 평균 ± 표준편차)
  save_dir/summary_ablationA.tex       (논문용 LaTeX)
"""
import os, json, argparse, traceback
import numpy as np
import pandas as pd
import torch

from ablation_common import (
    SEED, set_seed, load_clinical_model, load_all_data, evaluate,
    bootstrap_metrics, per_volume_pred, count_trainable,
)
from strategies import STRATEGIES


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="clinical-pretrained 체크포인트")
    ap.add_argument("--test_idx", default=None, help="clinical held-out test 인덱스 .pt")
    ap.add_argument("--strategies",
                    default="der++,cfp_derpp,rehearsal,device_only,joint,ewc,si,lwf,coral,mmd,lora_conv")
    ap.add_argument("--seeds", default="1,2,3,42")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clinical_ratio", type=float, default=1.0)
    ap.add_argument("--n_boot", type=int, default=2000)
    # 전략별 하이퍼파라미터
    ap.add_argument("--ewc_lambda", type=float, default=1e3)
    ap.add_argument("--si_lambda", type=float, default=1.0)
    ap.add_argument("--lwf_alpha", type=float, default=1.0)
    ap.add_argument("--align_beta", type=float, default=1.0)
    ap.add_argument("--lora_rank", type=int, default=4)
    ap.add_argument("--lora_alpha", type=float, default=8.0)
    ap.add_argument("--der_alpha", type=float, default=0.5)
    ap.add_argument("--der_beta", type=float, default=0.5)
    ap.add_argument("--rank_lambda", type=float, default=0.05)
    ap.add_argument("--relation_lambda", type=float, default=0.05)
    ap.add_argument("--pair_min_gap", type=float, default=50.0)
    ap.add_argument("--rank_margin", type=float, default=10.0)
    # CFP-DER++ 하이퍼파라미터
    ap.add_argument("--feature_lambda", type=float, default=0.05)
    ap.add_argument("--feature_loss_mode", choices=["cosine", "smooth_l1", "hybrid"],
                    default="cosine")
    ap.add_argument("--feature_schedule",
                    choices=["constant", "linear_decay", "cosine_decay",
                             "linear_warmup", "cosine_warmup", "early_half", "late_half"],
                    default="constant")
    ap.add_argument("--feature_min_ratio", type=float, default=0.1)
    ap.add_argument("--feature_grad_clip", type=float, default=5.0)
    ap.add_argument("--feature_channel_weights", default="")
    ap.add_argument("--save_dir", default="./results_final_multiseed_valsplit")
    return ap.parse_args()


def load_test_idx(path):
    if not path: return None
    ti = torch.load(path, weights_only=False)
    return ti.tolist() if hasattr(ti, 'tolist') else list(ti)


def save_predictions(res, path):
    pd.DataFrame({"gt": res['gt'], "pred": res['pred']}).to_csv(path, index=False)


def run_one(strategy, seed, args, data):
    """전략 1개 x seed 1개: 학습(val 선택) -> test 최종평가 -> 저장."""
    # ---- seed 설정 + probe (RNG-safe evaluate 덕분에 이후 학습에 영향 없음) ----
    set_seed(seed)
    print(f"[seed] requested={seed}")
    # probe: 서로 다른 seed가 실제로 다른 난수열을 쓰는지 확인용
    np_probe = float(np.random.rand())
    torch_probe = float(torch.rand(1).item())
    print(f"[seed probe] numpy={np_probe:.8f}, torch={torch_probe:.8f}")
    # probe가 RNG를 소비했으므로 학습 재현성을 위해 원래 seed로 복원
    set_seed(seed)

    model = load_clinical_model(args.ckpt)

    # 학습 전 성능 (참고; test로 평가하되 selection엔 미사용)
    dev_before = evaluate(model, data['dev_te'])
    clin_before = evaluate(model, data['clin_te'])

    cfg = argparse.Namespace(**vars(args)); cfg.seed = seed

    train_fn = STRATEGIES[strategy]
    model = train_fn(model, data, cfg)   # 내부에서 validation으로 best 선택 + 복원
    n_train, n_total = count_trainable(model)

    # ---- 최종 test 평가 (딱 한 번) ----
    dev_res = evaluate(model, data['dev_te'])
    clin_res = evaluate(model, data['clin_te'])
    dev_bs = bootstrap_metrics(dev_res, n_boot=args.n_boot, seed=seed)
    clin_bs = bootstrap_metrics(clin_res, n_boot=args.n_boot, seed=seed)

    out_dir = os.path.join(args.save_dir, strategy, f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    save_predictions(dev_res, os.path.join(out_dir, "predictions_device.csv"))
    save_predictions(clin_res, os.path.join(out_dir, "predictions_clinical.csv"))

    metrics = dict(
        strategy=strategy, seed=seed,
        seed_probe_numpy=np_probe, seed_probe_torch=torch_probe,
        trainable_params=int(n_train), total_params=int(n_total),
        trainable_pct=round(100*n_train/n_total, 3),
        device_mae=dev_res['mae'], device_rmse=dev_res['rmse'],
        device_r2=dev_res['r2'], device_acc50=dev_res['acc50'],
        device_mae_std=dev_bs['mae_std'], device_r2_std=dev_bs['r2_std'],
        clinical_mae=clin_res['mae'], clinical_rmse=clin_res['rmse'],
        clinical_r2=clin_res['r2'], clinical_acc50=clin_res['acc50'],
        clinical_mae_std=clin_bs['mae_std'], clinical_r2_std=clin_bs['r2_std'],
        device_r2_before=dev_before['r2'], clinical_r2_before=clin_before['r2'],
        tradeoff_sum=dev_res['r2'] + clin_res['r2'],
        device_per_volume=per_volume_pred(dev_res),
    )
    # CFP-DER++ 채널별 feature distance (있을 때만)
    if hasattr(model, "_cfp_channel_feature"):
        metrics["cfp_channel_feature"] = model._cfp_channel_feature

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  [저장] {out_dir}")
    print(f"  [TEST] {strategy} seed{seed}: "
          f"dev R2={dev_res['r2']:.3f} (MAE {dev_res['mae']:.1f}) | "
          f"clin R2={clin_res['r2']:.3f} (MAE {clin_res['mae']:.1f}) | "
          f"sum={dev_res['r2']+clin_res['r2']:.3f}")
    return metrics


def to_latex(df_mean, path):
    cols = ["strategy", "device_r2", "device_mae", "device_acc50",
            "clinical_r2", "clinical_mae", "tradeoff_sum", "trainable_pct"]
    d = df_mean[cols].copy()
    header = (" & ".join(["Strategy", "Dev $R^2$", "Dev MAE", "Dev $\\pm$50",
                          "Clin $R^2$", "Clin MAE", "$R^2$ sum", "Train\\%"]) + " \\\\")
    lines = ["\\begin{tabular}{lrrrrrrr}", "\\toprule", header, "\\midrule"]
    for _, r in d.iterrows():
        name = str(r['strategy']).replace("_", "\\_")
        lines.append(f"{name} & {r['device_r2']:.3f} & {r['device_mae']:.1f} & "
                     f"{r['device_acc50']:.1f} & {r['clinical_r2']:.3f} & "
                     f"{r['clinical_mae']:.1f} & {r['tradeoff_sum']:.3f} & "
                     f"{r['trainable_pct']:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]
    for s in strategies:
        if s not in STRATEGIES:
            raise ValueError(f"unknown strategy '{s}'. available: {list(STRATEGIES)}")

    print("=== Ablation A: 도메인적응 전략 비교 (validation-selected) ===")
    print(f"strategies: {strategies}")
    print(f"seeds: {seeds}  epochs: {args.epochs}")

    test_idx = load_test_idx(args.test_idx)
    data = load_all_data(test_idx)
    print(f"device  : train {len(data['dev_tr'])} / val {len(data['dev_va'])} / test {len(data['dev_te'])}")
    print(f"clinical: train {len(data['clin_tr'])} / val {len(data['clin_va'])} / test {len(data['clin_te'])}")
    if 'patient' in data['clin_tr'].columns:
        ov = set(data['clin_tr']['patient']) & set(data['clin_va']['patient'])
        print(f"clinical train/val 환자 중복: {len(ov)} (0이어야 leakage 없음)")

    all_rows = []
    for strat in strategies:
        for seed in seeds:
            print(f"\n{'='*60}\n[{strat}] seed={seed}\n{'='*60}")
            try:
                m = run_one(strat, seed, args, data)
                all_rows.append(m)
            except Exception as e:
                print(f"[ERROR] {strat} seed{seed}: {e}")
                traceback.print_exc()

    if not all_rows:
        print("결과 없음."); return

    # 전 seed flat 저장
    flat = []
    for m in all_rows:
        row = {k: v for k, v in m.items()
               if k not in ("device_per_volume", "cfp_channel_feature")}
        flat.append(row)
    df = pd.DataFrame(flat)
    df.to_csv(os.path.join(args.save_dir, "summary_ablationA.csv"), index=False)

    # seed 평균 + 표준편차 요약
    num_cols = [c for c in df.columns
                if c not in ("strategy", "seed") and pd.api.types.is_numeric_dtype(df[c])]
    grp = df.groupby("strategy")
    df_mean = grp[num_cols].mean().reset_index()
    df_std = grp[["device_r2", "clinical_r2", "tradeoff_sum",
                  "device_mae", "clinical_mae"]].std().reset_index()
    df_std = df_std.rename(columns={
        "device_r2": "device_r2_seedstd", "clinical_r2": "clinical_r2_seedstd",
        "tradeoff_sum": "tradeoff_sum_seedstd", "device_mae": "device_mae_seedstd",
        "clinical_mae": "clinical_mae_seedstd"})
    df_mean = df_mean.merge(df_std, on="strategy", how="left")
    df_mean = df_mean.sort_values("tradeoff_sum", ascending=False).reset_index(drop=True)
    df_mean.to_csv(os.path.join(args.save_dir, "summary_ablationA_mean.csv"), index=False)
    to_latex(df_mean, os.path.join(args.save_dir, "summary_ablationA.tex"))

    print(f"\n{'='*70}\n[요약] seed 평균 ± seed표준편차 (R2합 내림차순, TEST set)\n{'='*70}")
    for _, r in df_mean.iterrows():
        print(f"  {r['strategy']:<12} "
              f"dev={r['device_r2']:.3f}±{r.get('device_r2_seedstd', float('nan')):.3f} "
              f"clin={r['clinical_r2']:.3f}±{r.get('clinical_r2_seedstd', float('nan')):.3f} "
              f"sum={r['tradeoff_sum']:.3f}±{r.get('tradeoff_sum_seedstd', float('nan')):.3f} "
              f"train%={r['trainable_pct']:.2f}")
    print(f"\n[saved] {args.save_dir}/summary_ablationA.csv / _mean.csv / .tex")


if __name__ == "__main__":
    main()
