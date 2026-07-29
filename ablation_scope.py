"""
ablation_scope.py -- Supplementary Table S2 재실행
(adaptation scope ablation, validation-selected + multi-seed 프로토콜)

기존 Supp Table S2 는 옛 프로토콜(test-selected, single-seed)에서, 그것도
rehearsal 을 채택 방법으로 두고 만든 표였다. 채택 방법이 DER++ 로 바뀌었으므로
동일한 DER++ 목적함수 아래에서 "어느 파라미터를 풀 것인가"만 바꿔가며 다시 측정한다.

비교하는 scope
--------------
  head_only : 인코더 전체 freeze, regressor head 만 학습
  last_conv : 각 채널 인코더의 마지막 conv + head  (본문 채택 = 29.3%)
  all_conv  : 모든 conv + head

프로토콜은 본 실험과 완전히 동일하다.
  * 목적함수: DER++  (device Huber + alpha*dark experience + beta*clinical replay)
  * checkpoint 선택: validation(dev_va R2 + clin_va R2) 만 사용
  * test 는 선택 후 1회만 평가
  * seed 1,2,3,42 반복, mean +/- SD 보고

사용 예
-------
  python ablation_scope.py \
      --ckpt /path/best_model.pt --test_idx /path/test_idx.pt \
      --scopes head_only,last_conv,all_conv \
      --seeds 1,2,3,42 --epochs 150 \
      --save_dir ./results_scope_ablation

산출물
------
  save_dir/summary_scope.csv        seed별 전체 지표
  save_dir/summary_scope_mean.csv   scope별 mean/SD
  save_dir/table_s2.tex             논문 Supp Table S2 붙여넣기용
  save_dir/<scope>/seed<n>/         모델·예측·metrics.json
"""
import os, json, copy, argparse, traceback
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from ablation_common import (
    DEV, set_seed, load_clinical_model, load_all_data, evaluate,
    set_unfreeze_scope, count_trainable, per_volume_pred,
    evaluate_and_select_best, restore_best, new_best, make_loader,
)
from strategies import _forward, _precompute_old_preds, _make_opt, _should_eval

SCOPES = ["head_only", "last_conv", "all_conv"]
SCOPE_LABEL = {
    "head_only": "Head only (encoders frozen)",
    "last_conv": "Last conv + head (adopted)",
    "all_conv":  "All conv + head",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="clinical-pretrained 체크포인트")
    ap.add_argument("--test_idx", default=None)
    ap.add_argument("--scopes", default="head_only,last_conv,all_conv")
    ap.add_argument("--seeds", default="1,2,3,42")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--der_alpha", type=float, default=0.5)
    ap.add_argument("--der_beta", type=float, default=0.5)
    ap.add_argument("--save_dir", default="./results_scope_ablation")
    return ap.parse_args()


def load_test_idx(path):
    if not path:
        return None
    ti = torch.load(path, weights_only=False)
    return ti.tolist() if hasattr(ti, "tolist") else list(ti)


def train_derpp_with_scope(model, data, cfg, scope):
    """strategies.train_der_plus_plus 와 동일하되 adaptation scope 만 인자로 받는다.

    인코더를 .eval() 로 유지하는 동작까지 원본과 동일하게 맞춰,
    scope 이외의 조건이 달라지지 않도록 한다.
    """
    old_model = copy.deepcopy(model).to(DEV).eval()
    for p in old_model.parameters():
        p.requires_grad = False

    set_unfreeze_scope(model, scope, reinit_head=True)   # <-- 유일한 차이점

    alpha = getattr(cfg, "der_alpha", 0.5)
    beta = getattr(cfg, "der_beta", 0.5)
    crit = nn.HuberLoss(delta=10.0)
    mse = nn.MSELoss()

    dev_loader = make_loader(data["dev_tr"], shuffle=True, pad=True)
    clin_df = data["clin_tr"].reset_index(drop=True)
    z_old_all = _precompute_old_preds(old_model, clin_df)

    opt = _make_opt(model, cfg.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    best = new_best()

    for ep in range(1, cfg.epochs + 1):
        model.train()
        # 원본과 동일: BatchNorm 통계는 갱신하지 않는다
        model.h_encoders.eval()
        model.s_encoders.eval()

        clin_loader = make_loader(clin_df, shuffle=False, pad=True)  # 순서 고정
        clin_iter = iter(clin_loader)
        clin_start = 0
        for batch in dev_loader:
            pred, y = _forward(model, batch)
            loss = crit(pred, y)
            try:
                cbatch = next(clin_iter); bs = len(cbatch["volume_gt"])
            except StopIteration:
                clin_iter = iter(clin_loader); clin_start = 0
                cbatch = next(clin_iter); bs = len(cbatch["volume_gt"])
            idx = list(range(clin_start, clin_start + bs)); clin_start += bs
            z_old = z_old_all[idx].to(DEV) if len(z_old_all) else None

            cpred, cy = _forward(model, cbatch)
            if z_old is not None and len(z_old) == len(cpred):
                loss = loss + alpha * mse(cpred, z_old)
            loss = loss + beta * crit(cpred, cy)

            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        if _should_eval(ep, cfg.epochs):
            evaluate_and_select_best(model, data, best, ep, f"scope:{scope}", mode="sum")

    return restore_best(model, best, f"scope:{scope}")


def run_one(scope, seed, args, data):
    set_seed(seed)
    print(f"[seed] requested={seed}")
    np_probe = float(np.random.rand()); torch_probe = float(torch.rand(1).item())
    print(f"[seed probe] numpy={np_probe:.8f}, torch={torch_probe:.8f}")
    set_seed(seed)

    model = load_clinical_model(args.ckpt)
    cfg = argparse.Namespace(**vars(args)); cfg.seed = seed

    model = train_derpp_with_scope(model, data, cfg, scope)
    n_train, n_total = count_trainable(model)

    dev_res = evaluate(model, data["dev_te"])
    clin_res = evaluate(model, data["clin_te"])

    out_dir = os.path.join(args.save_dir, scope, f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    pd.DataFrame({"gt": dev_res["gt"], "pred": dev_res["pred"]}).to_csv(
        os.path.join(out_dir, "predictions_device.csv"), index=False)
    pd.DataFrame({"gt": clin_res["gt"], "pred": clin_res["pred"]}).to_csv(
        os.path.join(out_dir, "predictions_clinical.csv"), index=False)

    m = dict(
        scope=scope, seed=seed, scope_label=SCOPE_LABEL[scope],
        trainable_params=int(n_train), total_params=int(n_total),
        trainable_pct=round(100 * n_train / n_total, 3),
        device_mae=dev_res["mae"], device_rmse=dev_res["rmse"],
        device_r2=dev_res["r2"], device_acc50=dev_res["acc50"],
        clinical_mae=clin_res["mae"], clinical_rmse=clin_res["rmse"],
        clinical_r2=clin_res["r2"], clinical_acc50=clin_res["acc50"],
        tradeoff_sum=dev_res["r2"] + clin_res["r2"],
        device_per_volume=per_volume_pred(dev_res),
    )
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)

    print(f"  [TEST] {scope} seed{seed}: dev R2={dev_res['r2']:.3f} "
          f"(MAE {dev_res['mae']:.1f}) | clin R2={clin_res['r2']:.3f} "
          f"(MAE {clin_res['mae']:.1f}) | sum={m['tradeoff_sum']:.3f} "
          f"| trainable {m['trainable_pct']:.2f}%")
    return m


def write_latex(df_mean, path):
    """Supp Table S2 붙여넣기용. mean +/- SD, 채택 행 굵게."""
    def f(row, col, d=3):
        return f"${row[col]:.{d}f} \\pm {row[col + '_sd']:.{d}f}$"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Adaptation-scope ablation under the adopted DER++",
        r"objective.} Only the set of updated parameters differs between rows; the",
        r"objective, optimizer, schedule, epoch budget, data partitions, and",
        r"checkpoint-selection criterion are identical to the main experiments.",
        r"Values are mean $\pm$ standard deviation across seeds $1$, $2$, $3$, and",
        r"$42$, evaluated once on the held-out test sets after validation-based",
        r"checkpoint selection. Updating only the last convolutional layer of each",
        r"channel encoder together with the regression head is adopted because it",
        r"provides the best device--clinical trade-off at a modest parameter cost.",
        r"Note that the \texttt{all conv} configuration unfreezes every encoder",
        r"parameter, including normalization and projection layers, and therefore",
        r"corresponds to full-model fine-tuning.}",
        r"\label{stab:unfreeze}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Stage-2 adaptation scope} & \textbf{Trainable (\%)} & "
        r"\textbf{Device $R^2$} & \textbf{Clinical $R^2$} & \textbf{$R^2$ sum} \\",
        r"\midrule",
    ]
    for scope in SCOPES:
        if scope not in df_mean.index:
            continue
        r = df_mean.loc[scope]
        label = SCOPE_LABEL[scope]
        if scope == "last_conv":
            label = r"\textbf{" + label + "}"
        lines.append(
            f"{label} & {r['trainable_pct']:.2f} & "
            f"{f(r,'device_r2')} & {f(r,'clinical_r2')} & {f(r,'tradeoff_sum')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as fp:
        fp.write("\n".join(lines))


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    scopes = [s.strip() for s in args.scopes.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]
    for s in scopes:
        if s not in SCOPES:
            raise ValueError(f"unknown scope '{s}'. available: {SCOPES}")

    print("=== Supp Table S2: adaptation-scope ablation (DER++, validation-selected) ===")
    print(f"scopes={scopes}  seeds={seeds}  epochs={args.epochs}")

    data = load_all_data(load_test_idx(args.test_idx))
    print(f"device  : train {len(data['dev_tr'])} / val {len(data['dev_va'])} / test {len(data['dev_te'])}")
    print(f"clinical: train {len(data['clin_tr'])} / val {len(data['clin_va'])} / test {len(data['clin_te'])}")
    if "patient" in data["clin_tr"].columns:
        ov = set(data["clin_tr"]["patient"]) & set(data["clin_va"]["patient"])
        print(f"clinical train/val 환자 중복: {len(ov)} (0이어야 leakage 없음)")

    rows = []
    for scope in scopes:
        for seed in seeds:
            print(f"\n{'='*60}\n[{scope}] seed={seed}\n{'='*60}")
            try:
                rows.append(run_one(scope, seed, args, data))
            except Exception as e:
                print(f"[ERROR] {scope} seed{seed}: {e}")
                traceback.print_exc()

    if not rows:
        print("결과 없음."); return

    flat = [{k: v for k, v in r.items() if k != "device_per_volume"} for r in rows]
    df = pd.DataFrame(flat)
    df.to_csv(os.path.join(args.save_dir, "summary_scope.csv"), index=False)

    num = [c for c in df.columns if c not in ("scope", "seed")
           and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby("scope")
    mean = g[num].mean()
    sd = g[["device_r2", "clinical_r2", "tradeoff_sum", "device_mae", "clinical_mae"]].std(ddof=1)
    sd.columns = [c + "_sd" for c in sd.columns]
    dfm = mean.join(sd)
    dfm["n_seeds"] = g.size()
    dfm.to_csv(os.path.join(args.save_dir, "summary_scope_mean.csv"))
    write_latex(dfm, os.path.join(args.save_dir, "table_s2.tex"))

    print(f"\n{'='*74}\n[요약] scope별 mean ± SD (TEST set, validation-selected)\n{'='*74}")
    for scope in SCOPES:
        if scope not in dfm.index:
            continue
        r = dfm.loc[scope]
        star = "  <-- adopted" if scope == "last_conv" else ""
        print(f"  {scope:10s} train={r['trainable_pct']:5.2f}%  "
              f"dev={r['device_r2']:.3f}±{r['device_r2_sd']:.3f}  "
              f"clin={r['clinical_r2']:.3f}±{r['clinical_r2_sd']:.3f}  "
              f"sum={r['tradeoff_sum']:.3f}±{r['tradeoff_sum_sd']:.3f}{star}")
    print(f"\n[saved] {args.save_dir}/table_s2.tex  <- 논문에 붙여넣기")


if __name__ == "__main__":
    main()
