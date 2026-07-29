"""
lambda_sweep.py -- EWC / SI / LwF 의 정규화 강도(λ) 탐색

배경
----
본 실험의 기본 설정(EWC λ=1e3, SI λ=1, LwF α=1)에서 세 방법 모두
clinical retention 이 크게 음수였다. 이것이 "방법 자체의 한계"인지
"하이퍼파라미터가 이 문제에 안 맞았을 뿐"인지 구분하지 않으면
리뷰어가 straw-man 비교라고 지적할 수 있다. 이 스크립트는 그 구분을 위한
λ 탐색을 수행한다.

2단계 전략
----------
  1단계 (이 스크립트): seed 하나로 여러 λ 를 빠르게 스크리닝
  2단계 (stage2 모드): 1단계 최선 λ 만 나머지 seed 로 확정

전부 4 seed 로 돌리면 52회지만, 2단계로 하면 22회로 끝난다.

사용 예
-------
  # 1단계 (GPU0): EWC 전 구간 + LwF 절반
  python lambda_sweep.py --stage 1 \\
      --ckpt <ckpt> --test_idx <idx> \\
      --jobs "ewc:100,1000,10000,100000,1000000|lwf:0.1,1.0" \\
      --seed 1 --save_dir ./results_lambda_sweep

  # 2단계: 1단계 결과에서 최선 λ 를 골라 나머지 seed 로
  python lambda_sweep.py --stage 2 \\
      --ckpt <ckpt> --test_idx <idx> \\
      --jobs "ewc:10000|si:0.1|lwf:10.0" \\
      --seeds 2,3,42 --save_dir ./results_lambda_sweep
"""
import os, json, argparse, traceback
import numpy as np
import pandas as pd
import torch

from ablation_common import (
    set_seed, load_clinical_model, load_all_data, evaluate,
    count_trainable, per_volume_pred,
)
from strategies import STRATEGIES

# 각 전략에서 λ 를 담는 cfg 속성 이름
LAMBDA_ATTR = {
    "ewc": "ewc_lambda",
    "si":  "si_lambda",
    "lwf": "lwf_alpha",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_idx", default=None)
    ap.add_argument("--stage", type=int, choices=[1, 2], default=1)
    ap.add_argument("--jobs", required=True,
                    help="'ewc:100,1000|lwf:0.1,1.0' 형식. 전략:λ목록 을 | 로 구분")
    ap.add_argument("--seed", type=int, default=1, help="stage 1 에서 쓸 단일 seed")
    ap.add_argument("--seeds", default="2,3,42", help="stage 2 에서 쓸 seed 목록")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clinical_ratio", type=float, default=1.0)
    ap.add_argument("--save_dir", default="./results_lambda_sweep")
    return ap.parse_args()


def parse_jobs(spec):
    """'ewc:100,1000|lwf:0.1' -> [('ewc',100.0), ('ewc',1000.0), ('lwf',0.1)]"""
    jobs = []
    for part in spec.split("|"):
        part = part.strip()
        if not part:
            continue
        strat, lams = part.split(":")
        strat = strat.strip()
        if strat not in LAMBDA_ATTR:
            raise ValueError(f"λ 탐색 대상이 아닙니다: {strat} (가능: {list(LAMBDA_ATTR)})")
        for l in lams.split(","):
            jobs.append((strat, float(l)))
    return jobs


def load_test_idx(path):
    if not path:
        return None
    ti = torch.load(path, weights_only=False)
    return ti.tolist() if hasattr(ti, "tolist") else list(ti)


def lam_tag(lam):
    """폴더명에 쓸 안전한 λ 표기. 1e3 -> 1e+03 대신 '1000' / 0.01 -> '0.01'"""
    if lam >= 1 and float(lam).is_integer():
        return str(int(lam))
    return str(lam)


def run_one(strategy, lam, seed, args, data):
    set_seed(seed)
    model = load_clinical_model(args.ckpt)

    cfg = argparse.Namespace(**vars(args))
    cfg.seed = seed
    setattr(cfg, LAMBDA_ATTR[strategy], lam)          # <-- λ 주입
    # 다른 전략의 기본값도 채워둔다(사용되지 않지만 getattr 안전)
    for s, attr in LAMBDA_ATTR.items():
        if not hasattr(cfg, attr):
            setattr(cfg, attr, 1.0)

    print(f"\n{'='*62}\n[{strategy}]  {LAMBDA_ATTR[strategy]}={lam:g}  seed={seed}\n{'='*62}")
    model = STRATEGIES[strategy](model, data, cfg)
    n_train, n_total = count_trainable(model)

    dev = evaluate(model, data["dev_te"])
    clin = evaluate(model, data["clin_te"])

    out_dir = os.path.join(args.save_dir, strategy,
                           f"lam{lam_tag(lam)}", f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"gt": dev["gt"], "pred": dev["pred"]}).to_csv(
        os.path.join(out_dir, "predictions_device.csv"), index=False)
    pd.DataFrame({"gt": clin["gt"], "pred": clin["pred"]}).to_csv(
        os.path.join(out_dir, "predictions_clinical.csv"), index=False)

    m = dict(strategy=strategy, lam=lam, lam_attr=LAMBDA_ATTR[strategy], seed=seed,
             trainable_pct=round(100 * n_train / n_total, 3),
             device_mae=dev["mae"], device_rmse=dev["rmse"],
             device_r2=dev["r2"], device_acc50=dev["acc50"],
             clinical_mae=clin["mae"], clinical_rmse=clin["rmse"],
             clinical_r2=clin["r2"], clinical_acc50=clin["acc50"],
             tradeoff_sum=dev["r2"] + clin["r2"],
             device_per_volume=per_volume_pred(dev))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)

    print(f"  [TEST] {strategy} λ={lam:g} seed{seed}: "
          f"dev R2={dev['r2']:.3f} | clin R2={clin['r2']:.3f} | "
          f"sum={m['tradeoff_sum']:.3f}")
    return m


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    jobs = parse_jobs(args.jobs)
    seeds = [args.seed] if args.stage == 1 else [int(s) for s in args.seeds.split(",")]

    print("=== λ sweep ===")
    print(f"stage={args.stage}  seeds={seeds}")
    for s, l in jobs:
        print(f"  {s:4s} {LAMBDA_ATTR[s]}={l:g}")
    print(f"총 {len(jobs)*len(seeds)}회 학습 예정")

    data = load_all_data(load_test_idx(args.test_idx))
    print(f"device  : train {len(data['dev_tr'])} / val {len(data['dev_va'])} / test {len(data['dev_te'])}")
    print(f"clinical: train {len(data['clin_tr'])} / val {len(data['clin_va'])} / test {len(data['clin_te'])}")
    if "patient" in data["clin_tr"].columns:
        ov = set(data["clin_tr"]["patient"]) & set(data["clin_va"]["patient"])
        print(f"clinical train/val 환자 중복: {len(ov)} (0이어야 leakage 없음)")

    rows = []
    for strat, lam in jobs:
        for seed in seeds:
            try:
                rows.append(run_one(strat, lam, seed, args, data))
            except Exception as e:
                print(f"[ERROR] {strat} λ={lam} seed{seed}: {e}")
                traceback.print_exc()

    if not rows:
        print("결과 없음."); return

    flat = [{k: v for k, v in r.items() if k != "device_per_volume"} for r in rows]
    df = pd.DataFrame(flat).sort_values(["strategy", "lam", "seed"])
    out = os.path.join(args.save_dir, f"sweep_stage{args.stage}_seed{'-'.join(map(str,seeds))}.csv")
    df.to_csv(out, index=False)

    print(f"\n{'='*70}\n[요약] stage {args.stage}\n{'='*70}")
    for strat in df.strategy.unique():
        sub = df[df.strategy == strat].sort_values("lam")
        print(f"\n  {strat.upper()}  ({LAMBDA_ATTR[strat]})")
        for _, r in sub.iterrows():
            print(f"    λ={r['lam']:<10g} dev={r['device_r2']:+.3f}  "
                  f"clin={r['clinical_r2']:+.3f}  sum={r['tradeoff_sum']:+.3f}")
        best = sub.loc[sub.tradeoff_sum.idxmax()]
        print(f"    -> 최선 λ={best['lam']:g}  (sum={best['tradeoff_sum']:.3f})")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
