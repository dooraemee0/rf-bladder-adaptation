"""
eval_invitro_final.py -- 채택 모델(DER++)로 in vitro / cross-domain 수치 재생성

논문의 Fig.4 및 Table 4(cross-domain)에 필요한 값을 저장된 seed별 checkpoint에서
다시 계산한다. 학습은 하지 않고 평가만 하므로 빠르다.

생성되는 값
-----------
  * device / clinical test 의 MAE, RMSE, R2, ±50 mL accuracy  (seed별 + mean±SD)
  * device 볼륨별(50/150/300 mL) 평균 예측
  * >=200 mL alarm 성능: accuracy, precision, recall, F1  (seed별 + mean±SD)
  * 부트스트랩 요약(단일 대표 seed 기준, 논문 Table 4의 clinical-only 행과 형식 일치)
  * Fig.4a(예측 산점도) / Fig.4b(혼동행렬)용 예측 CSV

사용 예
-------
  python eval_invitro_final.py \
      --results_dir ./results_final_multiseed_valsplit \
      --strategy "der++" \
      --seeds 1,2,3,42 \
      --test_idx /path/to/checkpoints/test_idx.pt \
      --out_dir ./final_invitro

주의
----
  * ablation_common 과 같은 폴더에서 실행해야 한다(모듈 import).
  * 데이터 경로 환경변수(ABL_*)는 학습 때와 동일하게 설정해야 split이 재현된다.
  * LoRA 계열 checkpoint는 구조가 달라 이 스크립트로 로드되지 않는다(DER++ 전용).
"""
import os, json, argparse
import numpy as np
import pandas as pd
import torch

from ablation_common import (
    SEED, set_seed, fresh_model, load_all_data, evaluate,
    bootstrap_metrics, per_volume_pred, DEV,
)

ALARM_THRESHOLD = 200.0  # mL


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="./results_final_multiseed_valsplit",
                    help="run_final_*.sh 의 save_dir")
    ap.add_argument("--strategy", default="der++",
                    help="평가할 전략 폴더명 (채택 방법: der++)")
    ap.add_argument("--seeds", default="1,2,3,42")
    ap.add_argument("--test_idx", default=None,
                    help="학습 때 쓴 clinical test 인덱스 .pt (동일해야 함)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--boot_seed", type=int, default=42,
                    help="Table 4 부트스트랩 행에 쓸 대표 seed")
    ap.add_argument("--out_dir", default="./final_invitro")
    return ap.parse_args()


def load_test_idx(path):
    if not path:
        return None
    ti = torch.load(path, weights_only=False)
    return ti.tolist() if hasattr(ti, "tolist") else list(ti)


def load_state(ckpt_path):
    """저장된 state_dict 로부터 RFNet 복원."""
    model = fresh_model()
    sd = torch.load(ckpt_path, map_location=DEV, weights_only=False)
    if not isinstance(sd, dict):
        raise TypeError(f"unexpected checkpoint type: {type(sd)}")
    # state_dict 가 통째로 저장된 경우 / {'state_dict':...} 형태 모두 대응
    if "state_dict" in sd and all(not k.startswith(("h_encoders", "s_encoders", "head"))
                                  for k in sd.keys() if k != "state_dict"):
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"    [warn] missing={len(missing)} unexpected={len(unexpected)} keys")
        if unexpected:
            print(f"    [warn] 예: {unexpected[:3]} -> LoRA 등 다른 구조일 수 있음")
    model.eval()
    return model


def alarm_metrics(res, thr=ALARM_THRESHOLD):
    """>= thr mL 을 양성으로 하는 이진 분류 성능."""
    gt = np.asarray(res["gt"], float)
    pred = np.asarray(res["pred"], float)
    y_true = gt >= thr
    y_pred = pred >= thr
    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    n = tp + tn + fp + fn
    acc = 100.0 * (tp + tn) / n if n else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(alarm_acc=acc, alarm_precision=prec, alarm_recall=rec, alarm_f1=f1,
                tp=tp, tn=tn, fp=fp, fn=fn)


def fmt(m, s, d=3):
    return f"{m:.{d}f} $\\pm$ {s:.{d}f}"


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=== in vitro 재평가 (채택 모델) ===")
    print(f"strategy={args.strategy}  seeds={seeds}")

    test_idx = load_test_idx(args.test_idx)
    data = load_all_data(test_idx)
    print(f"device  : train {len(data['dev_tr'])} / val {len(data['dev_va'])} / test {len(data['dev_te'])}")
    print(f"clinical: train {len(data['clin_tr'])} / val {len(data['clin_va'])} / test {len(data['clin_te'])}")

    rows, per_vol_rows = [], []
    keep = {}  # seed -> (dev_res, clin_res)

    for seed in seeds:
        ckpt = os.path.join(args.results_dir, args.strategy, f"seed{seed}", "model.pt")
        if not os.path.exists(ckpt):
            print(f"[skip] checkpoint 없음: {ckpt}")
            continue
        print(f"\n--- seed {seed} ---\n  {ckpt}")
        set_seed(seed)
        model = load_state(ckpt)

        dev_res = evaluate(model, data["dev_te"])
        clin_res = evaluate(model, data["clin_te"])
        al = alarm_metrics(dev_res)
        keep[seed] = (dev_res, clin_res)

        print(f"  device  : R2={dev_res['r2']:.3f} MAE={dev_res['mae']:.1f} "
              f"RMSE={dev_res['rmse']:.1f} acc50={dev_res['acc50']:.1f}%")
        print(f"  clinical: R2={clin_res['r2']:.3f} MAE={clin_res['mae']:.1f} "
              f"RMSE={clin_res['rmse']:.1f} acc50={clin_res['acc50']:.1f}%")
        print(f"  alarm(>={ALARM_THRESHOLD:.0f}mL): acc={al['alarm_acc']:.2f}% "
              f"F1={al['alarm_f1']:.3f} (TP{al['tp']} TN{al['tn']} FP{al['fp']} FN{al['fn']})")

        rows.append(dict(
            seed=seed,
            device_mae=dev_res["mae"], device_rmse=dev_res["rmse"],
            device_r2=dev_res["r2"], device_acc50=dev_res["acc50"],
            clinical_mae=clin_res["mae"], clinical_rmse=clin_res["rmse"],
            clinical_r2=clin_res["r2"], clinical_acc50=clin_res["acc50"],
            sum_r2=dev_res["r2"] + clin_res["r2"], **al))

        pv = per_volume_pred(dev_res)
        for vol, st in pv.items():
            per_vol_rows.append(dict(seed=seed, volume=vol,
                                     pred_mean=st["mean"], pred_std=st["std"], n=st["n"]))

        # Fig.4용 예측 저장
        pd.DataFrame({"gt": dev_res["gt"], "pred": dev_res["pred"]}).to_csv(
            os.path.join(args.out_dir, f"pred_device_seed{seed}.csv"), index=False)
        pd.DataFrame({"gt": clin_res["gt"], "pred": clin_res["pred"]}).to_csv(
            os.path.join(args.out_dir, f"pred_clinical_seed{seed}.csv"), index=False)

    if not rows:
        print("\n[!] 평가된 checkpoint가 없습니다. --results_dir / --strategy 를 확인하세요.")
        return

    df = pd.DataFrame(rows).sort_values("seed")
    df.to_csv(os.path.join(args.out_dir, "invitro_seedwise.csv"), index=False)
    pd.DataFrame(per_vol_rows).to_csv(
        os.path.join(args.out_dir, "invitro_per_volume.csv"), index=False)

    m, s = df.mean(numeric_only=True), df.std(ddof=1, numeric_only=True)

    # ---- 부트스트랩 (대표 seed) : Table 4 형식 ----
    boot = {}
    if args.boot_seed in keep:
        dev_res, clin_res = keep[args.boot_seed]
        boot["device"] = bootstrap_metrics(dev_res, n_boot=args.n_boot, seed=args.boot_seed)
        boot["clinical"] = bootstrap_metrics(clin_res, n_boot=args.n_boot, seed=args.boot_seed)

    summary = dict(
        strategy=args.strategy, seeds=seeds, n_seeds=len(df),
        mean={k: float(m[k]) for k in m.index},
        sd={k: float(s[k]) for k in s.index},
        bootstrap_seed=args.boot_seed, bootstrap=boot,
    )
    with open(os.path.join(args.out_dir, "invitro_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---------------- 논문에 붙여넣을 텍스트 ----------------
    lines = []
    lines.append("=" * 72)
    lines.append(f"채택 모델({args.strategy}) in vitro / cross-domain 요약  "
                 f"(mean ± SD across {len(df)} seeds)")
    lines.append("=" * 72)
    lines.append(f"device   MAE   {m.device_mae:.1f} ± {s.device_mae:.1f} mL")
    lines.append(f"device   RMSE  {m.device_rmse:.1f} ± {s.device_rmse:.1f} mL")
    lines.append(f"device   R2    {m.device_r2:.3f} ± {s.device_r2:.3f}")
    lines.append(f"device   ±50mL {m.device_acc50:.1f} ± {s.device_acc50:.1f} %")
    lines.append(f"clinical MAE   {m.clinical_mae:.1f} ± {s.clinical_mae:.1f} mL")
    lines.append(f"clinical RMSE  {m.clinical_rmse:.1f} ± {s.clinical_rmse:.1f} mL")
    lines.append(f"clinical R2    {m.clinical_r2:.3f} ± {s.clinical_r2:.3f}")
    lines.append(f"clinical ±50mL {m.clinical_acc50:.1f} ± {s.clinical_acc50:.1f} %")
    lines.append(f"alarm    acc   {m.alarm_acc:.2f} ± {s.alarm_acc:.2f} %")
    lines.append(f"alarm    F1    {m.alarm_f1:.3f} ± {s.alarm_f1:.3f}")
    lines.append(f"alarm    prec  {m.alarm_precision:.3f} ± {s.alarm_precision:.3f}")
    lines.append(f"alarm    recall{m.alarm_recall:.3f} ± {s.alarm_recall:.3f}")
    lines.append("")
    lines.append("--- Table 4 (cross-domain) 붙여넣기용 두 행 ---")
    lines.append(f"Device, after adaptation & ${fmt(m.device_mae, s.device_mae,1)}$ & "
                 f"${fmt(m.device_rmse, s.device_rmse,1)}$ & "
                 f"${fmt(m.device_r2, s.device_r2,3)}$ & "
                 f"${fmt(m.device_acc50, s.device_acc50,1)}$ \\\\")
    lines.append(f"Clinical, after adaptation & ${fmt(m.clinical_mae, s.clinical_mae,1)}$ & "
                 f"${fmt(m.clinical_rmse, s.clinical_rmse,1)}$ & "
                 f"${fmt(m.clinical_r2, s.clinical_r2,3)}$ & "
                 f"${fmt(m.clinical_acc50, s.clinical_acc50,1)}$ \\\\")
    lines.append("")
    lines.append("--- Result 2.6 / Fig.4 캡션용 문장 ---")
    lines.append(f"Volume prediction on the phantom set reached "
                 f"R^2 = {m.device_r2:.3f} ± {s.device_r2:.3f} with ±50 mL accuracy "
                 f"{m.device_acc50:.1f} ± {s.device_acc50:.1f}%, and the >=200 mL fill alarm "
                 f"achieved {m.alarm_acc:.2f} ± {s.alarm_acc:.2f}% accuracy and "
                 f"F1 {m.alarm_f1:.3f} ± {s.alarm_f1:.3f} across {len(df)} seeds.")
    lines.append("")
    lines.append("--- device 볼륨별 평균 예측 (seed 평균) ---")
    pv = pd.DataFrame(per_vol_rows).groupby("volume")["pred_mean"].agg(["mean", "std"])
    for vol, r in pv.iterrows():
        sd_txt = f" ± {r['std']:.1f}" if not np.isnan(r["std"]) else ""
        lines.append(f"  {vol:.0f} mL 목표 -> 평균 예측 {r['mean']:.1f}{sd_txt} mL")
    if boot:
        lines.append("")
        lines.append(f"--- 부트스트랩 (seed {args.boot_seed}, n_boot={args.n_boot}) ---")
        for dom in ("device", "clinical"):
            b = boot[dom]
            lines.append(f"  {dom:8s} MAE {b['mae']:.1f}±{b['mae_std']:.1f}  "
                         f"RMSE {b['rmse']:.1f}±{b['rmse_std']:.1f}  "
                         f"R2 {b['r2']:.3f}±{b['r2_std']:.3f}  "
                         f"±50mL {b['acc50']:.1f}±{b['acc50_std']:.1f}")

    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(args.out_dir, "paper_values.txt"), "w") as f:
        f.write(txt + "\n")

    print(f"\n[saved] {args.out_dir}/")
    print("  invitro_seedwise.csv   : seed별 전체 지표")
    print("  invitro_per_volume.csv : 볼륨별 예측")
    print("  invitro_summary.json   : mean/SD/bootstrap")
    print("  paper_values.txt       : 논문 붙여넣기용 문장·표 행")
    print("  pred_device_seed*.csv  : Fig.4a/4b 재작도용 예측")


if __name__ == "__main__":
    main()
