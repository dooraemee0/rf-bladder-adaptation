"""
aggregate_summary.py -- 여러 GPU/실행에서 나온 metrics.json 을 모두 모아
하나의 통합 summary(전 seed + seed평균±표준편차)로 만든다.

병렬 실행(GPU0/GPU1)은 각자 자기 실행분만 summary CSV에 쓰기 때문에,
전부 끝난 뒤 이 스크립트로 <save_dir> 아래 모든 metrics.json 을 스캔해
최종 표를 재생성한다.

사용:
  python aggregate_summary.py --save_dir ./results_final_multiseed_valsplit
출력:
  <save_dir>/summary_all_seedwise.csv
  <save_dir>/summary_all_mean.csv
  <save_dir>/summary_all.tex
  <save_dir>/cfp_channel_feature.csv   (cfp_derpp 채널 feature distance)
"""
import os, json, glob, argparse
import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", default="./results_final_multiseed_valsplit")
    return ap.parse_args()


def main():
    args = parse_args()
    paths = glob.glob(os.path.join(args.save_dir, "*", "seed*", "metrics.json"))
    if not paths:
        print(f"[!] metrics.json 없음: {args.save_dir}")
        return
    rows, ch_rows = [], []
    for p in sorted(paths):
        with open(p) as f:
            m = json.load(f)
        ch = m.pop("cfp_channel_feature", None)
        m.pop("device_per_volume", None)
        rows.append(m)
        if ch:
            r = {"strategy": m["strategy"], "seed": m["seed"]}
            r.update(ch)
            ch_rows.append(r)

    df = pd.DataFrame(rows).sort_values(["strategy", "seed"]).reset_index(drop=True)
    df.to_csv(os.path.join(args.save_dir, "summary_all_seedwise.csv"), index=False)

    num_cols = [c for c in df.columns
                if c not in ("strategy", "seed") and pd.api.types.is_numeric_dtype(df[c])]
    grp = df.groupby("strategy")
    mean = grp[num_cols].mean()
    std = grp[["device_r2", "clinical_r2", "tradeoff_sum", "device_mae", "clinical_mae"]].std()
    std.columns = [c + "_seedstd" for c in std.columns]
    n = grp.size().rename("n_seeds")
    out = mean.join(std).join(n).reset_index()
    out = out.sort_values("tradeoff_sum", ascending=False).reset_index(drop=True)
    out.to_csv(os.path.join(args.save_dir, "summary_all_mean.csv"), index=False)

    # LaTeX (mean ± seed std)
    def fmt(m, s):
        return f"${m:.3f} \\pm {s:.3f}$"
    lines = ["\\begin{tabular}{lccccc}", "\\toprule",
             "Strategy & Device $R^2$ & Clinical $R^2$ & $R^2$ sum & Device MAE & Clinical MAE \\\\",
             "\\midrule"]
    for _, r in out.iterrows():
        name = str(r["strategy"]).replace("_", "\\_")
        lines.append(
            f"{name} & {fmt(r['device_r2'], r['device_r2_seedstd'])} & "
            f"{fmt(r['clinical_r2'], r['clinical_r2_seedstd'])} & "
            f"{fmt(r['tradeoff_sum'], r['tradeoff_sum_seedstd'])} & "
            f"${r['device_mae']:.1f} \\pm {r['device_mae_seedstd']:.1f}$ & "
            f"${r['clinical_mae']:.1f} \\pm {r['clinical_mae_seedstd']:.1f}$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(args.save_dir, "summary_all.tex"), "w") as f:
        f.write("\n".join(lines))

    if ch_rows:
        chdf = pd.DataFrame(ch_rows).sort_values(["strategy", "seed"])
        chdf.to_csv(os.path.join(args.save_dir, "cfp_channel_feature.csv"), index=False)

    print("=== 통합 요약 (seed 평균 ± seed표준편차, TEST) ===")
    for _, r in out.iterrows():
        print(f"  {r['strategy']:<12} n={int(r['n_seeds'])} "
              f"dev={r['device_r2']:.3f}±{r['device_r2_seedstd']:.3f} "
              f"clin={r['clinical_r2']:.3f}±{r['clinical_r2_seedstd']:.3f} "
              f"sum={r['tradeoff_sum']:.3f}±{r['tradeoff_sum_seedstd']:.3f}")
    print(f"\n[saved] {args.save_dir}/summary_all_seedwise.csv / _mean.csv / .tex")
    if ch_rows:
        print(f"[saved] {args.save_dir}/cfp_channel_feature.csv")


if __name__ == "__main__":
    main()
