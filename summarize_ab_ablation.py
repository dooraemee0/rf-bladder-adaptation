"""
summarize_ab_ablation.py -- DER++ loss-term ablation 요약

run_derpp_ablation.sh 결과(alpha0, beta0)와 기존 full DER++ 결과를 모아
논문에 넣을 표와 해석 문장을 만든다.

사용:
  python summarize_ab_ablation.py \
      --save_dir ./results_derpp_ablation \
      --full_dir ./results_final_multiseed_valsplit

산출물:
  <save_dir>/ab_ablation_seedwise.csv
  <save_dir>/ab_ablation_mean.csv
  <save_dir>/table_ab_ablation.tex     논문 붙여넣기용
  <save_dir>/interpretation.txt        결과 해석 초안
"""
import os, json, glob, argparse
import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", default="./results_derpp_ablation")
    ap.add_argument("--full_dir", default="./results_final_multiseed_valsplit",
                    help="기존 full DER++ (alpha=beta=0.5) 결과 폴더")
    return ap.parse_args()


def load_cond(root, strategy="der++"):
    """<root>/<strategy>/seed*/metrics.json 을 모아 DataFrame 으로."""
    rows = []
    for p in sorted(glob.glob(os.path.join(root, strategy, "seed*", "metrics.json"))):
        with open(p) as f:
            m = json.load(f)
        m.pop("device_per_volume", None)
        m.pop("cfp_channel_feature", None)
        rows.append(m)
    return pd.DataFrame(rows)


def main():
    args = parse_args()

    conds = {
        "full":   (args.full_dir,                         r"DER++ (full, $\alpha=\beta=0.5$)"),
        "alpha0": (os.path.join(args.save_dir, "alpha0"), r"$\alpha = 0$ (no dark experience)"),
        "beta0":  (os.path.join(args.save_dir, "beta0"),  r"$\beta = 0$ (no ground-truth replay)"),
    }

    frames = []
    for key, (root, _) in conds.items():
        df = load_cond(root)
        if df.empty:
            print(f"[!] 결과 없음: {root}/der++/seed*/metrics.json")
            continue
        df["condition"] = key
        frames.append(df)

    if not frames:
        print("결과를 찾지 못했습니다. --save_dir / --full_dir 확인.")
        return

    all_df = pd.concat(frames, ignore_index=True)
    keep = ["condition", "seed", "device_mae", "device_r2",
            "clinical_mae", "clinical_r2", "tradeoff_sum"]
    all_df = all_df[[c for c in keep if c in all_df.columns]]
    all_df = all_df.sort_values(["condition", "seed"])
    all_df.to_csv(os.path.join(args.save_dir, "ab_ablation_seedwise.csv"), index=False)

    g = all_df.groupby("condition")
    num = ["device_mae", "device_r2", "clinical_mae", "clinical_r2", "tradeoff_sum"]
    mean = g[num].mean()
    sd = g[num].std(ddof=1)
    sd.columns = [c + "_sd" for c in sd.columns]
    mm = mean.join(sd)
    mm["n"] = g.size()
    mm.to_csv(os.path.join(args.save_dir, "ab_ablation_mean.csv"))

    order = [k for k in ["full", "alpha0", "beta0"] if k in mm.index]

    # ---------------- LaTeX ----------------
    def f(c, col, d=3):
        return f"${mm.loc[c, col]:.{d}f} \\pm {mm.loc[c, col+'_sd']:.{d}f}$"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{\textbf{Contribution of the two clinical replay terms.} Both",
        r"terms of equation~(\ref{eq:derpp}) are removed in turn while the device",
        r"regression loss, adaptation scope, optimizer, schedule, data partitions,",
        r"and checkpoint criterion are held fixed. Values are mean $\pm$ standard",
        r"deviation across seeds $1$, $2$, $3$, and $42$ on the held-out test sets.}",
        r"\label{tab:ab-ablation}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Objective} & \textbf{Device $R^2$} & \textbf{Clinical MAE (mL)} &"
        r" \textbf{Clinical $R^2$} & \textbf{$R^2$ sum} \\",
        r"\midrule",
    ]
    for c in order:
        label = conds[c][1]
        if c == "full":
            label = r"\textbf{" + label + "}"
        lines.append(f"{label} & {f(c,'device_r2')} & {f(c,'clinical_mae',1)} & "
                     f"{f(c,'clinical_r2')} & {f(c,'tradeoff_sum')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)
    with open(os.path.join(args.save_dir, "table_ab_ablation.tex"), "w") as fp:
        fp.write(tex)

    # ---------------- 해석 ----------------
    out = []
    out.append("=" * 74)
    out.append("DER++ loss-term ablation  (mean ± SD across seeds)")
    out.append("=" * 74)
    for c in order:
        out.append(f"  {c:7s} dev R2={mm.loc[c,'device_r2']:.3f}±{mm.loc[c,'device_r2_sd']:.3f}  "
                   f"clin R2={mm.loc[c,'clinical_r2']:+.3f}±{mm.loc[c,'clinical_r2_sd']:.3f}  "
                   f"clin MAE={mm.loc[c,'clinical_mae']:.1f}  "
                   f"sum={mm.loc[c,'tradeoff_sum']:.3f}±{mm.loc[c,'tradeoff_sum_sd']:.3f}  "
                   f"(n={int(mm.loc[c,'n'])})")
    out.append("")

    if "full" in mm.index:
        base = mm.loc["full"]
        out.append("--- full 대비 변화 ---")
        for c in order:
            if c == "full":
                continue
            d_dev = mm.loc[c, "device_r2"] - base["device_r2"]
            d_cli = mm.loc[c, "clinical_r2"] - base["clinical_r2"]
            d_sum = mm.loc[c, "tradeoff_sum"] - base["tradeoff_sum"]
            out.append(f"  {c:7s} device {d_dev:+.3f}   clinical {d_cli:+.3f}   sum {d_sum:+.3f}")
        out.append("")

        # seed 별 승패 (부호 검정용)
        piv = all_df.pivot(index="seed", columns="condition", values="tradeoff_sum")
        out.append("--- seed 별 R2 합 (full 이 몇 번 이기는가) ---")
        for c in order:
            if c == "full":
                continue
            if c in piv.columns and "full" in piv.columns:
                wins = int((piv["full"] > piv[c]).sum())
                n = int(piv[[c, "full"]].dropna().shape[0])
                out.append(f"  full vs {c:7s}: {wins}/{n} 승")
        out.append("")

        # 해석 초안
        out.append("--- 논문 문장 초안 ---")
        a = mm.loc["alpha0"] if "alpha0" in mm.index else None
        b = mm.loc["beta0"] if "beta0" in mm.index else None
        if a is not None and b is not None:
            out.append(
                f"Removing the dark-experience term (alpha=0) changed clinical retention from "
                f"{base['clinical_r2']:.3f} to {a['clinical_r2']:.3f} and the combined score from "
                f"{base['tradeoff_sum']:.3f} to {a['tradeoff_sum']:.3f}. Removing the "
                f"ground-truth replay term (beta=0) changed them to {b['clinical_r2']:.3f} and "
                f"{b['tradeoff_sum']:.3f}, respectively.")
            out.append("")
            # 자동 판정
            worst = "alpha0" if a["tradeoff_sum"] < b["tradeoff_sum"] else "beta0"
            drop_a = base["tradeoff_sum"] - a["tradeoff_sum"]
            drop_b = base["tradeoff_sum"] - b["tradeoff_sum"]
            sd_ref = base["tradeoff_sum_sd"]
            out.append(f"  [자동 판정] full 대비 하락: alpha0={drop_a:+.3f}, beta0={drop_b:+.3f}"
                       f"  (full 의 seed SD={sd_ref:.3f})")
            if drop_a > 2 * sd_ref and drop_b > 2 * sd_ref:
                out.append("  -> 두 항 모두 제거 시 SD 의 2배 이상 하락. '조합이 필요하다' 주장 성립.")
            elif max(drop_a, drop_b) <= 2 * sd_ref:
                out.append("  -> 어느 항을 빼도 큰 차이 없음. 더 단순한 목적함수로 충분할 수 있음.")
                out.append("     이 경우 논문에서 '두 항의 조합' 주장을 완화해야 한다.")
            else:
                key = "dark experience" if drop_a > drop_b else "ground-truth replay"
                out.append(f"  -> 주로 '{key}' 항이 기여. 나머지 항의 기여는 작음.")
                out.append("     이 경우 '조합' 대신 해당 항 중심으로 서술을 조정할 것.")

    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(args.save_dir, "interpretation.txt"), "w") as fp:
        fp.write(txt + "\n")

    print(f"\n[saved] {args.save_dir}/table_ab_ablation.tex")
    print(f"[saved] {args.save_dir}/interpretation.txt")


if __name__ == "__main__":
    main()
