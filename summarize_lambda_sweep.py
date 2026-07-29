"""
summarize_lambda_sweep.py -- λ sweep 결과 요약 및 논문 반영

stage 1(스크리닝) 결과에서 전략별 최선 λ 를 고르고, stage 2 실행 명령을
자동 생성한다. stage 2 까지 끝나면 논문용 표와 서술 초안을 만든다.

사용:
  # stage 1 끝난 뒤
  python summarize_lambda_sweep.py --save_dir ./results_lambda_sweep

  # stage 2 까지 끝난 뒤 (4 seed 확정 결과 포함)
  python summarize_lambda_sweep.py --save_dir ./results_lambda_sweep --final
"""
import os, json, glob, argparse
import numpy as np
import pandas as pd

LAMBDA_ATTR = {"ewc": "ewc_lambda", "si": "si_lambda", "lwf": "lwf_alpha"}
# 본 실험에서 쓴 기본값 (Table 3 의 값이 나온 설정)
DEFAULT_LAM = {"ewc": 1000.0, "si": 1.0, "lwf": 1.0}
# Table 3 의 기본 λ 결과 (4 seed) — 비교 기준
BASELINE = {
    "ewc": dict(dev=0.718, clin=-3.491, sum=-2.773),
    "si":  dict(dev=0.151, clin=-4.589, sum=-4.438),
    "lwf": dict(dev=0.756, clin=-2.235, sum=-1.478),
}
DERPP = dict(dev=0.701, clin=0.641, sum=1.342)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", default="./results_lambda_sweep")
    ap.add_argument("--final", action="store_true",
                    help="stage 2 까지 완료된 경우 논문용 표/서술까지 생성")
    return ap.parse_args()


def load_all(save_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(save_dir, "*", "lam*", "seed*", "metrics.json"))):
        with open(p) as f:
            m = json.load(f)
        m.pop("device_per_volume", None)
        rows.append(m)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    df = load_all(args.save_dir)
    if df.empty:
        print(f"[!] 결과 없음: {args.save_dir}/*/lam*/seed*/metrics.json")
        return

    df = df.sort_values(["strategy", "lam", "seed"])
    df.to_csv(os.path.join(args.save_dir, "sweep_all.csv"), index=False)

    n_seeds = df.groupby(["strategy", "lam"]).seed.nunique()
    print("=" * 74)
    print("λ sweep 결과")
    print("=" * 74)

    best = {}
    for strat in sorted(df.strategy.unique()):
        sub = df[df.strategy == strat]
        g = sub.groupby("lam").agg(
            dev=("device_r2", "mean"), dev_sd=("device_r2", "std"),
            clin=("clinical_r2", "mean"), clin_sd=("clinical_r2", "std"),
            s=("tradeoff_sum", "mean"), s_sd=("tradeoff_sum", "std"),
            n=("seed", "nunique")).sort_index()
        print(f"\n  {strat.upper()}  ({LAMBDA_ATTR[strat]})")
        base = BASELINE.get(strat)
        for lam, r in g.iterrows():
            mark = "  <- 본 실험 기본값" if abs(lam - DEFAULT_LAM[strat]) < 1e-9 else ""
            sd = f" ± {r.s_sd:.3f}" if r.n > 1 and not np.isnan(r.s_sd) else ""
            print(f"    λ={lam:<10g} n={int(r.n)}  dev={r.dev:+.3f}  "
                  f"clin={r.clin:+.3f}  sum={r.s:+.3f}{sd}{mark}")
        bl = g.s.idxmax()
        best[strat] = dict(lam=float(bl), **{k: float(g.loc[bl, k]) for k in
                                             ["dev", "clin", "s"]},
                           n=int(g.loc[bl, "n"]))
        print(f"    => 최선 λ = {bl:g}  (sum={g.loc[bl,'s']:+.3f})")
        if base:
            d = g.loc[bl, "s"] - base["sum"]
            print(f"       기본 λ 대비 sum {d:+.3f}  "
                  f"(기본 {base['sum']:+.3f} -> 최선 {g.loc[bl,'s']:+.3f})")
            if g.loc[bl, "clin"] > 0:
                print(f"       [!] 최선 λ 에서 clinical R2 가 양수({g.loc[bl,'clin']:+.3f})로 회복됨")
                print(f"           -> 논문의 '이 방법들은 실패했다' 서술을 반드시 수정할 것")

    # ---------------- stage 2 명령 생성 ----------------
    need2 = {s: b for s, b in best.items() if b["n"] < 4}
    if need2:
        spec = "|".join(f"{s}:{b['lam']:g}" for s, b in need2.items())
        print("\n" + "=" * 74)
        print("Stage 2 실행 명령 (최선 λ 를 나머지 seed 로 확정)")
        print("=" * 74)
        print(f"""
CUDA_VISIBLE_DEVICES=0 python -u lambda_sweep.py \\
  --stage 2 --seeds 2,3,42 \\
  --ckpt "$CKPT" --test_idx "$TEST_IDX" \\
  --jobs "{spec}" \\
  --epochs 150 --lr 1e-3 --save_dir {args.save_dir}
""")
        print("  (GPU 2대로 나누려면 --jobs 를 전략별로 쪼개 두 번 실행)")

    # ---------------- 논문용 산출 ----------------
    if args.final:
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\caption{\textbf{Regularization strength for the",
            r"non-replay continual-learning baselines.} For each method the",
            r"regularization weight was swept on a single seed and the best",
            r"setting, by the same device--clinical criterion used throughout, was",
            r"then run with all four seeds. Values are mean $\pm$ standard",
            r"deviation across seeds where four runs are available.}",
            r"\label{tab:lambda-sweep}",
            r"\begin{tabular}{llccc}",
            r"\toprule",
            r"\textbf{Method} & \textbf{Weight} & \textbf{Device $R^2$} &"
            r" \textbf{Clinical $R^2$} & \textbf{$R^2$ sum} \\",
            r"\midrule",
        ]
        name = {"ewc": "EWC", "si": "SI", "lwf": "LwF"}

        def lam_tex(v):
            """1e+06 대신 10^{6} 형태로."""
            if v >= 1000 and float(v).is_integer():
                e = int(round(np.log10(v)))
                if abs(10 ** e - v) < 1e-6:
                    return f"10^{{{e}}}"
            return f"{v:g}"
        for strat in ["ewc", "si", "lwf"]:
            if strat not in best:
                continue
            sub = df[df.strategy == strat]
            for lam in [DEFAULT_LAM[strat], best[strat]["lam"]]:
                s2 = sub[np.isclose(sub.lam, lam)]
                if s2.empty:
                    continue
                tag = "default" if abs(lam - DEFAULT_LAM[strat]) < 1e-9 else "best"
                def fmt(col, d=3):
                    m, sd = s2[col].mean(), s2[col].std(ddof=1)
                    return (f"${m:.{d}f} \\pm {sd:.{d}f}$" if len(s2) > 1 and not np.isnan(sd)
                            else f"${m:.{d}f}$")
                lines.append(
                    f"{name[strat]} ({tag}) & $\\lambda = {lam_tex(lam)}$ & {fmt('device_r2')} & "
                    f"{fmt('clinical_r2')} & {fmt('tradeoff_sum')} \\\\")
            lines.append(r"\addlinespace[2pt]")
        lines += [
            r"\midrule",
            f"DER++ (adopted) & --- & ${DERPP['dev']:.3f}$ & ${DERPP['clin']:.3f}$ & ${DERPP['sum']:.3f}$ \\\\",
            r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        tex = "\n".join(lines)
        with open(os.path.join(args.save_dir, "table_lambda_sweep.tex"), "w") as f:
            f.write(tex)
        print(f"\n[saved] {args.save_dir}/table_lambda_sweep.tex")

        # 서술 초안
        recovered = [s for s, b in best.items() if b["clin"] > 0]
        note = []
        note.append("--- 논문 서술 초안 ---")
        if recovered:
            note.append(
                f"주의: {', '.join(x.upper() for x in recovered)} 는 λ 조정 후 clinical R2 가 "
                f"양수로 회복되었습니다. Results 2.3 과 Discussion 의 "
                f"'failed to prevent forgetting' 서술을 다음과 같이 바꿔야 합니다:")
            note.append(
                "  기존: 'EWC, SI, and LwF all produced strongly negative clinical-retention R2'")
            note.append(
                "  수정: 기본 설정에서는 음수였으나 λ 를 조정하면 일부 회복되며, "
                "그럼에도 replay 계열에는 미치지 못한다는 식으로 서술")
        else:
            note.append(
                "λ 를 넓게 조정해도 세 방법 모두 clinical R2 가 음수에 머물렀습니다. "
                "이는 '하이퍼파라미터 탓이 아니라 방법 계열의 한계'라는 기존 서술을 "
                "강하게 뒷받침합니다. Methods 에 sweep 범위를 명시하고 "
                "Supplementary 에 표를 넣으면 straw-man 비판을 차단할 수 있습니다.")
        for s, b in best.items():
            note.append(f"  {s.upper()}: 최선 λ={b['lam']:g} -> dev {b['dev']:+.3f}, "
                        f"clin {b['clin']:+.3f}, sum {b['s']:+.3f}  "
                        f"(DER++ sum {DERPP['sum']:.3f})")
        txt = "\n".join(note)
        print("\n" + txt)
        with open(os.path.join(args.save_dir, "interpretation.txt"), "w") as f:
            f.write(txt + "\n")
        print(f"[saved] {args.save_dir}/interpretation.txt")

    print(f"\n[saved] {args.save_dir}/sweep_all.csv")


if __name__ == "__main__":
    main()
