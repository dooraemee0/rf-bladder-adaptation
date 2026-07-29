"""
plot_fig4.py -- 재평가 결과로 Fig.4
(a) 예측 산점도
(b) alarm 혼동행렬 다시 그리기

eval_invitro_final.py가 만든 pred_device_seed*.csv를 읽어 논문 그림 스타일로
그린다.

사용 예
-------
단일 seed:
    python plot_fig4.py --in_dir ./final_invitro --seed 42

디렉터리 내 모든 seed pooling:
    python plot_fig4.py --in_dir ./final_invitro --pool

선택한 seed만 pooling:
    python plot_fig4.py --in_dir ./final_invitro --pool --seeds 1,2,3,42

공백 구분 방식도 가능:
    python plot_fig4.py --in_dir ./final_invitro --pool --seeds 1 2 3 42
"""

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALARM_THRESHOLD = 200.0
REQUIRED_COLUMNS = {"gt", "pred"}


def parse_seed_values(values):
    """
    argparse에서 받은 seed 문자열들을 정수 리스트로 변환한다.

    지원 형식:
        --seeds 1,2,3,42
        --seeds 1 2 3 42
        --seeds 1,2 3,42
    """
    if not values:
        return None

    seeds = []

    for value in values:
        for token in value.split(","):
            token = token.strip()

            if not token:
                continue

            try:
                seeds.append(int(token))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"잘못된 seed 값입니다: {token!r}"
                ) from exc

    if not seeds:
        raise argparse.ArgumentTypeError(
            "--seeds에 하나 이상의 정수 seed를 입력해야 합니다."
        )

    # 순서는 유지하면서 중복 제거
    return list(dict.fromkeys(seeds))


def parse_args():
    parser = argparse.ArgumentParser(
        description="pred_device_seed*.csv 결과로 Fig.4를 다시 그립니다."
    )

    parser.add_argument(
        "--in_dir",
        default="./final_invitro",
        help="pred_device_seed*.csv 파일이 위치한 디렉터리",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="단일 seed로 그릴 때 사용할 seed. 기본값: 42",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        metavar="SEED",
        help=(
            "pooling에 사용할 seed 목록. "
            "예: --seeds 1,2,3,42 또는 --seeds 1 2 3 42"
        ),
    )

    parser.add_argument(
        "--pool",
        action="store_true",
        help="여러 seed의 예측 결과를 합쳐서 그립니다.",
    )

    parser.add_argument(
        "--out_prefix",
        default="fig4_regen",
        help="출력 파일명 prefix. 기본값: fig4_regen",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="출력 이미지 DPI. 기본값: 300",
    )

    args = parser.parse_args()

    try:
        args.seeds = parse_seed_values(args.seeds)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    if args.seeds is not None and not args.pool:
        parser.error(
            "--seeds는 여러 seed를 합칠 때 사용하는 옵션입니다. "
            "--pool도 함께 지정하세요."
        )

    if args.dpi <= 0:
        parser.error("--dpi는 0보다 큰 정수여야 합니다.")

    return args


def extract_seed_from_filename(filepath):
    """pred_device_seed42.csv에서 42를 추출한다."""
    filename = os.path.basename(filepath)
    match = re.search(r"pred_device_seed(-?\d+)\.csv$", filename)

    if match is None:
        return None

    return int(match.group(1))


def validate_dataframe(df, filepath):
    """CSV에 필요한 열과 유효한 숫자 데이터가 있는지 검사한다."""
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{filepath}에 필수 열이 없습니다: "
            f"{sorted(missing_columns)}. "
            f"현재 열: {list(df.columns)}"
        )

    validated = df.copy()

    validated["gt"] = pd.to_numeric(validated["gt"], errors="coerce")
    validated["pred"] = pd.to_numeric(validated["pred"], errors="coerce")

    invalid_mask = validated[["gt", "pred"]].isna().any(axis=1)
    invalid_count = int(invalid_mask.sum())

    if invalid_count > 0:
        print(
            f"[warning] {filepath}: gt 또는 pred가 유효하지 않은 "
            f"{invalid_count}개 행을 제외합니다."
        )
        validated = validated.loc[~invalid_mask].copy()

    if validated.empty:
        raise ValueError(
            f"{filepath}에 사용할 수 있는 gt/pred 데이터가 없습니다."
        )

    return validated


def read_prediction_csv(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")

    try:
        df = pd.read_csv(filepath)
    except Exception as exc:
        raise RuntimeError(
            f"CSV 파일을 읽는 중 오류가 발생했습니다: {filepath}"
        ) from exc

    df = validate_dataframe(df, filepath)
    df["source_file"] = os.path.basename(filepath)

    seed = extract_seed_from_filename(filepath)
    if seed is not None:
        df["seed"] = seed

    return df


def load(args):
    if args.pool:
        if args.seeds is not None:
            files = [
                os.path.join(
                    args.in_dir,
                    f"pred_device_seed{seed}.csv",
                )
                for seed in args.seeds
            ]

            missing_files = [
                filepath
                for filepath in files
                if not os.path.isfile(filepath)
            ]

            if missing_files:
                missing_text = "\n".join(
                    f"  - {filepath}" for filepath in missing_files
                )
                raise FileNotFoundError(
                    "요청한 seed의 CSV 파일 중 일부를 찾을 수 없습니다:\n"
                    f"{missing_text}"
                )

            selected_seeds = args.seeds

        else:
            pattern = os.path.join(
                args.in_dir,
                "pred_device_seed*.csv",
            )
            files = glob.glob(pattern)

            if not files:
                raise FileNotFoundError(
                    f"{args.in_dir}에서 "
                    "pred_device_seed*.csv 파일을 찾을 수 없습니다."
                )

            # 파일명에 포함된 seed를 기준으로 정렬
            files = sorted(
                files,
                key=lambda filepath: (
                    extract_seed_from_filename(filepath) is None,
                    extract_seed_from_filename(filepath)
                    if extract_seed_from_filename(filepath) is not None
                    else filepath,
                ),
            )

            selected_seeds = [
                extract_seed_from_filename(filepath)
                for filepath in files
            ]
            selected_seeds = [
                seed for seed in selected_seeds if seed is not None
            ]

        frames = [read_prediction_csv(filepath) for filepath in files]
        df = pd.concat(frames, ignore_index=True)

        if selected_seeds:
            seed_text = ", ".join(map(str, selected_seeds))
            tag = (
                f"pooled over {len(files)} seeds "
                f"({seed_text})"
            )
        else:
            tag = f"pooled over {len(files)} files"

        print("[load] pooled files:")
        for filepath in files:
            print(f"  - {filepath}")

    else:
        filepath = os.path.join(
            args.in_dir,
            f"pred_device_seed{args.seed}.csv",
        )

        df = read_prediction_csv(filepath)
        tag = f"seed {args.seed}"

        print(f"[load] {filepath}")

    print(f"[load] total samples: {len(df)}")

    return df, tag


def calculate_r2(gt, pred):
    ss_res = np.sum((gt - pred) ** 2)
    ss_tot = np.sum((gt - gt.mean()) ** 2)

    if np.isclose(ss_tot, 0.0):
        return np.nan

    return 1.0 - ss_res / ss_tot


def format_r2(r2):
    if np.isnan(r2):
        return "N/A"

    return f"{r2:.3f}"


def make_scatter_plot(gt, pred, tag, args):
    r2 = calculate_r2(gt, pred)
    absolute_error = np.abs(pred - gt)
    acc50 = 100.0 * np.mean(absolute_error <= 50.0)
    mae = np.mean(absolute_error)

    fig, ax = plt.subplots(figsize=(4.2, 3.6))

    data_max = max(
        400.0,
        float(np.max(gt)),
        float(np.max(pred)),
    )
    upper_limit = np.ceil(data_max * 1.1 / 50.0) * 50.0
    lim = (0.0, upper_limit)

    ax.plot(
        lim,
        lim,
        "k--",
        linewidth=1,
        label="Ideal (y=x)",
        zorder=1,
    )

    x_band = np.linspace(lim[0], lim[1], 300)
    lower_band = np.maximum(lim[0], x_band - 50.0)
    upper_band = np.minimum(lim[1], x_band + 50.0)

    ax.fill_between(
        x_band,
        lower_band,
        upper_band,
        color="tab:green",
        alpha=0.12,
        label=r"$\pm$50 mL",
        zorder=0,
    )

    # 같은 실제 용량에 점이 겹치는 것을 완화하기 위한 고정 jitter
    rng = np.random.RandomState(0)
    jitter = rng.uniform(-6.0, 6.0, size=len(gt))

    ax.scatter(
        gt + jitter,
        pred,
        s=16,
        alpha=0.6,
        color="tab:blue",
        edgecolors="none",
        label="Predictions",
        zorder=2,
    )

    unique_volumes = np.sort(np.unique(gt))

    for index, volume in enumerate(unique_volumes):
        mask = gt == volume
        volume_predictions = pred[mask]

        mean_prediction = float(np.mean(volume_predictions))

        # 표본이 하나인 경우 표준편차는 0으로 표시
        if len(volume_predictions) > 1:
            std_prediction = float(
                np.std(volume_predictions, ddof=1)
            )
        else:
            std_prediction = 0.0

        ax.errorbar(
            volume,
            mean_prediction,
            yerr=std_prediction,
            fmt="o",
            color="tab:red",
            markersize=6,
            capsize=4,
            zorder=3,
            label=(
                r"Mean $\pm$ SD"
                if index == 0
                else None
            ),
        )

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual volume (mL)")
    ax.set_ylabel("Predicted volume (mL)")

    ax.set_title(
        f"Volume prediction ({tag})\n"
        f"$R^2$={format_r2(r2)}, "
        f"MAE={mae:.1f} mL, "
        f"$\\pm$50 mL acc {acc50:.1f}%",
        fontsize=9,
    )

    ax.legend(
        fontsize=7,
        loc="upper left",
        framealpha=0.9,
    )

    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    output_path = f"{args.out_prefix}_a_scatter.png"
    fig.savefig(
        output_path,
        dpi=args.dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    return output_path, r2, mae, acc50


def make_alarm_plot(gt, pred, tag, args):
    y_true = gt >= ALARM_THRESHOLD
    y_pred = pred >= ALARM_THRESHOLD

    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))

    cm = np.array(
        [
            [tn, fp],
            [fn, tp],
        ],
        dtype=int,
    )

    sample_count = int(cm.sum())

    if sample_count == 0:
        raise ValueError(
            "혼동행렬을 계산할 데이터가 없습니다."
        )

    accuracy = 100.0 * (tp + tn) / sample_count

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    fig, ax = plt.subplots(figsize=(3.6, 3.2))

    image = ax.imshow(cm, cmap="viridis")

    cm_max = int(cm.max())

    for row in range(2):
        for column in range(2):
            value = int(cm[row, column])

            # viridis의 밝은 영역에서는 검은색,
            # 어두운 영역에서는 흰색 글씨 사용
            if cm_max == 0:
                text_color = "white"
            else:
                text_color = (
                    "black"
                    if value >= cm_max * 0.6
                    else "white"
                )

            ax.text(
                column,
                row,
                str(value),
                horizontalalignment="center",
                verticalalignment="center",
                color=text_color,
                fontsize=15,
                fontweight="bold",
            )

    threshold_label_low = f"< {ALARM_THRESHOLD:.0f}"
    threshold_label_high = (
        rf"$\geq$ {ALARM_THRESHOLD:.0f}"
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        [
            threshold_label_low,
            threshold_label_high,
        ]
    )
    ax.set_yticklabels(
        [
            threshold_label_low,
            threshold_label_high,
        ]
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_title(
        rf"$\geq${ALARM_THRESHOLD:.0f} mL alarm "
        f"({tag})\n"
        f"Acc {accuracy:.2f}%, F1 {f1:.3f}",
        fontsize=9,
    )

    fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    fig.tight_layout()

    output_path = f"{args.out_prefix}_b_alarm.png"
    fig.savefig(
        output_path,
        dpi=args.dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    return output_path, metrics


def main():
    args = parse_args()

    df, tag = load(args)

    gt = df["gt"].to_numpy(dtype=float)
    pred = df["pred"].to_numpy(dtype=float)

    scatter_path, r2, mae, acc50 = make_scatter_plot(
        gt=gt,
        pred=pred,
        tag=tag,
        args=args,
    )

    alarm_path, alarm_metrics = make_alarm_plot(
        gt=gt,
        pred=pred,
        tag=tag,
        args=args,
    )

    print(f"\n[saved] {scatter_path}")
    print(f"[saved] {alarm_path}")

    print(
        "\n"
        f"(a) R2={format_r2(r2)}  "
        f"MAE={mae:.1f} mL  "
        f"±50mL={acc50:.1f}%"
    )

    print(
        f"(b) acc={alarm_metrics['accuracy']:.2f}%  "
        f"precision={alarm_metrics['precision']:.3f}  "
        f"recall={alarm_metrics['recall']:.3f}  "
        f"F1={alarm_metrics['f1']:.3f}"
    )

    print(
        "    confusion "
        f"[TN {alarm_metrics['tn']}]"
        f"[FP {alarm_metrics['fp']}] / "
        f"[FN {alarm_metrics['fn']}]"
        f"[TP {alarm_metrics['tp']}]"
    )

    print("\n캡션에 쓸 문장:")

    print(
        "  (a) Volume prediction on the phantom set "
        f"($R^2$ = {format_r2(r2)}, "
        f"$\\pm$50 mL accuracy {acc50:.1f}\\%)."
    )

    print(
        "  (b) $\\geq$200 mL alarm classification "
        "on the same set "
        f"(accuracy {alarm_metrics['accuracy']:.2f}\\%, "
        f"F1 {alarm_metrics['f1']:.3f})."
    )


if __name__ == "__main__":
    main()