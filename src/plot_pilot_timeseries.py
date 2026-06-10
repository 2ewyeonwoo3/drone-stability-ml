"""
plot_pilot_timeseries.py — 파일럿 run 시계열 시각화.

2차 파일럿 후 wind run의 "서서히 발산 vs 급추락"을 눈으로 확인하는 용도.
check_label_dist.py 숫자만으로는 판단이 어려울 때 반드시 이 그림을 먼저 봐야 함.

출력:
  results/plots/pilot_timeseries_<run_id>.png

실행 예시:
  # wind 전체 (medium 강도)
  python src/plot_pilot_timeseries.py --pattern "wind_medium"

  # 특정 run 하나
  python src/plot_pilot_timeseries.py --run-id wind_medium_s42

  # gain 전체
  python src/plot_pilot_timeseries.py --pattern "gain"

  # 외란유형 전체 비교 (각 유형에서 medium_s42)
  python src/plot_pilot_timeseries.py --run-id payload_medium_s42 wind_medium_s42 gain_medium_s42
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stability_metrics import StabilityThresholds

PILOT_DIR  = Path("data/raw/pilot")
OUTPUT_DIR = Path("results/plots")
TARGET_Z   = 1.0
WARMUP_SEC = 1.5
H          = 2.0


def find_csv(run_id: str) -> Path:
    p = PILOT_DIR / f"{run_id}.csv"
    if not p.exists():
        raise FileNotFoundError(f"CSV 없음: {p}")
    return p


def find_t_fail_simple(df: pd.DataFrame, dtype: str,
                       thr: StabilityThresholds) -> float | None:
    """
    1차 신호가 연속 3회 발화한 첫 시점을 반환 (그림용 간이 버전).
    check_label_dist.py의 슬라이딩 윈도우 기반과 동일한 로직.
    """
    from stability_metrics import (altitude_rmse, attitude_max_angle,
                                   angular_rate_rms, convergence_failure,
                                   crash_indicator)

    t_arr   = df["t"].to_numpy()
    fs      = 240.0
    win_n   = int(2.0 * fs)
    slide_n = int(0.5 * fs)
    consec  = 0

    for end in range(win_n, len(df) + 1, slide_n):
        start   = end - win_n
        t_start = t_arr[start]
        t_end   = t_arr[end - 1]
        if t_start < WARMUP_SEC:
            continue

        sl = slice(start, end)
        z     = df["z"].to_numpy()[sl]
        roll  = df["roll"].to_numpy()[sl]
        pitch = df["pitch"].to_numpy()[sl]
        wx    = df["wx"].to_numpy()[sl]
        wy    = df["wy"].to_numpy()[sl]
        wz    = df["wz"].to_numpy()[sl]
        cont  = df["contact"].to_numpy()[sl] if "contact" in df.columns else None

        if dtype == "payload":
            fired = altitude_rmse(z, TARGET_Z) > thr.alt_rmse
        elif dtype == "wind":
            fired = (attitude_max_angle(roll, pitch) > thr.tilt_max or
                     angular_rate_rms(wx, wy, wz)    > thr.ang_rate_rms)
        elif dtype == "gain":
            fired = convergence_failure(z, TARGET_Z, thr.conv_tol)
        else:
            fired = False

        consec = consec + 1 if fired else 0
        if consec >= 3:
            return float(t_end)

    return None


def plot_run(run_id: str, ax_list, thr: StabilityThresholds, color: str = "steelblue"):
    """하나의 run을 4개 서브플롯(고도·roll·pitch·각속도)에 그린다."""
    csv_path = find_csv(run_id)
    df       = pd.read_csv(csv_path)
    dtype    = df["disturbance_type"].iloc[0] if "disturbance_type" in df.columns else "unknown"
    t_fail   = find_t_fail_simple(df, dtype, thr)

    t    = df["t"].to_numpy()
    z    = df["z"].to_numpy()
    roll = np.degrees(df["roll"].to_numpy())
    pitch= np.degrees(df["pitch"].to_numpy())
    ang  = np.sqrt(df["wx"]**2 + df["wy"]**2 + df["wz"]**2).to_numpy()

    crashed = bool(df["crashed"].iloc[-1])
    title   = f"{run_id}  |  crashed={crashed}  |  t_fail={t_fail:.2f}s" if t_fail else \
              f"{run_id}  |  crashed={crashed}  |  t_fail=None"

    ax_z, ax_roll, ax_pitch, ax_ang = ax_list

    # 배경 영역 표시
    for ax in ax_list:
        ax.axvspan(0, WARMUP_SEC, alpha=0.08, color="gray", label="warmup" if ax == ax_z else "")
        if t_fail:
            ax.axvspan(t_fail - H, t_fail, alpha=0.12, color="orange", label="양성 구간" if ax == ax_z else "")
            ax.axvline(t_fail, color="red", lw=1.5, ls="--", label="t_fail" if ax == ax_z else "")

    # 신호별 임계선
    ax_z.axhline(TARGET_Z - thr.alt_rmse, color="red", lw=0.8, ls=":", alpha=0.6)
    ax_ang.axhline(thr.ang_rate_rms, color="red", lw=0.8, ls=":", alpha=0.6, label=f"R3 임계={thr.ang_rate_rms}")

    ax_z.plot(t, z,     color=color, lw=1.0)
    ax_z.set_ylabel("고도 z [m]")
    ax_z.set_title(title, fontsize=9)
    ax_z.legend(fontsize=7, loc="upper right")

    ax_roll.plot(t, roll,  color=color, lw=1.0)
    ax_roll.set_ylabel("roll [도]")
    ax_roll.axhline( np.degrees(thr.tilt_max), color="red", lw=0.8, ls=":", alpha=0.6)
    ax_roll.axhline(-np.degrees(thr.tilt_max), color="red", lw=0.8, ls=":", alpha=0.6)

    ax_pitch.plot(t, pitch, color=color, lw=1.0)
    ax_pitch.set_ylabel("pitch [도]")
    ax_pitch.axhline( np.degrees(thr.tilt_max), color="red", lw=0.8, ls=":", alpha=0.6)
    ax_pitch.axhline(-np.degrees(thr.tilt_max), color="red", lw=0.8, ls=":", alpha=0.6)

    ax_ang.plot(t, ang, color=color, lw=1.0)
    ax_ang.set_ylabel("각속도 크기 [rad/s]")
    ax_ang.set_xlabel("시간 [s]")
    ax_ang.legend(fontsize=7, loc="upper right")

    for ax in ax_list:
        ax.set_xlim(left=0)
        ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="파일럿 run 시계열 시각화")
    parser.add_argument("--run-id",  nargs="+", default=None,
                        help="그릴 run_id 목록 (예: wind_medium_s42 wind_strong_s42)")
    parser.add_argument("--pattern", default=None,
                        help="파일명 패턴 (예: wind_medium → wind_medium_*.csv 전체)")
    parser.add_argument("--pilot-dir", default="data/raw/pilot")
    parser.add_argument("--out-dir",   default="results/plots")
    args = parser.parse_args()

    global PILOT_DIR, OUTPUT_DIR
    PILOT_DIR  = Path(args.pilot_dir)
    OUTPUT_DIR = Path(args.out_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    thr = StabilityThresholds()

    # 그릴 run 목록 수집
    if args.run_id:
        run_ids = args.run_id
    elif args.pattern:
        csvs   = sorted(PILOT_DIR.glob(f"*{args.pattern}*.csv"))
        run_ids = [f.stem for f in csvs if f.stem != "run_summary"]
    else:
        # 기본: wind_medium 3개
        run_ids = [f.stem for f in sorted(PILOT_DIR.glob("wind_medium_*.csv"))]

    if not run_ids:
        print(f"해당 run이 없음. pilot_dir={PILOT_DIR}")
        return

    colors = ["steelblue", "darkorange", "seagreen",
              "crimson", "purple", "saddlebrown", "teal"]

    # run당 4개 서브플롯
    n = len(run_ids)
    fig = plt.figure(figsize=(14, 4 * n))
    gs  = gridspec.GridSpec(4 * n, 1, hspace=0.15)

    for i, run_id in enumerate(run_ids):
        try:
            axes = [fig.add_subplot(gs[4*i + j]) for j in range(4)]
            plot_run(run_id, axes, thr, color=colors[i % len(colors)])
            print(f"  그림: {run_id}")
        except FileNotFoundError as e:
            print(f"  건너뜀: {e}")

    out_name = args.pattern or "_".join(run_ids[:2])
    out_path = OUTPUT_DIR / f"pilot_timeseries_{out_name}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
