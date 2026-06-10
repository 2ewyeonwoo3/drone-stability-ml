"""
check_label_dist.py — Stage 2-A 파일럿 라벨 분포 점검.

각 run을 세 부류로 분류:
  (i)  good    : 양성 + 음성 둘 다 있음 (조기경보 학습 핵심)
  (ii) stable  : 음성만 (외란이 너무 약함)
  (iii) bad    : 양성만 or t_fail이 warmup 직후 (외란이 너무 강함 or 즉사)

t_fail 정의:
  해당 run군 1차 신호가 CONSECUTIVE_FIRES=3회 연속 임계 돌파한 첫 시점.
  warmup(1.5s) 이전은 탐지에서 제외.

라벨 규칙 (window_end 기준):
  window_end <= t_fail - H  → 음성(0)
  t_fail - H < window_end <= t_fail → 양성(1)
  window_end > t_fail       → 제외

실행:
  python src/check_label_dist.py
  python src/check_label_dist.py --pilot-dir data/raw/pilot --window 2.0 --slide 0.5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stability_metrics import (
    altitude_rmse, attitude_max_angle, angular_rate_rms,
    convergence_failure, crash_indicator, StabilityThresholds,
)

# ── 상수 ──────────────────────────────────────────────────────────────────────
WARMUP_SEC       = 1.5
CONSECUTIVE_FIRES = 3    # 연속 발화 횟수 (0.5s × 3 = 1.5초 연속이어야 t_fail 확정)
H                = 2.0   # 조기경보 horizon [s]
TARGET_Z         = 1.0

# 외란유형별 1차 신호
PRIMARY_SIGNAL = {
    "payload": "R1",
    "wind":    "R2",  # R2 or R3 중 먼저 뜨는 것
    "gain":    "R6",
}


def _rolling_windows(df: pd.DataFrame, window_sec: float, slide_sec: float,
                     fs: float = 240.0):
    """슬라이딩 윈도우 인덱스 생성기. (start_idx, end_idx, t_start, t_end) yield."""
    win_n   = max(8, int(round(window_sec * fs)))
    slide_n = max(1, int(round(slide_sec  * fs)))
    n = len(df)
    for end in range(win_n, n + 1, slide_n):
        start = end - win_n
        yield start, end, float(df["t"].iloc[start]), float(df["t"].iloc[end - 1])


def _window_signals(df: pd.DataFrame, start: int, end: int,
                    thr: StabilityThresholds) -> dict[str, bool]:
    """윈도우 하나에서 R1~R6 발화 여부 반환."""
    sl = slice(start, end)
    z      = df["z"].to_numpy()[sl]
    roll   = df["roll"].to_numpy()[sl]
    pitch  = df["pitch"].to_numpy()[sl]
    wx     = df["wx"].to_numpy()[sl]
    wy     = df["wy"].to_numpy()[sl]
    wz     = df["wz"].to_numpy()[sl]
    contact = df["contact"].to_numpy()[sl] if "contact" in df.columns else None
    t      = df["t"].to_numpy()[sl]

    fs = 240.0
    if len(t) >= 2:
        dt = np.median(np.diff(t))
        if dt > 0:
            fs = 1.0 / dt

    from stability_metrics import vibration_ratio
    r1 = altitude_rmse(z, TARGET_Z)          > thr.alt_rmse
    r2 = attitude_max_angle(roll, pitch)      > thr.tilt_max
    r3 = angular_rate_rms(wx, wy, wz)        > thr.ang_rate_rms
    r4 = vibration_ratio(roll, fs)            > thr.vib_ratio
    r5 = crash_indicator(z, contact, thr.crash_floor)
    r6 = convergence_failure(z, TARGET_Z, thr.conv_tol)
    return {"R1": r1, "R2": r2, "R3": r3, "R4": r4, "R5": bool(r5), "R6": bool(r6)}


def find_t_fail(df: pd.DataFrame, dtype: str,
                window_sec: float, slide_sec: float,
                thr: StabilityThresholds) -> float | None:
    """
    1차 신호가 CONSECUTIVE_FIRES 연속 돌파한 첫 t_end를 t_fail로 반환.
    없으면 None.
    """
    primary = PRIMARY_SIGNAL.get(dtype, "R1")
    # wind는 R2와 R3 중 하나라도 뜨면 발화로 처리
    consecutive = 0
    for start, end, t_start, t_end in _rolling_windows(df, window_sec, slide_sec):
        if t_start < WARMUP_SEC:
            continue
        signals = _window_signals(df, start, end, thr)
        # wind는 R2 or R3
        if dtype == "wind":
            fired = signals["R2"] or signals["R3"]
        else:
            fired = signals[primary]

        if fired:
            consecutive += 1
            if consecutive >= CONSECUTIVE_FIRES:
                return t_end
        else:
            consecutive = 0
    return None


def label_run(df: pd.DataFrame, t_fail: float | None,
              window_sec: float, slide_sec: float) -> pd.DataFrame:
    """
    윈도우별 라벨 부여. window_end 기준:
      <= t_fail - H  → 0 (음성)
      t_fail-H ~ t_fail → 1 (양성)
      > t_fail       → -1 (제외)
    t_fail이 None이면 전체 음성.
    """
    records = []
    for start, end, t_start, t_end in _rolling_windows(df, window_sec, slide_sec):
        if t_fail is None:
            label = 0
        elif t_end <= t_fail - H:
            label = 0
        elif t_end <= t_fail:
            label = 1
        else:
            label = -1  # 제외
        records.append({"t_start": t_start, "t_end": t_end, "label": label})
    return pd.DataFrame(records)


def classify_run(label_df: pd.DataFrame) -> str:
    """run을 good / stable / bad 중 하나로 분류."""
    has_pos = (label_df["label"] == 1).any()
    has_neg = (label_df["label"] == 0).any()
    if has_pos and has_neg:
        return "good"
    if has_neg and not has_pos:
        return "stable"
    return "bad"   # 양성만 or 라벨 없음


def analyze_pilot(pilot_dir: Path, window_sec: float, slide_sec: float) -> pd.DataFrame:
    thr = StabilityThresholds()
    csv_files = sorted(pilot_dir.glob("*.csv"))
    # run_summary.csv 제외
    csv_files = [f for f in csv_files if f.name != "run_summary.csv"]

    records = []
    for fpath in csv_files:
        df = pd.read_csv(fpath)
        run_id = fpath.stem
        dtype  = df["disturbance_type"].iloc[0] if "disturbance_type" in df.columns else "unknown"

        t_fail   = find_t_fail(df, dtype, window_sec, slide_sec, thr)
        label_df = label_run(df, t_fail, window_sec, slide_sec)
        category = classify_run(label_df)

        n_pos    = (label_df["label"] == 1).sum()
        n_neg    = (label_df["label"] == 0).sum()
        n_excl   = (label_df["label"] == -1).sum()
        crashed  = bool(df["crashed"].iloc[-1]) if "crashed" in df.columns else None

        records.append({
            "run_id":   run_id,
            "dtype":    dtype,
            "level":    run_id.split("_")[1] if "_" in run_id else "?",
            "rows":     len(df),
            "t_fail":   round(t_fail, 3) if t_fail else None,
            "crashed":  crashed,
            "category": category,
            "n_pos":    n_pos,
            "n_neg":    n_neg,
            "n_excl":   n_excl,
        })

    return pd.DataFrame(records).sort_values(["dtype", "level", "run_id"])


def print_report(result: pd.DataFrame):
    print("\n" + "=" * 65)
    print("파일럿 라벨 분포 점검 리포트")
    print("=" * 65)

    # 외란유형별 요약
    print("\n[외란유형 × 강도 × category 집계]")
    summary = result.groupby(["dtype", "level", "category"]).size().unstack(
        fill_value=0).reindex(columns=["good", "stable", "bad"], fill_value=0)
    print(summary.to_string())

    # t_fail 분포
    print("\n[t_fail 분포 (warmup=1.5s, None=끝까지 안정)]")
    print(result[["run_id", "dtype", "level", "t_fail", "crashed",
                  "category", "n_neg", "n_pos", "n_excl"]].to_string(index=False))

    # 진단 메시지
    print("\n[파라미터 보정 가이드]")
    for dtype in ["payload", "wind", "gain"]:
        sub = result[result["dtype"] == dtype]
        n_good   = (sub["category"] == "good").sum()
        n_stable = (sub["category"] == "stable").sum()
        n_bad    = (sub["category"] == "bad").sum()
        print(f"\n  {dtype}:")
        print(f"    good={n_good}, stable={n_stable}, bad={n_bad}")
        if n_stable > n_good:
            print(f"    ⚠️  stable이 많음 → 강도 범위를 올려야 함 (peak 값 증가)")
        if n_bad > n_good:
            print(f"    ⚠️  bad가 많음 → 강도가 너무 강함 (peak 값 감소 or ramp 늘리기)")
        if n_good >= 2:
            print(f"    ✅ good run이 {n_good}개 — 이 강도 범위는 유효")


def main():
    parser = argparse.ArgumentParser(description="파일럿 라벨 분포 점검")
    parser.add_argument("--pilot-dir", default="data/raw/pilot")
    parser.add_argument("--window",    type=float, default=2.0)
    parser.add_argument("--slide",     type=float, default=0.5)
    parser.add_argument("--save",      default="data/processed/pilot_label_dist.csv")
    args = parser.parse_args()

    pilot_dir = Path(args.pilot_dir)
    result    = analyze_pilot(pilot_dir, args.window, args.slide)

    print_report(result)

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.save, index=False)
    print(f"\n결과 저장: {args.save}")


if __name__ == "__main__":
    main()