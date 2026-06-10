"""
label_windows.py — Stage 2 본체: 본 생성 데이터에 조기경보 라벨 부여.

입력 : configs/workloads_main.csv + data/raw/main/*.csv
출력 : data/processed/labels.parquet (학습 데이터셋)

라벨 규칙 (window_end 기준, warmup 1.5s 제외):
  window_end ≤ t_fail − H  → 음성(0)
  t_fail − H < window_end ≤ t_fail  → 양성(1)
  window_end > t_fail      → 학습 제외
  t_fail 없는 run(끝까지 안정) → 전체 음성

t_fail 정의:
  1차 신호(R1)가 CONSECUTIVE_FIRES=3회 연속 임계 돌파한 첫 window_end 시점.
  R5(crash)는 t_fail 정의 아님 — 사후 검증용.

S2 시나리오 주의:
  R1(altitude_rmse)을 고정 1.0m 기준이 아니라 시점별 target_z 기준으로 계산.
  run CSV에 target_z 컬럼이 있으므로 매 윈도우에서 실제 target을 읽어 사용.

실행:
  python src/label_windows.py
  python src/label_windows.py --main-dir data/raw/main --output data/processed/labels.parquet
  python src/label_windows.py --limit 10  # 처음 N개 run만 (빠른 검증용)
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
    vibration_ratio, crash_indicator, convergence_failure,
    StabilityThresholds,
)

# ── 상수 ──────────────────────────────────────────────────────────────────────
WARMUP_SEC        = 1.5
CONSECUTIVE_FIRES = 3       # 연속 발화 횟수 (0.5s × 3 = 1.5초 연속이어야 t_fail 확정)
H                 = 2.0     # 조기경보 horizon [s]
WINDOW_SEC        = 2.0     # 슬라이딩 윈도우 길이 [s]
SLIDE_SEC         = 0.5     # 슬라이딩 간격 [s]
FS_DEFAULT        = 240.0   # 기본 샘플링 레이트 [Hz]


def _sliding_windows(df: pd.DataFrame, window_sec: float, slide_sec: float):
    """슬라이딩 윈도우 인덱스 생성기. (start_idx, end_idx, t_start, t_end) yield."""
    t = df["t"].to_numpy()
    fs = 1.0 / np.median(np.diff(t)) if len(t) > 1 else FS_DEFAULT
    win_n   = max(8, int(round(window_sec * fs)))
    slide_n = max(1, int(round(slide_sec * fs)))
    n = len(df)
    for end in range(win_n, n + 1, slide_n):
        start = end - win_n
        yield start, end, float(t[start]), float(t[end - 1])


def _window_r1(df: pd.DataFrame, start: int, end: int, scenario: str,
               thr: StabilityThresholds) -> float:
    """윈도우 R1 값 계산. S2는 tracking error 기반 (target_z 컬럼 활용)."""
    sl = slice(start, end)
    z = df["z"].to_numpy()[sl]

    if scenario == "S2" and "target_z" in df.columns:
        # S2: 시점별 목표 고도와의 오차 RMSE
        target_z_seq = df["target_z"].to_numpy()[sl]
        diff = z - target_z_seq
        return float(np.sqrt(np.mean(diff ** 2)))
    else:
        # S1: 고정 목표 1.0m 기준 (target_z 컬럼의 실제 값 사용)
        if "target_z" in df.columns:
            tgt = float(df["target_z"].iloc[start])
        else:
            tgt = 1.0
        return altitude_rmse(z, tgt)


def _window_features(df: pd.DataFrame, start: int, end: int,
                     scenario: str, thr: StabilityThresholds) -> dict:
    """윈도우 하나에서 R1~R6 feature 값과 발화 여부 반환."""
    sl = slice(start, end)
    z       = df["z"].to_numpy()[sl]
    roll    = df["roll"].to_numpy()[sl]
    pitch   = df["pitch"].to_numpy()[sl]
    wx      = df["wx"].to_numpy()[sl]
    wy      = df["wy"].to_numpy()[sl]
    wz      = df["wz"].to_numpy()[sl]
    contact = df["contact"].to_numpy()[sl] if "contact" in df.columns else None
    t_sl    = df["t"].to_numpy()[sl]

    fs = 1.0 / np.median(np.diff(t_sl)) if len(t_sl) > 1 else FS_DEFAULT

    r1_val = _window_r1(df, start, end, scenario, thr)
    r2_val = attitude_max_angle(roll, pitch)
    r3_val = angular_rate_rms(wx, wy, wz)
    r4_val = vibration_ratio(roll, fs)
    r5_val = crash_indicator(z, contact, thr.crash_floor)
    r6_val = convergence_failure(z,
                float(df["target_z"].iloc[start]) if "target_z" in df.columns else 1.0,
                thr.conv_tol)

    return {
        "R1": r1_val > thr.alt_rmse,
        "R2": r2_val > thr.tilt_max,
        "R3": r3_val > thr.ang_rate_rms,
        "R4": r4_val > thr.vib_ratio,
        "R5": bool(r5_val),
        "R6": bool(r6_val),
        "alt_rmse_val":    r1_val,
        "tilt_max_val":    r2_val,
        "ang_rate_rms_val": r3_val,
        "vib_ratio_val":   r4_val,
        "crash_val":       float(r5_val),
        "conv_fail_val":   float(r6_val),
    }


def find_t_fail(df: pd.DataFrame, scenario: str,
                thr: StabilityThresholds) -> float | None:
    """
    1차 신호(R1)가 CONSECUTIVE_FIRES 연속 돌파한 첫 window_end를 t_fail로 반환.
    warmup 구간 제외. 없으면 None.
    """
    consec = 0
    for start, end, t_start, t_end in _sliding_windows(df, WINDOW_SEC, SLIDE_SEC):
        if t_start < WARMUP_SEC:
            consec = 0
            continue
        r1_val = _window_r1(df, start, end, scenario, thr)
        fired  = r1_val > thr.alt_rmse

        if fired:
            consec += 1
            if consec >= CONSECUTIVE_FIRES:
                return float(t_end)
        else:
            consec = 0
    return None


def label_run(df: pd.DataFrame, run_meta: dict,
              thr: StabilityThresholds) -> list[dict]:
    """
    run 하나에서 모든 윈도우를 추출하고 라벨을 부여한다.
    window_end 기준 라벨 규칙 적용.
    """
    scenario = run_meta.get("scenario", "S1")
    t_fail   = find_t_fail(df, scenario, thr)

    records = []
    for start, end, t_start, t_end in _sliding_windows(df, WINDOW_SEC, SLIDE_SEC):

        # 라벨 결정
        if t_fail is None:
            label = 0   # 끝까지 안정 run → 전체 음성
        elif t_end <= t_fail - H:
            label = 0   # 정상 구간
        elif t_end <= t_fail:
            label = 1   # 위험 임박 (양성)
        else:
            label = -1  # t_fail 이후 → 제외

        feats = _window_features(df, start, end, scenario, thr)

        rec = {
            # 위치 정보
            "run_id":       run_meta["run_id"],
            "window_start": t_start,
            "window_end":   t_end,
            "label":        label,
            "t_fail":       t_fail,
            # 메타 (train/test 분할 및 해석용)
            "payload_factor": run_meta.get("payload_factor"),
            "payload_start":  run_meta.get("payload_start"),
            "payload_ramp":   run_meta.get("payload_ramp"),
            "scenario":       scenario,
            "seed":           run_meta.get("seed"),
            "crashed":        bool(df["crashed"].iloc[-1])
                              if "crashed" in df.columns else None,
            # feature 값
            "alt_rmse_val":    feats["alt_rmse_val"],
            "tilt_max_val":    feats["tilt_max_val"],
            "ang_rate_rms_val": feats["ang_rate_rms_val"],
            "vib_ratio_val":   feats["vib_ratio_val"],
            "crash_val":       feats["crash_val"],
            "conv_fail_val":   feats["conv_fail_val"],
            # 발화 여부 (디버깅·해석용)
            "R1": feats["R1"], "R2": feats["R2"], "R3": feats["R3"],
            "R4": feats["R4"], "R5": feats["R5"], "R6": feats["R6"],
        }
        records.append(rec)

    return records


def build_dataset(main_dir: Path, workloads_csv: Path,
                  limit: int | None = None) -> pd.DataFrame:
    """workloads_main.csv의 모든 run에 라벨 부여 → DataFrame 반환."""
    thr  = StabilityThresholds()
    meta = pd.read_csv(workloads_csv)
    if limit:
        meta = meta.head(limit)

    all_records = []
    n_ok = n_err = 0

    for _, row in meta.iterrows():
        run_id = row["run_id"]
        csv_path = main_dir / f"{run_id}.csv"

        if not csv_path.exists():
            print(f"  [SKIP] {run_id}: CSV 없음")
            n_err += 1
            continue

        try:
            df = pd.read_csv(csv_path)
            records = label_run(df, row.to_dict(), thr)
            all_records.extend(records)
            n_ok += 1
        except Exception as e:
            print(f"  [ERROR] {run_id}: {e}")
            n_err += 1

    print(f"\n라벨링 완료: {n_ok}개 성공, {n_err}개 실패")
    return pd.DataFrame(all_records)


def print_report(df: pd.DataFrame):
    """라벨 분포 리포트 출력."""
    total   = len(df)
    labeled = df[df["label"] >= 0]
    pos     = (labeled["label"] == 1).sum()
    neg     = (labeled["label"] == 0).sum()
    excl    = (df["label"] == -1).sum()

    print("\n" + "=" * 60)
    print("라벨 분포 리포트")
    print("=" * 60)
    print(f"전체 윈도우:  {total:,}")
    print(f"  양성(1):    {pos:,}  ({100*pos/total:.1f}%)")
    print(f"  음성(0):    {neg:,}  ({100*neg/total:.1f}%)")
    print(f"  제외(-1):   {excl:,} ({100*excl/total:.1f}%)")
    if pos > 0:
        print(f"  음성:양성 = {neg/pos:.1f}:1")

    print("\n[scenario별 양성/음성]")
    for sc in sorted(df["scenario"].dropna().unique()):
        sub = df[df["scenario"] == sc]
        p = (sub["label"] == 1).sum()
        n = (sub["label"] == 0).sum()
        print(f"  {sc}: 양성={p:,}, 음성={n:,}")

    print("\n[payload_factor별 t_fail 분포]")
    t_fail_df = (df[df["t_fail"].notna()]
                 .drop_duplicates("run_id")[["payload_factor", "t_fail"]]
                 .groupby("payload_factor")["t_fail"]
                 .describe()[["count","mean","min","max"]])
    print(t_fail_df.to_string())

    print("\n[lead time 분포 (t_fail이 있는 run의 첫 양성 윈도우까지)]")
    runs_with_pos = (df[df["label"] == 1]
                     .groupby("run_id")["window_end"].min()
                     .reset_index()
                     .rename(columns={"window_end": "first_alarm_t"}))
    t_fail_per_run = (df.drop_duplicates("run_id")[["run_id","t_fail"]]
                      .dropna())
    lead_df = runs_with_pos.merge(t_fail_per_run, on="run_id")
    lead_df["lead_time"] = lead_df["t_fail"] - lead_df["first_alarm_t"]
    print(lead_df["lead_time"].describe().to_string())


def main():
    parser = argparse.ArgumentParser(description="본 생성 데이터 조기경보 라벨 부여")
    parser.add_argument("--main-dir",  default="data/raw/main")
    parser.add_argument("--workloads", default="configs/workloads_main.csv")
    parser.add_argument("--output",    default="data/processed/labels.parquet")
    parser.add_argument("--limit",     type=int, default=None,
                        help="처음 N개 run만 처리 (빠른 검증용)")
    args = parser.parse_args()

    main_dir     = Path(args.main_dir)
    workloads    = Path(args.workloads)
    output_path  = Path(args.output)

    print(f"입력: {main_dir} / {workloads}")
    df = build_dataset(main_dir, workloads, limit=args.limit)

    print_report(df)

    # 학습에 쓸 행만 저장 (label >= 0)
    df_train = df[df["label"] >= 0].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_train.to_parquet(output_path, index=False)

    # 전체(제외 포함)도 따로 저장 (디버깅용)
    full_path = output_path.with_name("labels_full.parquet")
    df.to_parquet(full_path, index=False)

    print(f"\n저장:")
    print(f"  학습용 (label≥0): {output_path}  ({len(df_train):,}행)")
    print(f"  전체 (제외 포함): {full_path}  ({len(df):,}행)")


if __name__ == "__main__":
    main()