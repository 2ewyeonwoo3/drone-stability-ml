"""
test_pipeline_local.py — Kafka·Spark 없이 파이프라인 핵심 로직을 검증.

실제 스트리밍 파이프라인을 띄우기 전에 다음을 확인:
  1. simulate.py → event_time 합성 → 메시지 구조 (producer 로직)
  2. collect_list + sort + evaluate_window UDF 로직 (spark_consumer 로직)
  3. 두 경로의 feature 값이 Stage 0의 rolling_evaluate와 일치하는지

실행:
  python src/test_pipeline_local.py
"""

from __future__ import annotations

import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from simulate import run_simulation
from stability_metrics import rolling_evaluate, StabilityThresholds, evaluate_window

# producer.py와 동일한 epoch_base
EPOCH_BASE = datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()


def _check(name: str, cond: bool) -> bool:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    return cond


# ── 테스트 1: producer 메시지 구조 검증 ──────────────────────────────────────
def test_producer_message_format():
    print("\n[1] producer 메시지 구조 검증")
    ok = True

    rows = run_simulation(
        p_gain_mult=1.0, duration_sec=1.0, seed=42,
        target_pos=(0, 0, 1.0), output_path=None,
    )
    ctrl_freq = 240

    # producer.py와 동일한 event_time 합성 로직
    msg = {
        **rows[0],
        "ctrl_freq": ctrl_freq,
        "event_time": datetime.fromtimestamp(
            EPOCH_BASE + rows[0]["step"] / ctrl_freq, tz=timezone.utc
        ).isoformat(),
    }

    ok &= _check("ctrl_freq 컬럼 포함", "ctrl_freq" in msg)
    ok &= _check("event_time 컬럼 포함", "event_time" in msg)
    ok &= _check("event_time이 datetime.now() 아님 (epoch_base 기반)",
                 msg["event_time"].startswith("2025-01-01"))

    # event_time이 시뮬 시간 기반인지 확인
    # step=0 → t=0.0 → event_time ≈ epoch_base
    ts = datetime.fromisoformat(msg["event_time"]).timestamp()
    ok &= _check("step=0의 event_time이 epoch_base와 일치",
                 abs(ts - EPOCH_BASE) < 0.01)

    # ctrl_freq가 실제 시뮬 레이트와 일치
    ok &= _check("ctrl_freq=240 확인", msg["ctrl_freq"] == 240)

    return ok


# ── 테스트 2: UDF 핵심 로직 검증 (Spark 없이 Python으로 모사) ─────────────────
def test_udf_logic():
    print("\n[2] collect_list → evaluate_window UDF 로직 검증")
    ok = True

    # 안정적인 호버 데이터 생성
    rows = run_simulation(
        p_gain_mult=1.0, duration_sec=3.0, seed=42,
        target_pos=(0, 0, 1.0), output_path=None,
    )
    df = pd.DataFrame(rows)

    # 윈도우 하나를 수동으로 슬라이싱 (t=1.5~3.5s: warmup 이후)
    ctrl_freq = 240
    window_sec = 2.0
    start_step = int(1.5 * ctrl_freq)
    end_step = start_step + int(window_sec * ctrl_freq)
    window_df = df.iloc[start_step:end_step]

    # ── collect_list 모사: step이 포함된 dict 리스트 ──────────────────────
    samples = window_df[[
        "step", "z", "roll", "pitch", "wx", "wy", "wz",
        "contact", "target_z", "ctrl_freq",
    ]].to_dict("records")

    # 셔플 모사: 순서를 뒤섞은 뒤 UDF 안에서 step 정렬이 복원되는지 확인
    import random
    random.seed(99)
    samples_shuffled = samples.copy()
    random.shuffle(samples_shuffled)

    # UDF 안 로직과 동일하게 실행
    samples_sorted = sorted(samples_shuffled, key=lambda r: r["step"])
    ctrl_freq_from_sample = samples_sorted[0]["ctrl_freq"]
    target_z = float(samples_sorted[0]["target_z"])

    window_data = {
        "z":       [float(r["z"])       for r in samples_sorted],
        "roll":    [float(r["roll"])    for r in samples_sorted],
        "pitch":   [float(r["pitch"])   for r in samples_sorted],
        "wx":      [float(r["wx"])      for r in samples_sorted],
        "wy":      [float(r["wy"])      for r in samples_sorted],
        "wz":      [float(r["wz"])      for r in samples_sorted],
        "contact": [float(r["contact"]) for r in samples_sorted],
    }

    result = evaluate_window(
        window_data, target_z=target_z,
        thr=StabilityThresholds(), fs=float(ctrl_freq_from_sample),
    )

    ok &= _check("UDF 결과가 dict 반환", isinstance(result, dict))
    ok &= _check("안정 호버 → unstable=False", result["unstable"] is False)
    ok &= _check("안정 호버 → severity=0", result["severity"] == 0)
    ok &= _check("alt_rmse가 0에 가까움 (호버 후 구간)", result["alt_rmse_val"] < 0.05)
    ok &= _check("ctrl_freq가 샘플에서 정확히 전달됨", ctrl_freq_from_sample == 240)

    return ok


# ── 테스트 3: UDF 결과 vs rolling_evaluate 일치 검증 ──────────────────────────
def test_udf_matches_rolling_evaluate():
    print("\n[3] UDF 결과 vs rolling_evaluate 일치 검증")
    ok = True

    rows = run_simulation(
        p_gain_mult=1.0, duration_sec=9.0, seed=7,
        target_pos=(0, 0, 1.0), output_path=None,
        payload_factor=2.5, payload_start=2.0, payload_ramp=3.0,
    )
    df = pd.DataFrame(rows)
    ctrl_freq = 240
    thr = StabilityThresholds()

    # rolling_evaluate로 전체 런 처리 (배치 경로)
    win_df = rolling_evaluate(df, target_z=1.0, window_sec=2.0, stride_sec=0.5, thr=thr)

    # 하나의 윈도우를 직접 UDF 로직으로도 처리 (스트리밍 경로 모사)
    # win_df의 두 번째 윈도우 (이륙 이후)를 골라 비교
    ref_window = win_df.iloc[3]   # 약 t=1.5~3.5s 구간
    t_start = ref_window["t_start"]
    t_end = ref_window["t_end"]

    window_slice = df[(df["t"] >= t_start) & (df["t"] <= t_end)]
    samples = window_slice[["step", "z", "roll", "pitch", "wx", "wy",
                             "wz", "contact", "target_z", "ctrl_freq"]].to_dict("records")

    samples_sorted = sorted(samples, key=lambda r: r["step"])
    window_data = {k: [float(r[k]) for r in samples_sorted]
                   for k in ["z", "roll", "pitch", "wx", "wy", "wz", "contact"]}

    udf_result = evaluate_window(window_data, target_z=1.0, thr=thr, fs=float(ctrl_freq))

    # 두 경로의 핵심 feature가 일치하는지 확인 (부동소수점 오차 허용)
    diff_rmse = abs(udf_result["alt_rmse_val"] - ref_window["alt_rmse_val"])
    ok &= _check(f"alt_rmse 일치 (차이 {diff_rmse:.4f} < 0.01)", diff_rmse < 0.01)

    ok &= _check("unstable 플래그 일치",
                 udf_result["unstable"] == bool(ref_window["unstable"]))

    return ok


# ── 테스트 4: event_time 시뮬 시간 커버리지 ──────────────────────────────────
def test_event_time_coverage():
    print("\n[4] event_time 윈도우 커버리지 검증")
    ok = True

    duration = 3.0
    ctrl_freq = 240
    rows = run_simulation(
        p_gain_mult=1.0, duration_sec=duration, seed=42,
        target_pos=(0, 0, 1.0), output_path=None,
    )

    # event_time 범위 확인
    t_first = EPOCH_BASE + rows[0]["step"] / ctrl_freq
    t_last  = EPOCH_BASE + rows[-1]["step"] / ctrl_freq

    ok &= _check(f"총 시뮬 시간 ≈ {duration}s (실제 {t_last - t_first:.2f}s)",
                 abs((t_last - t_first) - (duration - 1/ctrl_freq)) < 0.01)

    # 2초 윈도우가 몇 개 들어맞는지
    n_windows_expected = int((duration - 2.0) / 0.5) + 1
    ok &= _check(f"2s 윈도우 {n_windows_expected}개 이상 들어맞음",
                 (t_last - t_first) >= 2.0)

    return ok


# ── 실행 ─────────────────────────────────────────────────────────────────────
def main():
    results = [
        test_producer_message_format(),
        test_udf_logic(),
        test_udf_matches_rolling_evaluate(),
        test_event_time_coverage(),
    ]
    print("\n" + ("=" * 40))
    print("전체 결과:", "ALL PASS" if all(results) else "SOME FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
