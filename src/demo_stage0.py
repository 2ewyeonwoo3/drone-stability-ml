"""
demo_stage0.py — Stage 0이 처음부터 끝까지 정상 동작하는지 검증하는 데모
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from simulate import run_simulation
from stability_metrics import rolling_evaluate, StabilityThresholds


TARGET_Z = 1.0
# 드론은 z=0.1m에서 이륙을 시작한다.
# 초기 약 1.5초는 상승 및 자세 안정화 구간이며, 실제 안정성 평가 대상이 아니다.

# 따라서 이 구간은 조기 경고 분석에서 제외한다.
# 그렇지 않으면 이륙 과정 자체가 고도 오차(R1)를 유발하여 잘못된 경고가 발생할 수 있다.
WARMUP_SEC = 1.5
RULES = ["R1", "R2", "R3", "R4", "R5", "R6"]
RULE_NAMES = {
    "R1": "고도 RMSE",
    "R2": "기울기 각도",
    "R3": "각속도 RMS",
    "R4": "진동 비율",
    "R5": "충돌/접촉",
    "R6": "수렴 실패",
}


def first_fire_times(win_df: pd.DataFrame) -> dict:
    """
    각 안정성 규칙이 처음 발생한 시각을 반환한다.

    워밍업 구간(WARMUP_SEC)은 제외한 뒤,
    각 규칙이 True가 된 첫 번째 윈도우 종료 시각(t_end)을 기록한다.

    한 번도 발생하지 않은 규칙은 None 반환.
    """
    win_df = win_df[win_df["t_start"] >= WARMUP_SEC]
    out = {}
    for r in RULES:
        hits = win_df.loc[win_df[r], "t_end"]
        out[r] = float(hits.iloc[0]) if len(hits) else None
    return out


def summarize(label: str, rows, thr: StabilityThresholds):
    """
    하나의 시뮬레이션 결과를 요약한다.

    출력 내용:
      - 전체 Telemetry 행 수
      - 슬라이딩 윈도우 수
      - 최종 고도
      - 충돌 여부
      - 불안정 윈도우 개수
      - 주요 지표 최대값

    반환:
      df  : 원본 Telemetry DataFrame
      win : 슬라이딩 윈도우 평가 결과 DataFrame
    """
    df = pd.DataFrame(rows)
    win = rolling_evaluate(df, TARGET_Z, window_sec=0.5, stride_sec=0.1, thr=thr)
    post = win[win["t_start"] >= WARMUP_SEC]
    n_unstable = int(post["unstable"].sum())
    print(f"\n=== {label} ===")
    print(f"  sim rows: {len(df)}   windows: {len(win)} (post-warmup {len(post)})   "
          f"final z: {df['z'].iloc[-1]:.3f}   crashed: {bool(df['crashed'].iloc[-1])}")
    print(f"  unstable windows (post-warmup): {n_unstable}/{len(post)}")
    print(f"  peak metrics (post-warmup) -> alt_rmse {post['alt_rmse_val'].max():.3f} | "
          f"tilt {post['tilt_max_val'].max():.3f} | "
          f"angrate {post['ang_rate_rms_val'].max():.3f} | "
          f"vib {post['vib_ratio_val'].max():.3f}")
    return df, win


def main():
    thr = StabilityThresholds()

    # ---- 1. 안정 비행 기준 실험 ---------------------------------------------
    #
    # 외란 없이 기본 PID 설정으로 비행
    #
    # 정상적으로 호버링한다면
    # 대부분의 안정성 규칙이 발생하지 않아야 한다.
    # -------------------------------------------------------------------------
    
    stable_rows = run_simulation(
        p_gain_mult=1.0, duration_sec=8.0, seed=42, target_pos=(0, 0, TARGET_Z),
    )
    summarize("STABLE (nominal, no disturbance)", stable_rows, thr)

    # ---- 2. 점진적 붕괴 시나리오 ---------------------------------------------
    # 초기에는 정상 이륙. 이후 무거운 페이로드를 천천히 추가
    #
    # DSLPIDControl의 중력 보상(feed-forward)은 URDF에 정의된 "기본 질량"만 알고 있기 때문에,실제 질량 증가를 인식하지 못한다.
    #
    # 결과적으로 페이로드 증가 -> 드론이 조금씩 가라앉음 -> 적분 제어기가 뒤늦게 보상 ->  고도가 수 초 동안 서서히 감소 -> 결국 지면 접촉
    #
    # 특징:
    #
    #   - Roll/Pitch는 비교적 작음
    #   - 뒤집히는 것이 아니라 천천히 침강
    #   - 조기 경고 시간을 충분히 확보 가능
    #
    # 따라서 Lead Time 실험에 적합한 시나리오이다.
    # -------------------------------------------------------------------------
    collapse_rows = run_simulation(
        p_gain_mult=1.0, duration_sec=9.0, seed=7,
        target_pos=(0, 0, TARGET_Z),
        payload_factor=2.5, payload_start=2.0, payload_ramp=3.0,
    )
    df, win = summarize("COLLAPSING (payload ramp -> slow altitude sink)",
                        collapse_rows, thr)

    # ---- Lead Time 분석 ---------------------------------------------------
    # 각 규칙이 처음 발생한 시각을 계산한 뒤, 실제 충돌(R5) 시각과 비교한다.
    # 충돌 전에 먼저 발생한 규칙들은 조기 경고 신호(Leading Indicator)로 간주한다.
    # -------------------------------------------------------------------------
    fires = first_fire_times(win)
    crash_t = fires["R5"]
    print("\n  --- 조기 경고 Lead Time 분석 (붕괴 시나리오) ---")
    for r in RULES:
        ft = fires[r]
        if ft is None:
            print(f"    {r} {RULE_NAMES[r]:<16}: never fired")
            continue
        if r == "R5":
            print(f"    {r} {RULE_NAMES[r]:<16}: first at t={ft:.2f}s  (the EVENT)")
        elif crash_t is not None:
            lead = crash_t - ft
            print(f"    {r} {RULE_NAMES[r]:<16}: first at t={ft:.2f}s  "
                  f"-> lead time {lead:+.2f}s before crash")
        else:
            print(f"    {r} {RULE_NAMES[r]:<16}: first at t={ft:.2f}s")

    if crash_t is not None:
        leading = [r for r in ["R1", "R2", "R3", "R4", "R6"]
                   if fires[r] is not None and fires[r] < crash_t]
        if leading:
            earliest = min(fires[r] for r in leading)
            print(
                f"\n  가장 빠른 경고 발생 시각: {earliest:.2f}s\n"
                f"  실제 충돌 시각: {crash_t:.2f}s\n"
                f"  확보 가능한 Lead Time: {crash_t - earliest:.2f}s\n\n"
                f"  이 시간 차이가 바로 Stage 2에서 머신러닝 모델이 예측하려는 대상이다."
            )
        else:
            print(
                "\n  충돌 이전에 발생한 조기 경고 신호가 없음.\n"
                "  현재 외란이 너무 급격하게 적용되고 있음.\n"
                "  wind_mag를 줄이거나 payload_ramp를 늘려 더 천천히 붕괴하도록 조정해야 한다."
            )
    else:
        print(
            "\n  드론이 충돌하지 않음.\n"
            "  payload_factor 또는 wind_mag를 증가시켜 붕괴 상황을 만들어야 한다."
        )


if __name__ == "__main__":
    main()
