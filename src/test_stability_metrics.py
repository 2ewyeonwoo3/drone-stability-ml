"""
test_stability_metrics.py — R1~R6 안정성 지표 함수에 대한 기본 검증 테스트
"""

from __future__ import annotations

import numpy as np

from stability_metrics import (
    altitude_rmse, attitude_max_angle, angular_rate_rms, vibration_ratio,
    crash_indicator, convergence_failure, evaluate_window, StabilityThresholds,
)


def _check(name, cond):
    """테스트 결과를 PASS/FAIL 형태로 출력한다."""
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    return cond


def run():
    """R1~R6 안정성 지표에 대한 기본 동작 검증을 수행한다."""
    ok = True
    fs = 240.0
    n = 240
    t = np.arange(n) / fs

    # R1: 완벽하게 목표 고도를 유지하면 오차는 0
    # 일정한 오프셋이 있으면 RMSE는 해당 오프셋과 같아야 함
    ok &= _check("R1 zero error", abs(altitude_rmse(np.ones(n), 1.0)) < 1e-9)
    ok &= _check("R1 constant 0.3 offset", abs(altitude_rmse(np.ones(n) * 1.3, 1.0) - 0.3) < 1e-9)

    # R2: 기울기 크기는 sqrt(roll² + pitch²)의 최대값과 같아야 함
    roll = np.zeros(n); pitch = np.zeros(n); pitch[10] = 0.3
    ok &= _check("R2 max tilt = 0.3", abs(attitude_max_angle(roll, pitch) - 0.3) < 1e-9)

    # R3: 크기가 일정한 각속도 벡터의 RMS 계산 검증
    wx = np.ones(n) * 3.0; wy = np.zeros(n); wz = np.zeros(n)
    ok &= _check("R3 rms = 3.0", abs(angular_rate_rms(wx, wy, wz) - 3.0) < 1e-9)

    # R4:순수한 5Hz 신호는 대부분의 에너지가 1~20Hz 대역 안에 존재해야 함
    tone = np.sin(2 * np.pi * 5 * t)
    ratio_in = vibration_ratio(tone, fs, 1.0, 20.0)
    ok &= _check("R4 5Hz tone in-band > 0.9", ratio_in > 0.9)
    # 50Hz 신호는 대부분의 에너지가 1~20Hz 대역 밖에 존재해야 함
    tone_hi = np.sin(2 * np.pi * 50 * t)
    ratio_out = vibration_ratio(tone_hi, fs, 1.0, 20.0)
    ok &= _check("R4 50Hz tone in-band < 0.1", ratio_out < 0.1)

    # R5: 지면 근접 및 접촉 여부에 따른 충돌 판정 검증
    ok &= _check("R5 floor trigger", crash_indicator(np.array([1.0, 0.5, 0.05]), None, 0.10) is True)
    ok &= _check("R5 no trigger", crash_indicator(np.array([1.0, 0.9, 0.8]), None, 0.10) is False)
    ok &= _check("R5 contact trigger",
                 crash_indicator(np.ones(n), np.array([0] * (n - 1) + [1]), 0.10) is True)

    # R6: 오차가 크고 계속 증가하면 실패로 판단 오차가 작으면 실패하지 않아야 함
    growing = 1.0 - np.linspace(0, 0.6, n)  # altitude decaying from 1.0
    ok &= _check("R6 growing error fires", convergence_failure(growing, 1.0, 0.20) is True)
    ok &= _check("R6 tight hold no fire", convergence_failure(np.ones(n) * 1.01, 1.0, 0.20) is False)

    # 오차가 크더라도 점점 감소하는 경우
    # (즉, 수렴 중인 경우) 실패로 판단하면 안 됨
    shrinking = 1.0 - np.linspace(0.6, 0.0, n)
    ok &= _check("R6 shrinking error no fire", convergence_failure(shrinking, 1.0, 0.20) is False)

    # 깨끗한 Hover(호버링) 상태에 대해 evaluate_window 통합 평가 수행
    window = {
        "t": t, "z": np.ones(n), "roll": np.zeros(n), "pitch": np.zeros(n),
        "wx": np.zeros(n), "wy": np.zeros(n), "wz": np.zeros(n),
        "contact": np.zeros(n),
    }
    res = evaluate_window(window, 1.0, StabilityThresholds())
    # 정상 Hover 상태는 unstable=False, severity=0 이어야 함
    ok &= _check("evaluate_window clean hover -> stable", res["unstable"] is False and res["severity"] == 0)

    print("\nALL PASS" if ok else "\nSOME FAILED")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
