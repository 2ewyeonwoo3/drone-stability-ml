"""
stability_metrics.py — R1~R6 안정성 지표를 재사용 가능한 윈도우 함수로 구현
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# 임계값 설정
# (target_z ≈ 1.0m 환경에서 CF2X Hover 기준, 240Hz Telemetry)
#
# 현재는 기본값이며,
# Stage 2에서 실제 붕괴 데이터를 이용해 재조정 예정
# ---------------------------------------------------------------------------

@dataclass
class StabilityThresholds:
    # R1 고도 RMSE [m]
    alt_rmse: float = 0.15
    # R2 최대 기울기 각도 [rad] (약 17도)
    tilt_max: float = 0.30
    # R3 각속도 RMS [rad/s]
    ang_rate_rms: float = 2.0
    # R4 진동 에너지 비율
    # [vib_band_lo, vib_band_hi] 구간의 에너지 비중
    vib_ratio: float = 0.35
    vib_band_lo: float = 1.0
    vib_band_hi: float = 20.0
    # R5 충돌 판정
    # 고도 바닥값 또는 접촉 플래그 발생 시
    crash_floor: float = 0.10
    # R6 수렴 실패
    # 평균 위치 오차 허용 범위 [m]
    conv_tol: float = 0.20


# ---------------------------------------------------------------------------
# 보조 함수
# ---------------------------------------------------------------------------

def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


def _rms(x: np.ndarray) -> float:
    x = _arr(x)
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(x))))


def _infer_fs(t: Sequence[float], default: float = 240.0) -> float:
    """
    타임스탬프로부터 샘플링 주파수를 추정한다.
    추정이 불가능하면 기본값(default)을 사용한다.
    """
    t = _arr(t)
    if t.size < 2:
        return default
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        return default
    return 1.0 / dt


# --------------------------------------------------------------------------- #
# R1 — 고도 RMSE
# --------------------------------------------------------------------------- #

def altitude_rmse(z: Sequence[float], target_z: float) -> float:
    """
    목표 고도 대비 RMSE 계산.

    값이 증가할수록 목표 고도 유지 능력이 저하되고 있음을 의미한다.
    """
    return _rms(_arr(z) - target_z)


# --------------------------------------------------------------------------- #
# R2 — 자세(기울기) 각도
# --------------------------------------------------------------------------- #

def attitude_max_angle(roll: Sequence[float], pitch: Sequence[float]) -> float:
    """윈도우 내 최대 기울기 크기 계산.

    sqrt(roll² + pitch²)의 최대값을 반환한다.
    단위: rad
    """
    r = _arr(roll)
    p = _arr(pitch)
    if r.size == 0:
        return float("nan")
    tilt = np.sqrt(r * r + p * p)
    return float(np.max(tilt))


# --------------------------------------------------------------------------- #
# R3 — 각속도 RMS
# --------------------------------------------------------------------------- #

def angular_rate_rms(wx: Sequence[float], wy: Sequence[float],
                     wz: Sequence[float]) -> float:
    """
    윈도우 내 각속도 크기의 RMS 계산.

    기체가 흔들리거나 진동하기 시작하면
    평균 자세가 아직 정상이어도 각속도 에너지가 증가한다.

    따라서 이는 종종 가장 빠른 조기 경고 지표가 된다.
    """
    wx, wy, wz = _arr(wx), _arr(wy), _arr(wz)
    if wx.size == 0:
        return float("nan")
    mag = np.sqrt(wx * wx + wy * wy + wz * wz)
    return _rms(mag)


# --------------------------------------------------------------------------- #
# R4 — FFT 기반 진동 비율
# --------------------------------------------------------------------------- #

def vibration_ratio(signal: Sequence[float], fs: float,
                    band_lo: float = 1.0, band_hi: float = 20.0) -> float:
    """
    주파수 대역 내 진동 에너지 비율 계산.

    전체 스펙트럼 에너지 중
    [band_lo, band_hi] Hz 구간이 차지하는 비율을 반환한다.

    제어 진동이 증가하면
    몇 Hz 대역에 에너지가 집중되며
    진동이 지배적일수록 비율이 1에 가까워진다.

    의미 있는 FFT 계산이 어려울 정도로
    윈도우가 짧으면 NaN을 반환한다.
    """
    x = _arr(signal)
    n = x.size
    if n < 8:
        return float("nan")
    x = x - np.mean(x)              # 평균 제거 (DC 성분 제거)
    win = np.hanning(n)
    xf = np.fft.rfft(x * win)
    psd = np.abs(xf) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    total = float(np.sum(psd))
    if total <= 0:
        return 0.0
    band = (freqs >= band_lo) & (freqs <= band_hi)
    return float(np.sum(psd[band]) / total)


# --------------------------------------------------------------------------- #
# R5 — 충돌 / 지면 접촉
# --------------------------------------------------------------------------- #

def crash_indicator(z: Sequence[float], contact: Optional[Sequence[float]] = None,
                    floor: float = 0.10) -> bool:
    """
    충돌 여부 판정.

    다음 중 하나라도 만족하면 True:

      * 고도가 floor 이하
      * contact 플래그 발생

    Stage 2에서 조기 경고 모델이 예측하려는
    실제 이벤트(Event)에 해당한다.
    """
    z = _arr(z)
    if z.size and np.min(z) <= floor:
        return True
    if contact is not None:
        c = _arr(contact)
        if c.size and np.any(c > 0):
            return True
    return False


# --------------------------------------------------------------------------- #
# R6 — 수렴 실패
# --------------------------------------------------------------------------- #

def convergence_failure(z: Sequence[float], target_z: float,
                        tol: float = 0.20) -> bool:
    """
    수렴 실패 여부 판정.

    조건:

      1. 평균 고도 오차가 충분히 큼
        mean(|z - target|) > tol

      2. 오차가 감소하지 않음
        (오차 추세 기울기 >= 0)

    즉, 드론이 목표 상태에 수렴하지 못하거나
    오히려 발산하는 상황을 탐지한다.

    이는 단순 충돌(R5)과 구분되는 문제이다.
    """
    z = _arr(z)
    if z.size < 3:
        return False
    err = np.abs(z - target_z)
    if float(np.mean(err)) <= tol:
        return False
    # 오차 크기(|error|)의 추세선 기울기 계산
    idx = np.arange(err.size, dtype=float)
    slope = np.polyfit(idx, err, 1)[0]
    return bool(slope >= 0.0)


# --------------------------------------------------------------------------- #
# 하나의 윈도우 평가
# --------------------------------------------------------------------------- #

def evaluate_window(window: Dict[str, Sequence[float]], target_z: float,
                    thr: StabilityThresholds = StabilityThresholds(),
                    fs: Optional[float] = None) -> Dict[str, float]:
    """
    하나의 윈도우에 대해 R1~R6 계산.

    window는 다음 키를 가진 dict:

        z, roll, pitch, wx, wy, wz
        (선택적으로 t, contact 포함)

    반환값:

        *_val   → 실제 지표값
        R1~R6   → 규칙 위반 여부
        unstable → 하나라도 위반 시 True
        severity → 위반 규칙 개수
    """
    z = window["z"]
    if fs is None:
        fs = _infer_fs(window.get("t", []))

    r1 = altitude_rmse(z, target_z)
    r2 = attitude_max_angle(window["roll"], window["pitch"])
    r3 = angular_rate_rms(window["wx"], window["wy"], window["wz"])
    # Roll 신호 기반 진동 분석
    # 제어 진동은 일반적으로 Roll에서 가장 먼저 나타남
    r4 = vibration_ratio(window["roll"], fs, thr.vib_band_lo, thr.vib_band_hi)
    r5 = crash_indicator(z, window.get("contact"), thr.crash_floor)
    r6 = convergence_failure(z, target_z, thr.conv_tol)

    fired = {
        "R1": bool(np.isfinite(r1) and r1 > thr.alt_rmse),
        "R2": bool(np.isfinite(r2) and r2 > thr.tilt_max),
        "R3": bool(np.isfinite(r3) and r3 > thr.ang_rate_rms),
        "R4": bool(np.isfinite(r4) and r4 > thr.vib_ratio),
        "R5": bool(r5),
        "R6": bool(r6),
    }
    severity = int(sum(fired.values()))
    out = {
        "alt_rmse_val": r1,
        "tilt_max_val": r2,
        "ang_rate_rms_val": r3,
        "vib_ratio_val": r4,
        "crash_val": float(bool(r5)),
        "conv_fail_val": float(bool(r6)),
        **fired,
        "severity": severity,
        "unstable": severity > 0,
    }
    return out


# --------------------------------------------------------------------------- #
# 전체 비행 로그에 대한 슬라이딩 윈도우 평가
# --------------------------------------------------------------------------- #

def rolling_evaluate(df, target_z: float,
                     window_sec: float = 0.5, stride_sec: float = 0.1,
                     thr: StabilityThresholds = StabilityThresholds()):
    """
    Telemetry DataFrame 전체에 대해 슬라이딩 윈도우 평가 수행.

    반환값:

        윈도우 종료 시각
        R1~R6 지표
        규칙 위반 여부
        종합 결과

    를 포함하는 pandas DataFrame

    이는 Stage 1/3의 Spark Structured Streaming이
    슬라이딩 윈도우마다 생성할 결과와 동일한 형태를 목표로 한다.
    """
    import pandas as pd  # pandas 의존성을 지표 함수와 분리하기 위한 지역 import

    t = df["t"].to_numpy()
    fs = _infer_fs(t)
    win_n = max(8, int(round(window_sec * fs)))
    stride_n = max(1, int(round(stride_sec * fs)))

    cols = ["z", "roll", "pitch", "wx", "wy", "wz"]
    has_contact = "contact" in df.columns
    rows = []
    n = len(df)
    for end in range(win_n, n + 1, stride_n):
        start = end - win_n
        sl = slice(start, end)
        window = {c: df[c].to_numpy()[sl] for c in cols}
        window["t"] = t[sl]
        if has_contact:
            window["contact"] = df["contact"].to_numpy()[sl]
        res = evaluate_window(window, target_z, thr, fs=fs)
        res["t_end"] = float(t[end - 1])
        res["t_start"] = float(t[start])
        rows.append(res)

    return pd.DataFrame(rows)
