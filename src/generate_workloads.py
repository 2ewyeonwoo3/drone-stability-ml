"""
generate_workloads.py — Stage 2-A 파일럿용 조건 조합표 생성.

파일럿 목적: 외란 강도 범위가 의도한 궤적(점진 발산)을 만드는지 확인.
본 생성은 파일럿 범위 보정 후 확정.

출력: configs/workloads_pilot.csv
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd


# ── 파일럿 파라미터 범위 ────────────────────────────────────────────────────
# 각 수준은 (약, 중, 강) 3단계
# 약: 끝까지 안정 기대 / 중: 점진 발산 목표 / 강: 즉사 경계

PAYLOAD_PEAKS   = [1.8, 2.5, 3.2]   # mass_factor 최종값 (1.0에서 ramp)
WIND_PEAKS      = [0.5, 1.0, 1.5]   # wind_mag 최종값 [N] (0에서 ramp)
GAIN_FLOORS     = [0.6, 0.45, 0.30] # p_gain_mult 최종값 (1.0에서 하강 ramp)

SEEDS = [42, 7, 123]

# ramp 파라미터 (고정)
PAYLOAD_START   = 2.0   # warmup 이후 시작 [s]
PAYLOAD_RAMP    = 4.0   # ramp 지속 시간 [s]
WIND_START      = 2.0
WIND_RAMP       = 5.0
GAIN_START      = 2.0   # 게인 하강 시작 [s]
GAIN_RAMP       = 5.0   # 게인 하강 지속 [s]

DURATION        = 15.0  # run 길이 [s] — ramp 끝 후 붕괴 여유 충분히
OUTPUT_DIR      = Path("data/raw/pilot")


def _level_name(values: list, idx: int) -> str:
    return ["weak", "medium", "strong"][idx]


def build_payload_rows() -> list[dict]:
    rows = []
    for (pi, peak), seed in itertools.product(enumerate(PAYLOAD_PEAKS), SEEDS):
        run_id = f"payload_{_level_name(PAYLOAD_PEAKS, pi)}_s{seed}"
        rows.append({
            "run_id":          run_id,
            "disturbance_type": "payload",
            "level":           _level_name(PAYLOAD_PEAKS, pi),
            "seed":            seed,
            "duration":        DURATION,
            # payload 파라미터
            "payload_factor":  peak,
            "payload_start":   PAYLOAD_START,
            "payload_ramp":    PAYLOAD_RAMP,
            # wind 없음
            "wind_mode":       "none",
            "wind_peak":       0.0,
            "wind_start":      0.0,
            "wind_ramp":       0.0,
            # gain 정상
            "p_gain_mult_start": 1.0,
            "p_gain_mult_end":   1.0,
            "gain_start":      0.0,
            "gain_ramp":       0.0,
            # 출력 경로
            "output": str(OUTPUT_DIR / f"{run_id}.csv"),
        })
    return rows


def build_wind_rows() -> list[dict]:
    rows = []
    for (wi, peak), seed in itertools.product(enumerate(WIND_PEAKS), SEEDS):
        run_id = f"wind_{_level_name(WIND_PEAKS, wi)}_s{seed}"
        rows.append({
            "run_id":          run_id,
            "disturbance_type": "wind",
            "level":           _level_name(WIND_PEAKS, wi),
            "seed":            seed,
            "duration":        DURATION,
            # payload 없음
            "payload_factor":  1.0,
            "payload_start":   0.0,
            "payload_ramp":    0.0,
            # wind 파라미터
            "wind_mode":       "turbulent_ramp",  # RampedTurbulentWind 사용
            "wind_peak":       peak,
            "wind_start":      WIND_START,
            "wind_ramp":       WIND_RAMP,
            # gain 정상
            "p_gain_mult_start": 1.0,
            "p_gain_mult_end":   1.0,
            "gain_start":      0.0,
            "gain_ramp":       0.0,
            "output": str(OUTPUT_DIR / f"{run_id}.csv"),
        })
    return rows


def build_gain_rows() -> list[dict]:
    rows = []
    for (gi, floor), seed in itertools.product(enumerate(GAIN_FLOORS), SEEDS):
        run_id = f"gain_{_level_name(GAIN_FLOORS, gi)}_s{seed}"
        rows.append({
            "run_id":          run_id,
            "disturbance_type": "gain",
            "level":           _level_name(GAIN_FLOORS, gi),
            "seed":            seed,
            "duration":        DURATION,
            # payload 없음
            "payload_factor":  1.0,
            "payload_start":   0.0,
            "payload_ramp":    0.0,
            # wind 없음
            "wind_mode":       "none",
            "wind_peak":       0.0,
            "wind_start":      0.0,
            "wind_ramp":       0.0,
            # gain ramp 하강
            "p_gain_mult_start": 1.0,
            "p_gain_mult_end":   floor,
            "gain_start":      GAIN_START,
            "gain_ramp":       GAIN_RAMP,
            "output": str(OUTPUT_DIR / f"{run_id}.csv"),
        })
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = build_payload_rows() + build_wind_rows() + build_gain_rows()
    df = pd.DataFrame(rows)

    out = Path("configs/workloads_pilot.csv")
    df.to_csv(out, index=False)

    print(f"파일럿 조합표 생성 완료: {len(df)}행 → {out}")
    print(df.groupby("disturbance_type")["level"].value_counts().to_string())


if __name__ == "__main__":
    main()