"""
generate_workloads.py — Stage 2-A 파일럿용 조건 조합표 생성.

파일럿 목적: 외란 강도 범위가 의도한 궤적(점진 발산)을 만드는지 확인.
본 생성은 파일럿 범위 보정 후 확정.

강도 3단계(weak/medium/strong)는 '기대'이지 '검증된 결과 라벨'이 아님.
(5) check_label_dist.py에서 실제 궤적과 대조해 파라미터 보정 필요.

[파일럿 이력]
- gain run군 제거: DSLPIDControl은 호버링·동적 추종 모두에서 게인 저하에 강건함이
  6번의 실험으로 증명됨 (적분기 ±2 clip + 중력 feed-forward 구조적 특성).
  이는 발견이지 실패가 아님 — 보고서 서사로 활용.
- payload (1차~): 검증 완료. PAYLOAD_PEAKS = [1.8, 2.5, 3.2] 확정.
- wind (1차): WIND_PEAKS=[0.5,1.0,1.5] → 0.15N도 즉사. 대폭 낮춤.
- wind (2차): WIND_PEAKS=[0.15,0.30,0.50] → 여전히 crashed=True, t_fail=None.
  R2/R3 연속 3회 발화 전 R5 먼저 터짐 → 급추락.
- wind (3차, 현재): peak 더 낮추고 ramp 더 길게.
  CONSECUTIVE_FIRES=3 유지 (1로 낮추면 노이즈 스파이크를 t_fail로 오판).

출력: configs/workloads_pilot.csv
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd


# ── 파일럿 파라미터 범위 ─────────────────────────────────────────────────────
# weak  : 끝까지 안정 기대 → 음성(0)만 있는 run
# medium: 점진 발산 목표   → 음성+양성 다 있는 run (조기경보 핵심)
# strong: 즉사 경계        → 양성만 있거나 t_fail이 너무 이른 run

PAYLOAD_PEAKS   = [1.8, 2.5, 3.2]    # 확정값 (1차~2차 파일럿 검증 완료)

# wind (3차): 2차에서도 급추락 → 한 번 더 대폭 낮춤 + ramp 길게
# 목표: R2/R3(자세/각속도)이 R5(추락) 보다 먼저 연속 3회 발화
WIND_PEAKS      = [0.03, 0.06, 0.12]  # [N] — 2차 0.15N도 급추락, 1/5 수준으로
WIND_RAMP       = 12.0                 # ramp 길게 → 자세 발산이 천천히 자랄 시간
WIND_DURATION   = 30.0                 # 더 길게 — ramp 12s + 발산 여유 충분히

SEEDS = [42, 7, 123]

PAYLOAD_START   = 2.0
PAYLOAD_RAMP    = 4.0
WIND_START      = 2.0
DURATION        = 15.0   # payload run (확정값 유지)

OUTPUT_DIR      = Path("data/raw/pilot")
LEVEL_NAMES     = ["weak", "medium", "strong"]


def _level(idx: int) -> str:
    if idx < len(LEVEL_NAMES):
        return LEVEL_NAMES[idx]
    return f"level_{idx}"


def build_payload_rows() -> list[dict]:
    rows = []
    for (pi, peak), seed in itertools.product(enumerate(PAYLOAD_PEAKS), SEEDS):
        run_id = f"payload_{_level(pi)}_s{seed}"
        rows.append({
            "run_id":            run_id,
            "disturbance_type":  "payload",
            "level":             _level(pi),
            "seed":              seed,
            "duration":          DURATION,
            "payload_factor":    peak,
            "payload_start":     PAYLOAD_START,
            "payload_ramp":      PAYLOAD_RAMP,
            "wind_mode":         "none",
            "wind_peak":         0.0,
            "wind_start":        0.0,
            "wind_ramp":         0.0,
            "p_gain_mult_start": 1.0,
            "p_gain_mult_end":   1.0,
            "gain_start":        0.0,
            "gain_ramp":         0.0,
            "output":            str(OUTPUT_DIR / f"{run_id}.csv"),
        })
    return rows


def build_wind_rows() -> list[dict]:
    rows = []\
    
    for (wi, peak), seed in itertools.product(enumerate(WIND_PEAKS), SEEDS):
        run_id = f"wind_{_level(wi)}_s{seed}"
        rows.append({
            "run_id":            run_id,
            "disturbance_type":  "wind",
            "level":             _level(wi),
            "seed":              seed,
            "duration":          WIND_DURATION,
            "payload_factor":    1.0,
            "payload_start":     0.0,
            "payload_ramp":      0.0,
            "wind_mode":         "turbulent_ramp",
            "wind_peak":         peak,
            "wind_start":        WIND_START,
            "wind_ramp":         WIND_RAMP,
            "p_gain_mult_start": 1.0,
            "p_gain_mult_end":   1.0,
            "gain_start":        0.0,
            "gain_ramp":         0.0,
            "output":            str(OUTPUT_DIR / f"{run_id}.csv"),
        })
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # gain run군 제거 — payload + wind 2종만
    rows = build_payload_rows() + build_wind_rows()
    df = pd.DataFrame(rows)

    out = Path("configs/workloads_pilot.csv")
    df.to_csv(out, index=False)

    print(f"파일럿 조합표 생성 완료: {len(df)}행 → {out}")
    print(df.groupby("disturbance_type")["level"].value_counts().to_string())


if __name__ == "__main__":
    main()