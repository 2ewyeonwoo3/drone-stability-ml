"""
generate_workloads_main.py — Stage 2-A 본 생성용 조건 조합표.

파일럿 결론:
- payload ✅ 점진 발산 확정 (S1·S2 둘 다 검증)
- gain ❌ DSLPIDControl 구조적 강건성 — 발견으로 보고서 기록
- wind ❌ 자세 발산 본질적 비점진 — 발견으로 보고서 기록

다양성 전략 — run 수가 아닌 축으로:
  payload_factor × payload_start × payload_ramp × scenario × seed
  = 5 × 3 × 3 × 2 × 3 = 270 run
  run당 ~56윈도우 → 약 15,000 윈도우 샘플 (조기경보 학습 충분)

scenario:
  S1 = 호버링 (target_pos 고정 (0,0,1))
  S2 = 원형 궤적 추종 (radius=0.5m, omega=1.0rad/s)
     → "정지 중 무거워짐" vs "이동 중 무거워짐" 맥락 다양성

출력: configs/workloads_main.csv
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd

# ── 본 생성 파라미터 격자 ─────────────────────────────────────────────────────
PAYLOAD_FACTORS = [1.8, 2.2, 2.5, 3.0, 3.2]  # weak → strong 5단계
PAYLOAD_STARTS  = [2.0, 4.0, 6.0]              # 외란 시작 시각 [s] — lead time 다양성
PAYLOAD_RAMPS   = [2.0, 4.0, 7.0]              # 외란 ramp 속도 [s] — 침강 속도 다양성
SCENARIOS       = ["S1", "S2"]                  # 정적 호버링 / 동적 원형 추종
SEEDS           = [42, 7, 123]

# S2 궤적 파라미터 (검증 완료: radius=0.5m, omega=1.0rad/s)
S2_RADIUS = 0.5
S2_OMEGA  = 1.0

# duration: payload ramp 종료 후 발산 여유 충분히
# max(payload_start) + max(payload_ramp) + 여유 = 6 + 7 + 7 = 20s
DURATION = 20.0

OUTPUT_DIR = Path("data/raw/main")
OUTPUT_CSV = Path("configs/workloads_main.csv")


def build_rows() -> list[dict]:
    rows = []
    for (pf, ps, pr, sc, seed) in itertools.product(
        PAYLOAD_FACTORS, PAYLOAD_STARTS, PAYLOAD_RAMPS, SCENARIOS, SEEDS
    ):
        run_id = f"payload_f{pf}_s{ps}_r{pr}_{sc}_seed{seed}"
        rows.append({
            "run_id":            run_id,
            "disturbance_type":  "payload",
            "scenario":          sc,
            "seed":              seed,
            "duration":          DURATION,
            # payload 파라미터
            "payload_factor":    pf,
            "payload_start":     ps,
            "payload_ramp":      pr,
            # wind 없음
            "wind_mode":         "none",
            "wind_peak":         0.0,
            "wind_start":        0.0,
            "wind_ramp":         0.0,
            # gain 고정 (정상값)
            "p_gain_mult_start": 1.0,
            "p_gain_mult_end":   1.0,
            "gain_start":        0.0,
            "gain_ramp":         0.0,
            # S2 궤적 파라미터
            "s2_radius":         S2_RADIUS if sc == "S2" else 0.0,
            "s2_omega":          S2_OMEGA  if sc == "S2" else 0.0,
            # 출력 경로
            "output":            str(OUTPUT_DIR / f"{run_id}.csv"),
        })
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    df = pd.DataFrame(rows)

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"본 생성 조합표 완료: {len(df)}행 → {OUTPUT_CSV}")
    print(f"\n[축별 분포]")
    print(f"  payload_factor: {sorted(df['payload_factor'].unique())}")
    print(f"  payload_start:  {sorted(df['payload_start'].unique())}")
    print(f"  payload_ramp:   {sorted(df['payload_ramp'].unique())}")
    print(f"  scenario:       {sorted(df['scenario'].unique())}")
    print(f"  seed:           {sorted(df['seed'].unique())}")
    print(f"\n총 {len(df)}행 = "
          f"{len(PAYLOAD_FACTORS)} factor × "
          f"{len(PAYLOAD_STARTS)} start × "
          f"{len(PAYLOAD_RAMPS)} ramp × "
          f"{len(SCENARIOS)} scenario × "
          f"{len(SEEDS)} seed")
    print(f"예상 윈도우: {len(df)} run × ~56 windows ≈ {len(df)*56:,}개")


if __name__ == "__main__":
    main()