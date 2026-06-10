"""
run_batch_pilot.py — Stage 2-A 파일럿 배치 실행.

configs/workloads_pilot.csv를 읽어 멀티프로세싱으로 시뮬을 돌리고
data/raw/pilot/ 아래에 run별 CSV를 저장한다.

실행:
  python src/run_batch_pilot.py                        # 전체 실행
  python src/run_batch_pilot.py --workers 4            # 병렬 수 지정
  python src/run_batch_pilot.py --dry-run              # 조합표만 출력, 시뮬 안 돌림

주의:
  - 파일럿 목적은 외란 강도 범위 검증 (즉사/점진발산/영원안정 구분)
  - 결과는 (5) check_label_dist.py로 확인 후 범위 보정
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulate import run_simulation


def _run_one(row: dict) -> dict:
    """조합표 한 행 → 시뮬 실행 → CSV 저장. 멀티프로세싱 worker."""
    run_id = row["run_id"]
    output = row["output"]
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        kwargs = dict(
            seed=int(row["seed"]),
            duration_sec=float(row["duration"]),
            target_pos=(0.0, 0.0, 1.0),
            output_path=output,
            disturbance_type=row["disturbance_type"],
            init_rpy_std=0.0,    # 0.01 → 0.0 으로 변경
        )

        dtype = row["disturbance_type"]

        if dtype == "payload":
            kwargs.update(
                payload_factor=float(row["payload_factor"]),
                payload_start=float(row["payload_start"]),
                payload_ramp=float(row["payload_ramp"]),
            )

        elif dtype == "wind":
            kwargs.update(
                wind_mode="turbulent_ramp",
                wind_mag=float(row["wind_peak"]),
                wind_start=float(row["wind_start"]),
                wind_ramp=float(row["wind_ramp"]),
            )

        elif dtype == "gain":
            kwargs.update(
                p_gain_mult=float(row["p_gain_mult_start"]),
                gain_end=float(row["p_gain_mult_end"]),
                gain_start_t=float(row["gain_start"]),
                gain_ramp=float(row["gain_ramp"]),
            )

        run_simulation(**kwargs)
        elapsed = time.time() - t0
        return {"run_id": run_id, "status": "ok", "elapsed": elapsed}

    except Exception as e:
        elapsed = time.time() - t0
        return {"run_id": run_id, "status": "error", "error": str(e), "elapsed": elapsed}


def main():
    parser = argparse.ArgumentParser(description="Stage 2-A 파일럿 배치 실행")
    parser.add_argument("--config",   default="configs/workloads_pilot.csv")
    parser.add_argument("--workers",  type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--dry-run",  action="store_true", help="조합표만 출력, 실행 안 함")
    args = parser.parse_args()

    df = pd.read_csv(args.config)
    print(f"조합표 로드: {len(df)}행 / workers={args.workers}")
    print(df.groupby(["disturbance_type", "level"]).size().to_string())

    if args.dry_run:
        print("\n[dry-run] 시뮬 실행 없이 종료.")
        return

    rows = df.to_dict("records")
    t_start = time.time()

    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(_run_one, rows)

    # 결과 요약
    ok  = [r for r in results if r["status"] == "ok"]
    err = [r for r in results if r["status"] == "error"]

    print(f"\n완료: {len(ok)}/{len(rows)} 성공, {len(err)} 실패 "
          f"(총 {time.time() - t_start:.1f}초)")

    if err:
        print("\n실패 목록:")
        for r in err:
            print(f"  {r['run_id']}: {r.get('error', '?')}")

    # 결과 요약 CSV 저장
    summary_path = Path("data/raw/pilot/run_summary.csv")
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"\n실행 요약 저장: {summary_path}")


if __name__ == "__main__":
    main()
