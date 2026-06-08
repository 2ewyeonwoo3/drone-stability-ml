"""
run_batch.py — configs/workloads.csv에 정의된 여러 시뮬레이션을 일괄 실행

"""

from __future__ import annotations

import pandas as pd

from simulate import run_simulation


# CSV 컬럼명 -> (run_simulation 인자명, 자료형, 기본값)
_OPTIONAL = {
    "duration": ("duration_sec", float, 8.0),
    "target_z": ("target_z", float, 1.0),
    "payload_factor": ("payload_factor", float, 1.0),
    "payload_start": ("payload_start", float, 0.0),
    "payload_ramp": ("payload_ramp", float, 0.0),
    "wind_mode": ("wind_mode", str, "none"),
    "wind_mag": ("wind_mag", float, 0.0),
    "wind_start": ("wind_start", float, 0.0),
    "wind_gust_hz": ("wind_gust_hz", float, 1.0),
}


def _get(row, col, caster, default):
    """
    행(row)에서 값을 읽어오고,

    값이 존재하면 caster로 형변환하여 반환,
    없거나 NaN이면 기본값을 반환한다.
    """
    if col in row and pd.notna(row[col]):
        return caster(row[col])
    return default


def main(config_path: str = "configs/workloads.csv"):
    """workloads.csv를 읽어 각 실험을 순차적으로 실행한다."""
    df = pd.read_csv(config_path)
    for i, (_, row) in enumerate(df.iterrows()):
        kwargs = dict(
            p_gain_mult=_get(row, "p_gain_mult", float, 1.0),
            seed=int(_get(row, "seed", int, 42)),
            output_path=row["output"],
        )
        target_z = _get(row, "target_z", float, 1.0)
        kwargs["target_pos"] = (0.0, 0.0, target_z)
        for col, (kw, caster, default) in _OPTIONAL.items():
            if col == "target_z":
                continue
            kwargs[kw] = _get(row, col, caster, default)

        rid = row.get("run_id", f"run_{i}")
        print(f"[{i + 1}/{len(df)}] {rid}: p_gain={kwargs['p_gain_mult']} "
              f"payload={kwargs['payload_factor']} wind={kwargs['wind_mode']}/{kwargs['wind_mag']}")
        run_simulation(**kwargs)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/workloads.csv")
