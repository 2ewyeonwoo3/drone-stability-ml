"""
run_batch_main.py — Stage 2-A 본 생성 배치 실행.

configs/workloads_main.csv를 읽어 멀티프로세싱으로 시뮬을 돌리고
data/raw/main/ 아래에 run별 CSV를 저장한다.

S1(호버링)과 S2(원형 궤적 추종)를 동일 코드로 처리.
S2는 run_id에 "S2"가 포함된 경우 circle_target으로 목표를 교체.

실행:
  python src/run_batch_main.py                     # 전체 실행
  python src/run_batch_main.py --workers 6         # 병렬 수 지정
  python src/run_batch_main.py --config configs/workloads_main.csv
  python src/run_batch_main.py --dry-run           # 조합표만 출력
  python src/run_batch_main.py --limit 10          # 처음 N개만 (검증용)

HTCondor 실행 시:
  condor_submit htcondor/run_main.sub
  (break-even 비교용 — 로컬 본 생성 완료 후 동일 워크로드 재실행)
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulate import run_simulation


def circle_target(t: float, radius: float, omega: float, z: float = 1.0) -> np.ndarray:
    """원형 궤적 목표 위치. S2 시나리오 전용."""
    return np.array([radius * np.cos(omega * t),
                     radius * np.sin(omega * t),
                     z])


def _run_one(row: dict) -> dict:
    """조합표 한 행 → 시뮬 실행 → CSV 저장."""
    run_id = row["run_id"]
    output = row["output"]
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        sc = row.get("scenario", "S1")

        if sc == "S2":
            # S2: 원형 궤적 추종 — simulate.py를 직접 호출하지 않고
            # 내부 루프를 재현해서 target_pos를 매 스텝 갱신
            result = _run_s2(row)
        else:
            # S1: 기존 run_simulation 그대로
            result = _run_s1(row)

        elapsed = time.time() - t0
        return {"run_id": run_id, "status": "ok", "elapsed": elapsed,
                "rows": len(result)}

    except Exception as e:
        elapsed = time.time() - t0
        return {"run_id": run_id, "status": "error",
                "error": str(e), "elapsed": elapsed}


def _run_s1(row: dict) -> list:
    """S1 호버링 — run_simulation 직접 호출."""
    return run_simulation(
        seed=int(row["seed"]),
        duration_sec=float(row["duration"]),
        target_pos=(0.0, 0.0, 1.0),
        output_path=row["output"],
        disturbance_type=row["disturbance_type"],
        init_rpy_std=0.0,
        payload_factor=float(row["payload_factor"]),
        payload_start=float(row["payload_start"]),
        payload_ramp=float(row["payload_ramp"]),
    )


def _run_s2(row: dict) -> list:
    """S2 원형 궤적 추종 — run_simulation을 S2 래퍼로 호출.

    simulate.py에 scenario/target_trajectory 파라미터가 없으므로
    target_pos를 고정값으로 넘기되, 실제로는 루프 외부에서 제어가 필요.
    현재 run_simulation은 target_pos를 고정으로만 씀 →
    S2는 run_simulation 호출 후 target 컬럼을 후처리로 교체하는 방식.

    TODO(Stage 2-A 완료 후): simulate.py에 trajectory 파라미터 추가해
    내부에서 circle_target을 호출하도록 리팩터링.
    """
    import pybullet as p
    import csv
    from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
    from gym_pybullet_drones.utils.enums import DroneModel, Physics
    from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
    from disturbances import PayloadSchedule
    from simulate import TELEMETRY_COLUMNS

    radius = float(row.get("s2_radius", 0.5))
    omega  = float(row.get("s2_omega",  1.0))
    seed   = int(row["seed"])
    duration = float(row["duration"])

    np.random.seed(seed)
    init_xyzs = np.array([[radius, 0.0, 0.1]])  # 원 시작점에서 이륙

    env = CtrlAviary(
        drone_model=DroneModel.CF2X, num_drones=1,
        initial_xyzs=init_xyzs,
        initial_rpys=np.array([[0., 0., 0.]]),
        physics=Physics.PYB, pyb_freq=240, ctrl_freq=240,
        gui=False, record=False, obstacles=False,
    )
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    client   = env.CLIENT
    drone_id = int(env.getDroneIds()[0])
    plane_id = int(getattr(env, "PLANE_ID", 0))

    dyn = p.getDynamicsInfo(drone_id, -1, physicsClientId=client)
    nominal_mass, nominal_inertia = dyn[0], dyn[2]

    payload = PayloadSchedule(
        factor=float(row["payload_factor"]),
        start=float(row["payload_start"]),
        ramp=float(row["payload_ramp"]),
    )
    last_factor = 1.0
    steps  = int(duration * env.CTRL_FREQ)
    action = np.zeros((1, 4))
    crashed = False
    crash_step = None
    rows_out = []

    for i in range(steps):
        t = i / env.CTRL_FREQ
        target_pos = circle_target(t, radius, omega)

        # payload 주입
        if payload.is_active:
            f = payload.mass_factor(t)
            if abs(f - last_factor) > 1e-9:
                p.changeDynamics(drone_id, -1,
                    mass=nominal_mass * f,
                    localInertiaDiagonal=[c * f for c in nominal_inertia],
                    physicsClientId=client)
                last_factor = f
        mass_factor = last_factor

        obs, _, _, _, _ = env.step(action)
        s = obs[0]
        pos = s[0:3]; quat = s[3:7]; rpy = s[7:10]
        vel = s[10:13]; ang_v = s[13:16]; rpm = s[16:20]

        contacts = p.getContactPoints(bodyA=drone_id, bodyB=plane_id,
                                      physicsClientId=client)
        contact = 1 if (contacts and len(contacts) > 0) else 0
        if (contact or pos[2] <= 0.02) and not crashed:
            crashed = True
            crash_step = i

        rows_out.append({
            "step": i, "t": t,
            "x": pos[0], "y": pos[1], "z": pos[2],
            "qx": quat[0], "qy": quat[1], "qz": quat[2], "qw": quat[3],
            "roll": rpy[0], "pitch": rpy[1], "yaw": rpy[2],
            "vx": vel[0], "vy": vel[1], "vz": vel[2],
            "wx": ang_v[0], "wy": ang_v[1], "wz": ang_v[2],
            "rpm0": rpm[0], "rpm1": rpm[1], "rpm2": rpm[2], "rpm3": rpm[3],
            "target_x": target_pos[0], "target_y": target_pos[1],
            "target_z": target_pos[2],
            "mass_factor": mass_factor,
            "wind_x": 0.0, "wind_y": 0.0, "wind_z": 0.0,
            "contact": contact, "crashed": int(crashed),
            "p_gain_mult": 1.0, "seed": seed, "drone_id": 0,
            "ctrl_freq": 240,
            "disturbance_type": row["disturbance_type"],
        })

        rpm_cmd, _, _ = ctrl.computeControl(
            control_timestep=env.CTRL_TIMESTEP,
            cur_pos=pos, cur_quat=quat, cur_vel=vel, cur_ang_vel=ang_v,
            target_pos=target_pos,
        )
        action = np.array([rpm_cmd])

        if crashed and crash_step is not None:
            if (i - crash_step) >= int(1.0 * env.CTRL_FREQ):
                break

    env.close()

    # CSV 저장
    output = row["output"]
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TELEMETRY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Saved {len(rows_out)} rows to {output}")
    return rows_out


def main():
    parser = argparse.ArgumentParser(description="Stage 2-A 본 생성 배치 실행")
    parser.add_argument("--config",   default="configs/workloads_main.csv")
    parser.add_argument("--workers",  type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--limit",    type=int, default=None,
                        help="처음 N개만 실행 (검증용)")
    args = parser.parse_args()

    df = pd.read_csv(args.config)
    if args.limit:
        df = df.head(args.limit)

    print(f"조합표 로드: {len(df)}행 / workers={args.workers}")
    print(df.groupby(["disturbance_type", "scenario"]).size().to_string()
          if "scenario" in df.columns else
          df.groupby("disturbance_type").size().to_string())

    if args.dry_run:
        print("\n[dry-run] 실행 없이 종료.")
        return

    rows = df.to_dict("records")
    t_start = time.time()

    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(_run_one, rows)

    ok  = [r for r in results if r["status"] == "ok"]
    err = [r for r in results if r["status"] == "error"]
    total = time.time() - t_start

    print(f"\n완료: {len(ok)}/{len(rows)} 성공, {len(err)} 실패 (총 {total:.1f}초)")
    if err:
        print("\n실패 목록:")
        for r in err:
            print(f"  {r['run_id']}: {r.get('error','?')}")

    summary = Path("data/raw/main/run_summary.csv")
    pd.DataFrame(results).to_csv(summary, index=False)
    print(f"\n실행 요약: {summary}")
    print(f"로컬 실행 총 시간: {total:.1f}초 (HTCondor break-even 비교용 기준값)")


if __name__ == "__main__":
    main()