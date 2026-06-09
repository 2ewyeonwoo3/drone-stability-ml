"""
simulate.py — 풍부한 텔레메트리 데이터를 생성하는 헤드리스 드론 시뮬레이터
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import List, Dict, Optional

import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from disturbances import make_wind, PayloadSchedule, Wind


# 텔레메트리 컬럼 스키마 — producer·consumer가 이 목록을 공유
# ctrl_freq: Spark consumer의 FFT UDF가 실제 샘플링 레이트(fs)로 사용
# pyb_freq != ctrl_freq일 수 있으므로 반드시 명시
TELEMETRY_COLUMNS = [
    "step", "t",
    "x", "y", "z",
    "qx", "qy", "qz", "qw",
    "roll", "pitch", "yaw",
    "vx", "vy", "vz",
    "wx", "wy", "wz",
    "rpm0", "rpm1", "rpm2", "rpm3",
    "target_x", "target_y", "target_z",
    "mass_factor", "wind_x", "wind_y", "wind_z",
    "contact", "crashed",
    "p_gain_mult", "seed", "drone_id",
    "ctrl_freq",                                # FFT fs 계산용 — Spark consumer가 직접 읽어야 함
]


# 드론 본체의 질량과 관성을 factor 배 만큼 스케일링한다.
def _set_payload(client: int, drone_id: int, nominal_mass: float,
                 nominal_inertia, factor: float):
    p.changeDynamics(
        drone_id, -1,
        mass=nominal_mass * factor,
        localInertiaDiagonal=[c * factor for c in nominal_inertia],
        physicsClientId=client,
    )

def run_simulation(
    p_gain_mult: float = 1.0,
    duration_sec: float = 8.0,
    seed: int = 42,
    target_pos=(0.0, 0.0, 1.0),
    output_path: Optional[str] = None,
    *,
    # payload
    payload_factor: float = 1.0,
    payload_start: float = 0.0,
    payload_ramp: float = 0.0,
    # wind
    wind_mode: str = "none",
    wind_mag: float = 0.0,
    wind_dir=(1.0, 0.0, 0.0),
    wind_start: float = 0.0,
    wind_gust_hz: float = 1.0,
    # engine
    ctrl_freq: int = 240,
    pyb_freq: int = 240,
    gui: bool = False,
    stop_after_crash_sec: float = 1.0,
    drone_id_tag: int = 0,
) -> List[Dict]:
    """
    헤드리스 드론 시뮬레이션을 수행하고
    텔레메트리 행(row) 목록을 반환한다.

    output_path가 지정되면 CSV 파일도 함께 저장한다.

    외란 관련 파라미터를 기본값으로 두면
    기존 시뮬레이터와 동일하게 동작한다.
    """
    np.random.seed(seed)
    target_pos = np.asarray(target_pos, dtype=float)

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, 0.1]]),
        initial_rpys=np.array([[0.0, 0.0, 0.0]]),
        physics=Physics.PYB,
        pyb_freq=pyb_freq,
        ctrl_freq=ctrl_freq,
        gui=gui,
        record=False,
        obstacles=False,
    )

    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    # PID 제어기의 위치 P 게인 배수 조정
    # (기존 코드의 "기체 강성(stiffness)" 조절 노브 역할)
    ctrl.P_COEFF_FOR = ctrl.P_COEFF_FOR * p_gain_mult  

    client = env.CLIENT
    drone_id = int(env.getDroneIds()[0])
    plane_id = int(getattr(env, "PLANE_ID", 0))

    # 기본 동역학 정보 저장
    # 이후 payload 적용 시 상대적으로 스케일링하기 위해 사용
    dyn = p.getDynamicsInfo(drone_id, -1, physicsClientId=client)
    nominal_mass = dyn[0]
    nominal_inertia = dyn[2]

    # 외란 생성기 초기화
    payload = PayloadSchedule(factor=payload_factor, start=payload_start, ramp=payload_ramp)
    wind: Wind = make_wind(
        wind_mode, dt=env.CTRL_TIMESTEP, mag=wind_mag, seed=seed,
        direction=wind_dir, start=wind_start, gust_hz=wind_gust_hz,
    )

    steps = int(duration_sec * env.CTRL_FREQ)
    rows: List[Dict] = []

    # 이전 제어 입력
    action = np.zeros((1, 4))
    last_applied_factor = 1.0
    crashed = False
    crash_step = None

    for i in range(steps):
        t = i / env.CTRL_FREQ

        # ------------------------------
        # Payload 주입
        # 질량 변화가 있을 때만 changeDynamics 호출
        # ------------------------------
        if payload.is_active:
            f = payload.mass_factor(t)
            if abs(f - last_applied_factor) > 1e-9:
                _set_payload(client, drone_id, nominal_mass, nominal_inertia, f)
                last_applied_factor = f
        # 현재 적용 중인 질량 배수
        mass_factor = last_applied_factor

        # ------------------------------
        # Wind 주입
        # 외력은 매 시뮬레이션 스텝마다 초기화되므로
        # 매 스텝 다시 적용해야 함
        # ------------------------------
        wind_vec = wind.force(t)
        if np.any(wind_vec):
            p.applyExternalForce(
                drone_id, -1,
                forceObj=wind_vec.tolist(), posObj=[0.0, 0.0, 0.0],
                flags=p.WORLD_FRAME, physicsClientId=client,
            )

        # ------------------------------
        # 이전 action으로 물리 시뮬레이션 1회 수행
        # ------------------------------
        obs, _, _, _, _ = env.step(action)
        s = obs[0]
        pos = s[0:3]; quat = s[3:7]; rpy = s[7:10]
        vel = s[10:13]; ang_v = s[13:16]; rpm = s[16:20]

        # 지면과 접촉 여부 확인
        contacts = p.getContactPoints(
            bodyA=drone_id, bodyB=plane_id, physicsClientId=client
        )
        contact = 1 if (contacts and len(contacts) > 0) else 0
        # 충돌 또는 추락 판정
        if (contact or pos[2] <= 0.02) and not crashed:
            crashed = True
            crash_step = i

        rows.append({
            "step": i, "t": t,
            "x": pos[0], "y": pos[1], "z": pos[2],
            "qx": quat[0], "qy": quat[1], "qz": quat[2], "qw": quat[3],
            "roll": rpy[0], "pitch": rpy[1], "yaw": rpy[2],
            "vx": vel[0], "vy": vel[1], "vz": vel[2],
            "wx": ang_v[0], "wy": ang_v[1], "wz": ang_v[2],
            "rpm0": rpm[0], "rpm1": rpm[1], "rpm2": rpm[2], "rpm3": rpm[3],
            "target_x": target_pos[0], "target_y": target_pos[1], "target_z": target_pos[2],
            "mass_factor": mass_factor,
            "wind_x": wind_vec[0], "wind_y": wind_vec[1], "wind_z": wind_vec[2],
            "contact": contact, "crashed": int(crashed),
            "p_gain_mult": p_gain_mult, "seed": seed, "drone_id": drone_id_tag,
            "ctrl_freq": ctrl_freq,
        })

        # ------------------------------
        # 현재 상태를 기반으로
        # 다음 제어 입력(RPM) 계산
        # ------------------------------
        rpm_cmd, _, _ = ctrl.computeControl(
            control_timestep=env.CTRL_TIMESTEP,
            cur_pos=pos, cur_quat=quat, cur_vel=vel, cur_ang_vel=ang_v,
            target_pos=target_pos,
        )
        action = np.array([rpm_cmd])

        # 추락 이후에도 잠시 더 기록한 뒤 종료
        # (붕괴 과정은 저장하고,
        # 바닥에 누워있는 데이터는 길게 기록하지 않음)
        if crashed and crash_step is not None:
            if (i - crash_step) >= int(stop_after_crash_sec * env.CTRL_FREQ):
                break

    env.close()

    # CSV 저장
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=TELEMETRY_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} rows to {output_path}")

    return rows


def _build_parser() -> argparse.ArgumentParser:
    """명령행 인자 파서 생성"""
    parser = argparse.ArgumentParser(description="Headless drone telemetry generator")
    parser.add_argument("--p_gain_mult", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_z", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="results/logs/run_001.csv")
    # payload 관련 옵션
    parser.add_argument("--payload_factor", type=float, default=1.0,
                        help="final mass multiplier (1.0 = none)")
    parser.add_argument("--payload_start", type=float, default=0.0)
    parser.add_argument("--payload_ramp", type=float, default=0.0,
                        help="seconds to ramp from 1.0 to payload_factor (0 = step)")
    # wind 관련 옵션
    parser.add_argument("--wind_mode", type=str, default="none",
                        choices=["none", "constant", "gust", "turbulent"])
    parser.add_argument("--wind_mag", type=float, default=0.0, help="force magnitude [N]")
    parser.add_argument("--wind_start", type=float, default=0.0)
    parser.add_argument("--wind_gust_hz", type=float, default=1.0)
    # 디버깅용 GUI 활성화
    parser.add_argument("--gui", action="store_true", help="enable PyBullet GUI (debug only)")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_simulation(
        p_gain_mult=args.p_gain_mult,
        duration_sec=args.duration,
        seed=args.seed,
        target_pos=(0.0, 0.0, args.target_z),
        output_path=args.output,
        payload_factor=args.payload_factor,
        payload_start=args.payload_start,
        payload_ramp=args.payload_ramp,
        wind_mode=args.wind_mode,
        wind_mag=args.wind_mag,
        wind_start=args.wind_start,
        wind_gust_hz=args.wind_gust_hz,
        gui=args.gui,
    )
