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

from disturbances import make_wind, PayloadSchedule, Wind, RampedTurbulentWind


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
    "disturbance_type",                         # Stage 2-A 메타: 외란유형 (라벨링·feature importance용)
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
    wind_ramp: float = 0.0,        # RampedTurbulentWind용 ramp 지속시간 [s]. 기존 모드엔 무시됨
    # ── Stage 2-A 추가 파라미터 ───────────────────────────────────────────────
    # gain ramp: p_gain_mult(시작) → gain_end(종료)까지 선형 하강. (다) gain run군 전용.
    gain_end: float = 1.0,         # ramp 종료 시점의 게인 배수 (기본=시작값과 같음 → ramp 없음)
    gain_start_t: float = 0.0,     # 게인 하강 시작 시각 [s]
    gain_ramp: float = 0.0,        # 게인 하강 지속 시간 [s] (0=고정, ramp 없음)
    # 초기조건 섭동: seed마다 다른 이륙 자세 → payload/gain run에서도 seed가 의미를 가짐.
    # 현실 드론의 이륙 자세 공차(기계·지면 불균일)를 모사. std=0.01rad ≈ 0.57도.
    init_rpy_std: float = 0.01,
    # 외란유형 메타: 라벨링·feature importance 해석용. 호출부(run_batch_pilot)에서 주입.
    disturbance_type: str = "none",
    # ─────────────────────────────────────────────────────────────────────────
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

    Stage 2-A 추가 파라미터(wind_ramp, gain_end, gain_start_t, gain_ramp,
    init_rpy_std, disturbance_type)는 모두 기본값이 있어 기존 호출에 영향 없음.
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)   # 재현 가능한 전용 RNG (init_rpy 섭동용)
    target_pos = np.asarray(target_pos, dtype=float)

    # 초기 자세 섭동: seed마다 다른 미세 오차 → payload/gain run에서도 seed가 의미를 가짐
    init_rpy = rng.normal(0.0, init_rpy_std, size=3) if init_rpy_std > 0 else np.zeros(3)

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, 0.1]]),
        initial_rpys=np.array([init_rpy]),
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
    # gain_ramp > 0 이면 루프 안에서 동적으로 갱신 (Stage 2-A (다) gain run군)
    # P·I 동시 하강: P만 낮추면 I 게인이 보완해서 발산이 안 일어남
    _p_coeff_for_base  = ctrl.P_COEFF_FOR.copy()   # 위치 P 게인 기준점
    _i_coeff_for_base  = ctrl.I_COEFF_FOR.copy()   # 위치 I 게인 기준점
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

    # RampedTurbulentWind는 wind_ramp 파라미터가 필요해 직접 생성
    if wind_mode == "turbulent_ramp":
        wind: Wind = RampedTurbulentWind(
            peak_mag=wind_mag, start=wind_start, ramp=wind_ramp,
            theta=2.0, dt=env.CTRL_TIMESTEP, seed=seed,
        )
    else:
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
        # Gain ramp 주입 (Stage 2-A (다) gain run군)
        # p_gain_mult(시작) → gain_end(종료)까지 선형 하강
        # gain_ramp=0이면 이 블록은 실행되지 않음 (기존 동작 유지)
        # ------------------------------
        if gain_ramp > 0 and t >= gain_start_t:
            elapsed = t - gain_start_t
            frac = min(1.0, elapsed / gain_ramp)
            current_mult = p_gain_mult + (gain_end - p_gain_mult) * frac
            # P·I 동시 하강: I 게인도 같이 낮춰야 적분 보상을 억제할 수 있음
            ctrl.P_COEFF_FOR = _p_coeff_for_base * current_mult
            ctrl.I_COEFF_FOR = _i_coeff_for_base * current_mult

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
            "disturbance_type": disturbance_type,
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
                        choices=["none", "constant", "gust", "turbulent", "turbulent_ramp"])
    parser.add_argument("--wind_mag", type=float, default=0.0, help="force magnitude [N]")
    parser.add_argument("--wind_start", type=float, default=0.0)
    parser.add_argument("--wind_gust_hz", type=float, default=1.0)
    parser.add_argument("--wind_ramp", type=float, default=0.0,
                        help="turbulent_ramp 모드: ramp 지속 시간 [s]")
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
        wind_ramp=args.wind_ramp,
        gui=args.gui,
    )
