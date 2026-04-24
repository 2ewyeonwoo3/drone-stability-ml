import argparse
import csv
import os
import time

import numpy as np

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


def run_simulation(p_gain_mult: float, duration_sec: int, output_path: str, seed: int):
    np.random.seed(seed)

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([[0, 0, 0.1]]),
        initial_rpys=np.array([[0, 0, 0]]),
        physics=Physics.PYB,
        gui=True,
        record=False,
        obstacles=False,
    )

    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

    # 기본 P 게인에 배율 적용
    ctrl.P_COEFF_FOR = ctrl.P_COEFF_FOR * p_gain_mult

    target_pos = np.array([0, 0, 1.0])
    steps = int(duration_sec * env.CTRL_FREQ)

    rows = []

    for i in range(steps):
        obs, reward, terminated, truncated, info = env.step(
            np.array([[0, 0, 0, 0]])
        )

        state = obs[0]
        current_pos = state[0:3]
        current_rpy = state[7:10]

        rpm, _, _ = ctrl.computeControl(
            control_timestep=env.CTRL_TIMESTEP,
            cur_pos=current_pos,
            cur_quat=state[3:7],
            cur_vel=state[10:13],
            cur_ang_vel=state[13:16],
            target_pos=target_pos,
        )

        obs, reward, terminated, truncated, info = env.step(np.array([rpm]))

        rows.append({
            "step": i,
            "time": i / env.CTRL_FREQ,
            "x": current_pos[0],
            "y": current_pos[1],
            "z": current_pos[2],
            "roll": current_rpy[0],
            "pitch": current_rpy[1],
            "yaw": current_rpy[2],
            "p_gain_mult": p_gain_mult,
            "seed": seed,
        })

        time.sleep(env.CTRL_TIMESTEP)

    env.close()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved log to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--p_gain_mult", type=float, default=1.0)
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/logs/run_001.csv")

    args = parser.parse_args()

    run_simulation(
      p_gain_mult=args.p_gain_mult,
      duration_sec=args.duration,
      output_path=args.output,
      seed=args.seed,
  )