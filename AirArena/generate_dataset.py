"""
generate_dataset.py - Generates and saves a standalone .npz dataset for satellite reorientation.
"""
import os
import multiprocessing as mp
from math import pi, sqrt, cos, sin
import numpy as np
import casadi as ca

from simulator import (
    MAX_ANG_SPEED, TORQUE, TORQUE_APPLY_SCALE, slerp, quat_from_axis_angle,
    qmul, qrot, BASIS_QUAT, vehicle_dynamics
)

MODEL_FILE = "casadi_solver.casadi"
N_STEPS = 25
OUTPUT_FILE = "reorientation_dataset.npz"

_worker_solver = None


def _init_worker():
    global _worker_solver
    _worker_solver = ca.Function.load(MODEL_FILE)


def _solve_single_scenario(args):
    q0, w0, target_rot = args
    x0 = np.concatenate([q0, w0])

    q_tgt = target_rot if np.dot(q0, target_rot) >= 0 else -target_rot
    w_spd = float(np.linalg.norm(w0))
    inner = min(1.0, max(-1.0, float(abs(np.dot(q0, q_tgt)))))
    delta_angle = 2.0 * np.arccos(inner)
    Tf_init = float(np.clip((w_spd / 22.0) + 2.0 * sqrt(max(0.01, delta_angle) / 22.0), 0.2, 4.5))

    X_init = np.zeros((7, N_STEPS + 1))
    for k in range(N_STEPS + 1):
        ratio = k / N_STEPS
        X_init[0:4, k] = slerp(q0, q_tgt, ratio)
        X_init[4:7, k] = (1.0 - ratio) * w0

    try:
        Tf_val, U_val = _worker_solver(x0, q_tgt, Tf_init, X_init)
        tf = float(Tf_val)
        u_mat = np.array(U_val)

        if 0.05 < tf < 7.0 and not np.isnan(tf):
            dt = tf / N_STEPS
            curr_x = x0.copy()
            states = [curr_x.copy()]

            for k in range(N_STEPS):
                u_k = u_mat[:, k]
                k1_q, k1_w = vehicle_dynamics(curr_x[0:4], curr_x[4:7], u_k)
                k2_q, k2_w = vehicle_dynamics(curr_x[0:4] + 0.5 * dt * k1_q, curr_x[4:7] + 0.5 * dt * k1_w, u_k)
                k3_q, k3_w = vehicle_dynamics(curr_x[0:4] + 0.5 * dt * k2_q, curr_x[4:7] + 0.5 * dt * k2_w, u_k)
                k4_q, k4_w = vehicle_dynamics(curr_x[0:4] + dt * k3_q, curr_x[4:7] + dt * k3_w, u_k)

                next_q = curr_x[0:4] + (dt / 6.0) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
                next_w = curr_x[4:7] + (dt / 6.0) * (k1_w + 2 * k2_w + 2 * k3_w + k4_w)
                next_q /= np.linalg.norm(next_q)
                curr_x = np.concatenate([next_q, next_w])
                states.append(curr_x.copy())

            return {
                "states": np.array(states),
                "controls": u_mat.T,
                "target": q_tgt
            }
    except Exception:
        pass
    return None


def generate_dataset(num_scenarios=1500):
    print(f"1. Sampling {num_scenarios} stratified scenarios...")
    scenarios = []

    # Tier 1: Large angles & high opposing momentum
    for _ in range(int(num_scenarios * 0.40)):
        axis = np.random.normal(0, 1, 3);
        axis /= np.linalg.norm(axis)
        q0 = quat_from_axis_angle(axis, np.random.uniform(pi * 0.6, pi))
        w0 = np.clip(-axis * np.random.uniform(2.5, 5.0) + np.random.normal(0, 0.5, 3), -MAX_ANG_SPEED, MAX_ANG_SPEED)
        scenarios.append((q0, w0, np.array([0., 0., 0., 1.])))

    # Tier 2: Yaw-heavy turns (Roll whips)
    for _ in range(int(num_scenarios * 0.35)):
        yaw_angle = np.random.uniform(pi * 0.4, pi) * (1 if np.random.rand() > 0.5 else -1)
        q0 = np.array([0.0, sin(yaw_angle / 2.0), 0.0, cos(yaw_angle / 2.0)])
        w0 = np.random.uniform(-2.0, 2.0, 3)
        scenarios.append((q0, w0, np.array([0., 0., 0., 1.])))

    # Tier 3: Fine terminal boundary
    for _ in range(num_scenarios - len(scenarios)):
        axis = np.random.normal(0, 1, 3);
        axis /= np.linalg.norm(axis)
        q0 = quat_from_axis_angle(axis, np.random.uniform(0.01, 0.25))
        w0 = np.random.uniform(-1.0, 1.0, 3)
        scenarios.append((q0, w0, np.array([0., 0., 0., 1.])))

    print(f"2. Solving via CasADi on {mp.cpu_count()} CPU cores...")
    with mp.Pool(processes=mp.cpu_count(), initializer=_init_worker) as pool:
        raw_results = pool.map(_solve_single_scenario, scenarios, chunksize=25)

    valid_trajs = [r for r in raw_results if r is not None]
    print(f"   Successfully solved {len(valid_trajs)} / {num_scenarios} scenarios.")

    print("3. Extracting 6D zero-centered features & applying 8-fold parity symmetries...")
    X_train, Y_train = [], []
    parity_flips = [(1, 1, 1), (-1, 1, -1), (1, -1, -1), (-1, -1, 1)]

    for traj in valid_trajs:
        states = traj["states"]
        controls = traj["controls"]
        target_rot = traj["target"]

        for k in range(len(controls)):
            q_world = states[k, 0:4]
            w_world = states[k, 4:7]
            u_ctrl = controls[k]

            dir_curr = qmul(q_world, BASIS_QUAT)
            dir_tgt = qmul(target_rot, BASIS_QUAT)
            inv_curr = np.array([-dir_curr[0], -dir_curr[1], -dir_curr[2], dir_curr[3]])

            q_rel = qmul(inv_curr, dir_tgt)
            if q_rel[3] < 0:
                q_rel = -q_rel
            w_local = qrot(inv_curr, w_world)

            # 6D zero-centered vector: [qx, qy, qz, wx, wy, wz]
            qx, qy, qz = q_rel[0], q_rel[1], q_rel[2]
            wx, wy, wz = w_local[0], w_local[1], w_local[2]
            up, uy, ur = u_ctrl[0], u_ctrl[1], u_ctrl[2]

            for sp, sy, sr in parity_flips:
                X_train.append([qx * sp, qy * sy, qz * sr, wx * sp, wy * sy, wz * sr])
                Y_train.append([up * sp, uy * sy, ur * sr])

    # 4. Stationary Anchor Augmentation: Teach zero output and active damping near the origin
    print("4. Injecting stationary anchor samples at origin...")
    for _ in range(int(len(X_train) * 0.15)):
        # Small residual orientation jitter around origin
        q_jit = np.random.normal(0, 0.015, 3)
        # Small residual velocity jitter
        w_jit = np.random.normal(0, 0.05, 3)

        # Desired control at origin is pure critical damping to zero: u = -w / (a_max * dt)
        accel = TORQUE * TORQUE_APPLY_SCALE
        u_damp = -w_jit / (accel * (1.0 / 120.0))
        u_damp = np.clip(u_damp, -1.0, 1.0)

        X_train.append(np.concatenate([q_jit, w_jit]))
        Y_train.append(u_damp)

    X_train = np.array(X_train, dtype=np.float32)
    Y_train = np.array(Y_train, dtype=np.float32)

    print(f"5. Saving {len(X_train):,} state-action pairs to '{OUTPUT_FILE}'...")
    np.savez_compressed(
        OUTPUT_FILE,
        X=X_train,
        Y=Y_train,
        meta={
            "description": "Satellite 3D reorientation dataset",
            "state_format": "[qx_rel, qy_rel, qz_rel, wx_local, wy_local, wz_local]",
            "action_format": "[u_pitch, u_yaw, u_roll]",
            "num_samples": len(X_train)
        }
    )
    print("Done! Dataset is ready for model experimentation.")


if __name__ == "__main__":
    if not os.path.exists(MODEL_FILE):
        print(f"Error: {MODEL_FILE} not found. Run generate_solver.py first.")
        exit(1)
    generate_dataset(num_scenarios=1000)
