"""
solver_casadi_fast.py - Fast, pre-compiled CasADi solver with SLERP warm-starting.
"""
import os
import numpy as np
import casadi as ca
from math import pi, sqrt
from simulator import (
    slerp, quat_from_axis_angle, qmul, Car, CarControls, SIM_FPS, SIM_DT
)

MODEL_FILE = "casadi_solver.casadi"
solver_fn = ca.Function.load(MODEL_FILE)
N_STEPS = 25


def _solve_branch_fast(q0, w0, q_target_signed, extra_rot=False):
    w_spd = float(np.linalg.norm(w0))
    a_avg = 22.0
    N = N_STEPS
    X_init = np.zeros((7, N + 1))

    # Construct the exact SLERP initial guess that guides IPOPT
    if not extra_rot:
        inner = min(1.0, max(-1.0, float(abs(np.dot(q0, q_target_signed)))))
        delta_angle = 2.0 * np.arccos(inner)
        Tf_init = float(np.clip((w_spd / a_avg) + 2.0 * sqrt(max(0.01, delta_angle) / a_avg), 0.15, 4.5))
        for k in range(N + 1):
            ratio = k / N
            X_init[0:4, k] = slerp(q0, q_target_signed, ratio)
            X_init[4:7, k] = (1.0 - ratio) * w0
    else:
        axis = w0 / max(1e-4, w_spd)
        t_coast = (2.0 * pi) / max(1.0, w_spd)
        Tf_init = float(np.clip(t_coast + (w_spd / a_avg), 0.4, 5.5))
        for k in range(N + 1):
            ratio = k / N
            q_spin = quat_from_axis_angle(axis, 2.0 * pi * ratio)
            q_base = slerp(q0, -q_target_signed, ratio)
            qk = qmul(q_spin, q_base)
            X_init[0:4, k] = qk / np.linalg.norm(qk)
            X_init[4:7, k] = (1.0 - ratio ** 2) * w0

    x0 = np.concatenate([q0, w0])

    try:
        Tf_val, U_val = solver_fn(x0, q_target_signed, Tf_init, X_init)
        return float(Tf_val), np.array(U_val)
    except Exception:
        return float("inf"), None


def solve_casadi_fast(q0, w0, target_rot, tol_angle=0.05, tol_speed=0.07):
    # Ensure shortest path on S^3
    q_tgt_direct = target_rot.copy() if np.dot(q0, target_rot) >= 0 else -target_rot.copy()
    q_tgt_extra = -q_tgt_direct

    best_Tf = float("inf")
    best_U = None

    # Branch 0: Direct turn
    tf0, u0 = _solve_branch_fast(q0, w0, q_tgt_direct, extra_rot=False)
    if tf0 < best_Tf:
        best_Tf, best_U = tf0, u0

    # Branch 1: Wrap-around rotation (if already spinning fast)
    if np.linalg.norm(w0) > 1.2:
        tf1, u1 = _solve_branch_fast(q0, w0, q_tgt_extra, extra_rot=True)
        if tf1 < best_Tf:
            best_Tf, best_U = tf1, u1

    if best_U is None or np.isinf(best_Tf):
        return float("inf"), []

    # Simulate at 120 FPS
    car = Car(rot=q0.copy(), ang_vel=w0.copy())
    num_ticks = max(1, int(round(best_Tf / SIM_DT)))
    dt_ctrl = best_Tf / N_STEPS
    hist = []

    for tick in range(num_ticks + 25):
        sim_time = tick * SIM_DT
        k = min(N_STEPS - 1, int(sim_time / dt_ctrl))
        u_cmd = np.array(
            [float(best_U[0, k]), float(best_U[1, k]), float(best_U[2, k])]) if tick < num_ticks else np.zeros(3)

        inner = min(1.0, max(-1.0, abs(float(np.dot(car.rot, target_rot)))))
        angle_rad = 2.0 * np.arccos(inner)
        spd = float(np.linalg.norm(car.ang_vel))
        status = "SETTLED" if tick >= num_ticks else "ACTIVE"

        hist.append({
            "rot": car.rot.copy(),
            "u": u_cmd,
            "t": sim_time,
            "angle_rad": angle_rad,
            "speed": spd,
            "status": status
        })

        if tick < num_ticks:
            car.step_turn(CarControls(*u_cmd), SIM_DT)

    return best_Tf, hist
