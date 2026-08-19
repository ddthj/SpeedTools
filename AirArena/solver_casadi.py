"""
solver_casadi.py - CasADi time-optimal solver (ca.Opti + RK4).
"""
import numpy as np
import casadi as ca
from math import pi, cos, sqrt
from simulator import (
    MAX_ANG_SPEED, vehicle_dynamics, slerp, quat_from_axis_angle,
    qmul, Car, CarControls, SIM_FPS, SIM_DT
)


def _solve_single_branch(q0, w0, q_target_signed, N, tol_angle, tol_speed, extra_rot=False):
    w_spd = float(np.linalg.norm(w0))
    a_avg = 22.0
    X_init = np.zeros((7, N + 1))

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

    opti = ca.Opti()
    Tf = opti.variable()
    X = opti.variable(7, N + 1)
    U = opti.variable(3, N)

    dt = Tf / N
    opti.minimize(Tf + 1e-3 * ca.sumsqr(U) * dt)

    opti.subject_to(X[:, 0] == np.concatenate([q0, w0]))
    opti.subject_to(opti.bounded(0.02, Tf, 8.0))

    # Terminal tolerances
    min_dot = float(cos(tol_angle / 2.0))
    opti.subject_to(ca.dot(X[0:4, -1], q_target_signed) >= min_dot)
    opti.subject_to(ca.sumsqr(X[4:7, -1]) <= tol_speed ** 2)

    for k in range(N):
        x_k, u_k = X[:, k], U[:, k]
        opti.subject_to(opti.bounded(-1.0, u_k, 1.0))
        opti.subject_to(ca.sumsqr(x_k[4:7]) <= MAX_ANG_SPEED ** 2)

        def f_rk(x, u):
            dq, dw = vehicle_dynamics(x[0:4], x[4:7], u)
            return ca.vertcat(dq, dw)

        k1 = f_rk(x_k, u_k)
        k2 = f_rk(x_k + dt / 2 * k1, u_k)
        k3 = f_rk(x_k + dt / 2 * k2, u_k)
        k4 = f_rk(x_k + dt * k3, u_k)
        opti.subject_to(X[:, k + 1] == x_k + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4))

    opti.set_initial(Tf, Tf_init)
    opti.set_initial(X, X_init)
    opti.set_initial(U, 0.0)

    opts = {
        "ipopt.print_level": 0,
        "ipopt.max_iter": 250,
        "ipopt.tol": 1e-4,
        "ipopt.acceptable_tol": 1e-3,
        "ipopt.acceptable_iter": 5,
        "print_time": False,
        "ipopt.hessian_approximation": "limited-memory",
        "ipopt.mu_strategy": "adaptive",
    }
    opti.solver("ipopt", opts)

    try:
        sol = opti.solve()
        return sol.value(Tf), sol.value(U)
    except Exception:
        try:
            return float(opti.debug.value(Tf)), opti.debug.value(U)
        except Exception:
            return float("inf"), None


def solve_casadi(q0, w0, target_rot, tol_angle=0.05, tol_speed=0.07, N=30):
    """
    Main entry point for CasADi solver.
    Evaluates direct and +360-deg coasting branches and returns (Tf, 120 FPS history).
    """
    q_tgt_direct = target_rot.copy() if np.dot(q0, target_rot) >= 0 else -target_rot.copy()
    q_tgt_extra = -q_tgt_direct

    best_Tf = float("inf")
    best_U = None

    # Branch 0: Direct turn
    tf0, u0 = _solve_single_branch(q0, w0, q_tgt_direct, N, tol_angle, tol_speed, extra_rot=False)
    if tf0 < best_Tf:
        best_Tf, best_U = tf0, u0

    # Branch 1: Extra rotation
    if np.linalg.norm(w0) > 1.2:
        tf1, u1 = _solve_single_branch(q0, w0, q_tgt_extra, N, tol_angle, tol_speed, extra_rot=True)
        if tf1 < best_Tf:
            best_Tf, best_U = tf1, u1

    # Simulate at 120 FPS
    car = Car(rot=q0.copy(), ang_vel=w0.copy())
    num_ticks = max(1, int(round(best_Tf / SIM_DT)))
    dt_ctrl = best_Tf / N
    hist = []

    for tick in range(num_ticks + 25):
        sim_time = tick * SIM_DT
        k = min(N - 1, int(sim_time / dt_ctrl))
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
