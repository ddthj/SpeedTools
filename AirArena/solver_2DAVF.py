"""
solver_dir2d.py - 2D Coupled Directional Acceleration Vector Field Controller.
"""
from math import pi, sqrt, atan2, cos, sin
import numpy as np
from simulator import (
    MAX_ANG_SPEED, TORQUE, DAMPING, TORQUE_APPLY_SCALE, BASIS_QUAT,
    qmul, qrot, Car, CarControls, SIM_FPS, SIM_DT
)

ACCEL_MAX = TORQUE * TORQUE_APPLY_SCALE


def _get_control_frame(car_rot, target_rot, car_ang_vel):
    dir_curr = qmul(car_rot, BASIS_QUAT)
    dir_tgt = qmul(target_rot, BASIS_QUAT)
    inv_curr = np.array([-dir_curr[0], -dir_curr[1], -dir_curr[2], dir_curr[3]])

    q_rel = qmul(inv_curr, dir_tgt)
    if q_rel[3] < 0:
        q_rel = -q_rel

    local_ang_vel = qrot(inv_curr, car_ang_vel)
    return q_rel, local_ang_vel


def _get_directional_max_accel(dir_vec, a_p, a_y):
    """Computes exact maximum acceleration along a 2D direction under box bounds |u| <= 1."""
    dx = abs(dir_vec[0])
    dy = abs(dir_vec[1])
    if dx < 1e-7 and dy < 1e-7:
        return min(a_p, a_y)
    scale_p = a_p / (dx + 1e-8)
    scale_y = a_y / (dy + 1e-8)
    return min(scale_p, scale_y)


def _solve_2d_vector_field(err_2d, vel_2d, a_p, a_y):
    """Unified 2D time-optimal controller on the Pitch-Yaw plane."""
    v_norm = float(np.linalg.norm(vel_2d))
    e_norm = float(np.linalg.norm(err_2d))

    if v_norm < 1e-6:
        a_dir_max = _get_directional_max_accel(err_2d, a_p, a_y)
        d_stop = np.zeros(2)
    else:
        v_hat = vel_2d / v_norm
        a_dir_max = _get_directional_max_accel(v_hat, a_p, a_y)
        # 2D stopping vector with discrete half-step offset
        stop_dist_mag = (v_norm ** 2) / (2.0 * a_dir_max) + 0.5 * v_norm * SIM_DT
        d_stop = v_hat * stop_dist_mag

    # Remaining distance vector from the switching surface
    delta_2d = err_2d - d_stop
    delta_norm = float(np.linalg.norm(delta_2d))

    # Boundary layer for critical damping
    t_bl = 1.5 * SIM_DT
    deadband = 0.5 * a_dir_max * (t_bl ** 2)

    if e_norm <= deadband and v_norm <= a_dir_max * t_bl:
        # Linear settling zone
        u_unbounded = (err_2d / t_bl - vel_2d) / (np.array([a_p, a_y]) * SIM_DT)
        return float(np.clip(u_unbounded[0], -1.0, 1.0)), float(np.clip(u_unbounded[1], -1.0, 1.0))

    if delta_norm < 1e-6:
        return 0.0, 0.0

    # Scale desired direction vector to rectangular control box [-1, 1]^2
    u_dir = delta_2d / delta_norm
    scale = max(abs(u_dir[0]), abs(u_dir[1]))
    u_box = u_dir / max(scale, 1e-6)

    # Proportional clamp near switching surface to prevent overshoot
    gain = min(1.0, delta_norm / (a_dir_max * (SIM_DT ** 2) * 1.5))
    u_2d = u_box * gain
    return float(np.clip(u_2d[0], -1.0, 1.0)), float(np.clip(u_2d[1], -1.0, 1.0))


def run_rm_dir2d(car: Car, target_rot):
    q_rel, local_vel = _get_control_frame(car.rot, target_rot, car.ang_vel)

    v_norm = sqrt(q_rel[0] ** 2 + q_rel[1] ** 2 + q_rel[2] ** 2)
    if v_norm < 1e-6:
        u = -local_vel / (ACCEL_MAX * SIM_DT)
        return CarControls(
            pitch=float(np.clip(u[0], -1.0, 1.0)),
            yaw=float(np.clip(u[1], -1.0, 1.0)),
            roll=float(np.clip(u[2], -1.0, 1.0))
        )

    angle = 2.0 * np.arctan2(v_norm, q_rel[3])
    err_pyr = (angle / v_norm) * q_rel[:3]

    # 1. Solve 2D coupled pitch-yaw control
    u_p, u_y = _solve_2d_vector_field(err_pyr[:2], local_vel[:2], ACCEL_MAX[0], ACCEL_MAX[1])

    # 2. Solve 1D roll with dynamic headroom on the 5.5 rad/s sphere
    py_spd_sq = local_vel[0] ** 2 + local_vel[1] ** 2
    roll_spd_limit = MAX_ANG_SPEED if py_spd_sq >= MAX_ANG_SPEED ** 2 else sqrt(MAX_ANG_SPEED ** 2 - py_spd_sq)
    roll_spd_limit = max(0.5, roll_spd_limit)

    eff_a_roll = ACCEL_MAX[2] + DAMPING[2] * abs(local_vel[2]) * TORQUE_APPLY_SCALE
    roll_err = (err_pyr[2] + pi) % (2.0 * pi) - pi
    roll_stop = (local_vel[2] * abs(local_vel[2])) / (2.0 * eff_a_roll) + 0.5 * local_vel[2] * SIM_DT
    roll_delta = roll_err - roll_stop

    t_bl = 1.5 * SIM_DT
    if abs(roll_err) <= 0.5 * eff_a_roll * (t_bl ** 2) and abs(local_vel[2]) <= eff_a_roll * t_bl:
        u_r = (roll_err / t_bl - local_vel[2]) / (eff_a_roll * SIM_DT)
    else:
        u_r = roll_delta / (eff_a_roll * (SIM_DT ** 2) * 1.5)

    return CarControls(
        pitch=u_p,
        yaw=u_y,
        roll=float(np.clip(u_r, -1.0, 1.0))
    )


def solve_dir2d(q0, w0, target_rot, tol_angle=0.05, tol_speed=0.07, max_seconds=6.0):
    car = Car(rot=q0.copy(), ang_vel=w0.copy())
    hist = []
    max_ticks = int(max_seconds * SIM_FPS)
    settled = False
    t_finish = max_seconds

    for tick in range(max_ticks):
        sim_time = tick * SIM_DT
        inner = min(1.0, max(-1.0, abs(float(np.dot(car.rot, target_rot)))))
        angle_rad = 2.0 * np.arccos(inner)
        spd = float(np.linalg.norm(car.ang_vel))

        if not settled and angle_rad < tol_angle and spd < tol_speed:
            settled = True
            t_finish = sim_time

        ctrl = run_rm_dir2d(car, target_rot) if not settled else CarControls(0.0, 0.0, 0.0)
        status = "SETTLED" if settled else "ACTIVE"
        hist.append({
            "rot": car.rot.copy(),
            "u": np.array([ctrl.pitch, ctrl.yaw, ctrl.roll]),
            "t": sim_time,
            "angle_rad": angle_rad,
            "speed": spd,
            "status": status
        })

        if not settled:
            car.step_turn(ctrl, SIM_DT)
        elif tick > int(t_finish / SIM_DT) + 25:
            break

    return t_finish, hist
