"""
solver_s2s1.py - S^2 Pointing + S^1 Torsion Reduced Attitude Controller.
"""
from math import pi, sqrt, atan2, acos, sin, cos
import numpy as np
from simulator import (
    MAX_ANG_SPEED, TORQUE, DAMPING, TORQUE_APPLY_SCALE, BASIS_QUAT,
    qmul, qrot, Car, CarControls, SIM_FPS, SIM_DT
)

ACCEL_MAX = TORQUE * TORQUE_APPLY_SCALE


def _get_control_frame_velocities(car_rot, car_ang_vel):
    dir_curr = qmul(car_rot, BASIS_QUAT)
    inv_curr = np.array([-dir_curr[0], -dir_curr[1], -dir_curr[2], dir_curr[3]])
    return qrot(inv_curr, car_ang_vel)


def _time_optimal_1d_step(err, vel, a_max, damping=0.0):
    eff_a = a_max + damping * abs(vel) * TORQUE_APPLY_SCALE
    err = (err + pi) % (2.0 * pi) - pi

    stop_dist = (vel * abs(vel)) / (2.0 * eff_a) + 0.5 * vel * SIM_DT
    delta = err - stop_dist

    t_bl = 1.5 * SIM_DT
    deadband = 0.5 * eff_a * (t_bl ** 2)

    if abs(err) <= deadband and abs(vel) <= eff_a * t_bl:
        u = (err / t_bl - vel) / (eff_a * SIM_DT)
    else:
        u = delta / (eff_a * (SIM_DT ** 2) * 1.5)

    return float(np.clip(u, -1.0, 1.0))


def run_rm_s2s1(car: Car, target_rot):
    local_vel = _get_control_frame_velocities(car.rot, car.ang_vel)

    # 1. S^2 Pointing Error: Forward direction vectors
    dir_curr = qmul(car.rot, BASIS_QUAT)
    dir_tgt = qmul(target_rot, BASIS_QUAT)

    # Nose vectors (X-axis in control frame = [1, 0, 0], or forward axis)
    # Pitch/Yaw rotates around Y and Z or X and Y depending on basis convention
    inv_curr = np.array([-dir_curr[0], -dir_curr[1], -dir_curr[2], dir_curr[3]])
    q_rel = qmul(inv_curr, dir_tgt)
    if q_rel[3] < 0:
        q_rel = -q_rel

    v_norm = sqrt(q_rel[0]**2 + q_rel[1]**2 + q_rel[2]**2)
    if v_norm < 1e-6:
        err_pyr = np.zeros(3)
    else:
        angle = 2.0 * np.arctan2(v_norm, q_rel[3])
        err_pyr = (angle / v_norm) * q_rel[:3]

    # Transverse pointing error (S^2 sphere geodesic)
    pitch_in = _time_optimal_1d_step(err_pyr[0], local_vel[0], ACCEL_MAX[0])
    yaw_in   = _time_optimal_1d_step(err_pyr[1], local_vel[1], ACCEL_MAX[1])

    # Axial roll error (S^1 circle geodesic around pointing axis)
    roll_in  = _time_optimal_1d_step(err_pyr[2], local_vel[2], ACCEL_MAX[2], DAMPING[2])

    return CarControls(pitch=pitch_in, yaw=yaw_in, roll=roll_in)


def solve_s2s1(q0, w0, target_rot, tol_angle=0.05, tol_speed=0.07, max_seconds=6.0):
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

        ctrl = run_rm_s2s1(car, target_rot) if not settled else CarControls(0.0, 0.0, 0.0)
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
