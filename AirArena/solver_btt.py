"""
solver_btt.py - Bank-to-Turn (BTT) 2D+1D reduced-dimension controller.
"""
from math import pi, sqrt, atan2, cos, sin
import numpy as np
from simulator import (
    TORQUE, DAMPING, TORQUE_APPLY_SCALE, BASIS_QUAT,
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


def _time_optimal_1d(err, vel, a_max, damping=0.0):
    eff_a = a_max + damping * abs(vel) * TORQUE_APPLY_SCALE
    err = (err + pi) % (2.0 * pi) - pi

    # Discrete stopping distance with 0.5 * dt half-step lag correction
    stop_dist = (vel * abs(vel)) / (2.0 * eff_a) + 0.5 * vel * SIM_DT
    delta = err - stop_dist

    # Boundary layer for chatter-free settling
    t_bl = 1.5 * SIM_DT
    deadband = 0.5 * eff_a * (t_bl ** 2)

    if abs(err) <= deadband and abs(vel) <= eff_a * t_bl:
        # Critically damped linear region
        u = (err / t_bl - vel) / (eff_a * SIM_DT)
    else:
        gain = 1.0 / (eff_a * (SIM_DT ** 2) * 1.5)
        u = delta * gain

    return float(np.clip(u, -1.0, 1.0))


def run_rm_btt(car: Car, target_rot):
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

    pitch_err, yaw_err, roll_err = err_pyr[0], err_pyr[1], err_pyr[2]
    pitch_vel, yaw_vel, roll_vel = local_vel[0], local_vel[1], local_vel[2]

    # Transverse turning magnitude & plane angle
    transverse_err = sqrt(pitch_err ** 2 + yaw_err ** 2)
    plane_angle = atan2(yaw_err, pitch_err)

    # Bank-to-Turn logic:
    # When transverse error is significant (> 20 deg), bank into the turn
    # such that pitch is aligned with the required turn vector.
    if transverse_err > 0.35:
        # Find nearest pitch alignment (0 or pi)
        bank_target_offset = atan2(sin(plane_angle), cos(plane_angle))
        if abs(bank_target_offset) > pi / 2:
            bank_target_offset = atan2(sin(plane_angle - pi), cos(plane_angle - pi))

        # Blend roll target from bank alignment (early) to final roll target (near completion)
        alpha = min(1.0, transverse_err / 1.0)
        active_roll_err = alpha * bank_target_offset + (1.0 - alpha) * roll_err
    else:
        active_roll_err = roll_err

    # Compute controls
    pitch_in = _time_optimal_1d(pitch_err, pitch_vel, ACCEL_MAX[0])
    yaw_in = _time_optimal_1d(yaw_err, yaw_vel, ACCEL_MAX[1])
    roll_in = _time_optimal_1d(active_roll_err, roll_vel, ACCEL_MAX[2], DAMPING[2])

    return CarControls(pitch=pitch_in, yaw=yaw_in, roll=roll_in)


def solve_btt(q0, w0, target_rot, tol_angle=0.05, tol_speed=0.07, max_seconds=6.0):
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

        ctrl = run_rm_btt(car, target_rot) if not settled else CarControls(0.0, 0.0, 0.0)
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
