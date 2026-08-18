"""
solver_kinematic.py - Port of RM_Kinematic controller from Methods.cpp.
"""
import numpy as np
from math import pi, sin, sqrt
from simulator import (
    MAX_ANG_SPEED, TORQUE, DAMPING, TORQUE_APPLY_SCALE, BASIS_QUAT,
    qmul, qrot, Car, CarControls, SIM_FPS, SIM_DT
)


def _get_control_frame_errors(car_rot, target_rot):
    dir_curr = qmul(car_rot, BASIS_QUAT)
    dir_tgt = qmul(target_rot, BASIS_QUAT)
    inv_curr = np.array([-dir_curr[0], -dir_curr[1], -dir_curr[2], dir_curr[3]])
    q_rel = qmul(inv_curr, dir_tgt)
    if q_rel[3] < 0:
        q_rel = -q_rel
    v_norm = sqrt(q_rel[0]**2 + q_rel[1]**2 + q_rel[2]**2)
    if v_norm < 1e-6:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(v_norm, q_rel[3])
    return (angle / v_norm) * q_rel[:3]


def _kinematic_solve_1d(error, velocity, torque, damping):
    eff_accel = (torque + damping * abs(velocity)) * TORQUE_APPLY_SCALE
    stopping_dist = (velocity * abs(velocity)) / (2.0 * eff_accel)
    future_error = (error - stopping_dist + pi) % (2.0 * pi) - pi
    gain = 1.0 / (eff_accel * SIM_DT * SIM_DT)
    return float(np.clip(future_error * gain, -1.0, 1.0))


def run_rm_kinematic(car: Car, target_rot):
    direction = qmul(car.rot, BASIS_QUAT)
    inv_dir = np.array([-direction[0], -direction[1], -direction[2], direction[3]])
    local_ang_vel = qrot(inv_dir, car.ang_vel)

    pitch_vel, yaw_vel, roll_vel = local_ang_vel[0], local_ang_vel[1], local_ang_vel[2]
    err_pyr = _get_control_frame_errors(car.rot, target_rot)
    pitch_err, yaw_err, roll_err = err_pyr[0], err_pyr[1], err_pyr[2]

    pitch_in = _kinematic_solve_1d(pitch_err, pitch_vel, TORQUE[0], 0.0)
    yaw_in   = _kinematic_solve_1d(yaw_err,   yaw_vel,   TORQUE[1], 0.0)

    roll_threshold = pi * 0.56
    pitch_yaw_spd = sqrt(pitch_vel**2 + yaw_vel**2)
    roll_budget = max(0.0, 1.0 - pitch_yaw_spd / MAX_ANG_SPEED)
    pitch_yaw_err = sqrt(pitch_err**2 + yaw_err**2)
    err_gate = float(np.clip(1.0 - pitch_yaw_err / roll_threshold, 0.0, 1.0))
    gated_roll_err = roll_err * (0.45 * roll_budget + 0.55 * err_gate * err_gate)

    roll_in = _kinematic_solve_1d(gated_roll_err, roll_vel, TORQUE[2], DAMPING[2])
    return CarControls(pitch=pitch_in, yaw=yaw_in, roll=roll_in)


def solve_kinematic(q0, w0, target_rot, tol_angle=0.05, tol_speed=0.07, max_seconds=6.0):
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

        ctrl = run_rm_kinematic(car, target_rot) if not settled else CarControls(0.0, 0.0, 0.0)
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