"""
solver_roundy.py - Fast 3D time-optimal controller with discrete boundary layer settling.
"""
from math import pi, sqrt
import numpy as np
from simulator import (
    MAX_ANG_SPEED, TORQUE, DAMPING, TORQUE_APPLY_SCALE, BASIS_QUAT,
    qmul, qrot, Car, CarControls, SIM_FPS, SIM_DT
)


def _get_control_frame(car_rot, target_rot, car_ang_vel):
    """Transforms orientation error and angular velocity into the control basis frame."""
    dir_curr = qmul(car_rot, BASIS_QUAT)
    dir_tgt = qmul(target_rot, BASIS_QUAT)
    inv_curr = np.array([-dir_curr[0], -dir_curr[1], -dir_curr[2], dir_curr[3]])

    # Relative rotation in control frame
    q_rel = qmul(inv_curr, dir_tgt)
    if q_rel[3] < 0:
        q_rel = -q_rel

    # Local angular velocity in control frame
    local_ang_vel = qrot(inv_curr, car_ang_vel)
    return q_rel, local_ang_vel


def _smooth_bang_coast_brake(err, vel, torque, damping, v_max):
    """
    Time-optimal 1D control with discrete-step compensation and
    a critically damped boundary layer to eliminate chattering.
    """
    eff_accel = (torque + damping * abs(vel)) * TORQUE_APPLY_SCALE

    # Wrap error to [-pi, pi]
    err = (err + pi) % (2.0 * pi) - pi

    # Wrap-around check: If moving fast away from target, wrapping around 2pi is faster
    if vel > 2.5 and err < -1.2:
        err += 2.0 * pi
    elif vel < -2.5 and err > 1.2:
        err -= 2.0 * pi

    # Boundary layer thickness (distance covered in ~2.5 ticks of deceleration)
    # Inside this layer, we switch from sqrt() profile to linear critical damping
    e_bl = 0.5 * eff_accel * (2.5 * SIM_DT) ** 2

    # Target velocity generation
    if abs(err) <= e_bl:
        # Critically damped linear region: v_target proportional to error
        v_target = (err / (2.5 * SIM_DT))
    else:
        # Time-optimal sqrt deceleration curve with discrete half-step offset
        sign = 1.0 if err > 0 else -1.0
        # Offset by e_bl so sqrt matches the linear slope continuously at the boundary
        v_target = sign * sqrt(2.0 * eff_accel * (abs(err) - 0.5 * e_bl))

    # Clamp target velocity to max speed ceiling
    v_target = float(np.clip(v_target, -v_max, v_max))

    # High-gain tracking of the target velocity profile
    # Commands max acceleration |u| = 1.0 until vel matches v_target
    u = (v_target - vel) / (eff_accel * SIM_DT)
    return float(np.clip(u, -1.0, 1.0))


def run_rm_vector_field(car: Car, target_rot):
    q_rel, local_ang_vel = _get_control_frame(car.rot, target_rot, car.ang_vel)

    v_norm = sqrt(q_rel[0]**2 + q_rel[1]**2 + q_rel[2]**2)
    if v_norm < 1e-6:
        err_pyr = np.zeros(3)
    else:
        angle = 2.0 * np.arctan2(v_norm, q_rel[3])
        err_pyr = (angle / v_norm) * q_rel[:3]

    pitch_err, yaw_err, roll_err = err_pyr[0], err_pyr[1], err_pyr[2]
    pitch_vel, yaw_vel, roll_vel = local_ang_vel[0], local_ang_vel[1], local_ang_vel[2]

    # Dynamically allocate velocity headroom on the 5.5 rad/s sphere
    pitch_in = _smooth_bang_coast_brake(pitch_err, pitch_vel, TORQUE[0], 0.0, MAX_ANG_SPEED)
    yaw_in   = _smooth_bang_coast_brake(yaw_err,   yaw_vel,   TORQUE[1], 0.0, MAX_ANG_SPEED)

    # Allow roll to run at full speed, capped dynamically by pitch/yaw speed consumption
    py_spd_sq = pitch_vel**2 + yaw_vel**2
    roll_spd_limit = MAX_ANG_SPEED if py_spd_sq >= MAX_ANG_SPEED**2 else sqrt(MAX_ANG_SPEED**2 - py_spd_sq)
    roll_spd_limit = max(0.5, roll_spd_limit)

    roll_in = _smooth_bang_coast_brake(roll_err, roll_vel, TORQUE[2], DAMPING[2], roll_spd_limit)

    return CarControls(pitch=pitch_in, yaw=yaw_in, roll=roll_in)


def solve_roundy(q0, w0, target_rot, tol_angle=0.05, tol_speed=0.07, max_seconds=6.0):
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

        ctrl = run_rm_vector_field(car, target_rot) if not settled else CarControls(0.0, 0.0, 0.0)
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
