import numpy as np
from math import pi, sqrt, atan2
from simulator import (
    MAX_ANG_SPEED, TORQUE, DAMPING, TORQUE_APPLY_SCALE,
    BASIS_QUAT, qmul, qrot, Car, CarControls, SIM_FPS, SIM_DT
)


def _kinematic_solve_1d(error, velocity, torque, damping):
    eff_accel = (torque + damping * abs(velocity)) * TORQUE_APPLY_SCALE
    stopping_dist = (velocity * abs(velocity)) / (2.0 * eff_accel)
    future_error = (error - stopping_dist + pi) % (2.0 * pi) - pi
    gain = 1.0 / (eff_accel * SIM_DT * SIM_DT)
    return float(np.clip(future_error * gain, -1.0, 1.0))


def _geodesic_error(car_rot, target_rot):
    """Return (unit_axis, angle) for the shortest arc from car_rot to target_rot."""
    d = qmul(np.array([-car_rot[0], -car_rot[1], -car_rot[2], car_rot[3]]), target_rot)
    if d[3] < 0:
        d = -d
    v = d[:3]
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    angle = 2.0 * atan2(s, d[3])
    return v / s, angle  # axis in world/inertial frame


def _body_torque_limit_along(axis_body):
    """
    Given a unit direction in the BODY frame, what's the max |torque| we can
    produce along that direction, respecting per-axis saturation?

    If we want torque τ = u * axis_body where u is scalar, and each axis of
    axis_body must satisfy |u * axis_body[i]| <= TORQUE[i], then:
        |u| <= min_i (TORQUE[i] / |axis_body[i]|)  for non-zero components
    """
    limits = np.full(3, np.inf)
    for i in range(3):
        if abs(axis_body[i]) > 1e-9:
            limits[i] = TORQUE[i] / abs(axis_body[i])
    return float(np.min(limits))


def solve_roundy(car: Car, target_rot):
    # --- 1. Geodesic error in the inertial frame ---
    err_axis_world, theta = _geodesic_error(car.rot, target_rot)
    if theta < 1e-6:
        return CarControls(0.0, 0.0, 0.0)

    # --- 2. Decompose angular velocity into along / perpendicular ---
    #    ω is in body frame; err_axis is in world frame.
    #    Rotate err_axis into body frame: R_body = car.rot^{-1} applied to world vector
    inv_rot = np.array([-car.rot[0], -car.rot[1], -car.rot[2], car.rot[3]])
    err_axis_body = qrot(inv_rot, np.array([0.0, 0.0, 0.0, 0.0]) +
                         np.concatenate([err_axis_world, [0.0]]))[:3]
    err_axis_body /= (np.linalg.norm(err_axis_body) + 1e-12)

    v_along = float(np.dot(car.ang_vel, err_axis_body))
    v_perp = car.ang_vel - v_along * err_axis_body

    # --- 3. 1D kinematic solve for the along-geodesic component ---
    #    Effective torque along err_axis_body (accounts for per-axis limits)
    tau_along_max = _body_torque_limit_along(err_axis_body)
    #    Damping along this direction
    damp_along = float(np.dot(DAMPING, err_axis_body * err_axis_body))  # quadratic form approx

    u_along = _kinematic_solve_1d(theta, v_along, tau_along_max, damp_along)

    # --- 4. Steering: kill perpendicular velocity ---
    #    We want to apply torque opposite to v_perp to curve the velocity
    #    back onto the geodesic.  Gain is tuned so that steering doesn't
    #    steal too much budget from the along-axis braking.
    #
    #    Key insight: if |v_perp| is large relative to v_along, we're "wasting"
    #    speed budget.  Prioritize steering.  If v_perp is small, prioritize
    #    the along-axis kinematic solve.
    v_perp_mag = np.linalg.norm(v_perp)
    v_total = np.linalg.norm(car.ang_vel)
    #    Steering gain: proportional to how much of the speed is "sideways"
    steer_gain = 2.0 / max(tau_along_max, 1e-6)  # 1/accel gives time constant
    u_perp_body = -v_perp * steer_gain

    # --- 5. Combine in body frame ---
    u_body = u_along * err_axis_body + u_perp_body

    # --- 6. Saturate per-axis (this is the real constraint) ---
    #    Scale down if any axis would exceed ±1
    max_comp = np.max(np.abs(u_body))
    if max_comp > 1.0:
        u_body /= max_comp

    return CarControls(
        pitch=float(np.clip(u_body[0], -1, 1)),
        yaw=float(np.clip(u_body[1], -1, 1)),
        roll=float(np.clip(u_body[2], -1, 1)),
    )
