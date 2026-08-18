"""
simulator.py - Physical constants, car simulation at 120 FPS, and unified math.
"""
import numpy as np
import casadi as ca
from dataclasses import dataclass
from math import pi, sin, cos, sqrt

# ─── 1. Physical Constants ────────────────────────────────────────────────────
MAX_ANG_SPEED = 5.5
TORQUE = np.array([130., 95., 400.])            # Pitch, Yaw, Roll
DAMPING = np.array([30., 20., 50.])
TORQUE_APPLY_SCALE = (2 * pi) / (1 << 16) * 1000.0
BASIS_QUAT = np.array([0.5, -0.5, -0.5, 0.5])
SIM_FPS = 120.0
SIM_DT = 1.0 / SIM_FPS


def _wrap(vals, is_ca):
    return ca.vertcat(*vals) if is_ca else np.array(vals)


def qmul(a, b):
    """Hamilton product for quaternions [x, y, z, w]. Works with NumPy & CasADi."""
    is_ca = isinstance(a, (ca.SX, ca.MX)) or isinstance(b, (ca.SX, ca.MX))
    ax, ay, az, aw = a[0], a[1], a[2], a[3]
    bx, by, bz, bw = b[0], b[1], b[2], b[3]
    return _wrap([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ], is_ca)


def qrot(q, v):
    """Rotates vector v by unit quaternion q = [x, y, z, w]."""
    is_ca = isinstance(q, (ca.SX, ca.MX)) or isinstance(v, (ca.SX, ca.MX))
    qv = [q[0], q[1], q[2]]
    qw = q[3]
    cx = qv[1]*v[2] - qv[2]*v[1] + qw*v[0]
    cy = qv[2]*v[0] - qv[0]*v[2] + qw*v[1]
    cz = qv[0]*v[1] - qv[1]*v[0] + qw*v[2]
    fx = v[0] + 2.0 * (qv[1]*cz - qv[2]*cy)
    fy = v[1] + 2.0 * (qv[2]*cx - qv[0]*cz)
    fz = v[2] + 2.0 * (qv[0]*cy - qv[1]*cx)
    return _wrap([fx, fy, fz], is_ca)


def quat_from_axis_angle(axis, angle):
    half = angle * 0.5
    s = sin(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, cos(half)])


def slerp(q0, q1, t):
    """Spherical linear interpolation."""
    dot = float(np.dot(q0, q1))
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        res = (1.0 - t) * q0 + t * q1
        return res / np.linalg.norm(res)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    w0 = np.sin((1.0 - t) * theta) / sin_theta
    w1 = np.sin(t * theta) / sin_theta
    res = w0 * q0 + w1 * q1
    return res / np.linalg.norm(res)


def vehicle_dynamics(q, w, u):
    """Continuous ODE: returns (dq/dt, dw/dt)."""
    is_ca = isinstance(q, (ca.SX, ca.MX)) or isinstance(u, (ca.SX, ca.MX))
    direction = qmul(q, BASIS_QUAT)
    inv_dir = _wrap([-direction[0], -direction[1], -direction[2], direction[3]], is_ca)

    torque_body = u * TORQUE
    torque_world = qrot(direction, torque_body)

    abs_fn = ca.fabs if is_ca else abs
    damp_factor = _wrap([1.0 - abs_fn(u[0]), 1.0 - abs_fn(u[1]), 1.0], is_ca)
    w_body = qrot(inv_dir, w)
    damp_pyr = w_body * DAMPING * damp_factor
    damping_world = qrot(direction, damp_pyr)

    dw_dt = (torque_world - damping_world) * TORQUE_APPLY_SCALE
    w_quat = _wrap([w[0], w[1], w[2], 0.0], is_ca)
    dq_dt = 0.5 * qmul(w_quat, q)
    return dq_dt, dw_dt


# ─── 2. Simulator Environment ──────────────────────────────────────────────────
@dataclass
class CarControls:
    pitch: float
    yaw: float
    roll: float


class Car:
    def __init__(self, rot=None, ang_vel=None):
        self.rot = rot if rot is not None else np.array([0., 0., 0., 1.])
        self.ang_vel = ang_vel if ang_vel is not None else np.zeros(3)

    def step_turn(self, ctrls: CarControls, dt: float):
        u = np.array([ctrls.pitch, ctrls.yaw, ctrls.roll])
        dq_dt, dw_dt = vehicle_dynamics(self.rot, self.ang_vel, u)
        self.ang_vel += dw_dt * dt
        self.rot = self.integrate_transform(self.rot, self.ang_vel, dt)
        speed_sq = float(self.ang_vel @ self.ang_vel)
        if speed_sq > MAX_ANG_SPEED ** 2:
            self.ang_vel *= MAX_ANG_SPEED / sqrt(speed_sq)

    @staticmethod
    def integrate_transform(rot, ang_vel, dt):
        angle = min(float(np.linalg.norm(ang_vel)), (pi / 4) / dt)
        half = 0.5 * angle * dt
        scale = (1.0 - dt * dt * 2.0 * 0.020_833_334) * angle * half if angle < 0.001 else sin(half) / angle
        axis = ang_vel * scale
        dorn = np.array([*axis, cos(half)])
        res = qmul(dorn, rot)
        return res / np.linalg.norm(res)


def random_state():
    """Generates uniform random initial attitude and high-spin velocity."""
    u1, u2, u3 = np.random.uniform(0.0, 1.0, 3)
    q = np.array([
        sqrt(1.0 - u1) * sin(2.0 * pi * u2),
        sqrt(1.0 - u1) * cos(2.0 * pi * u2),
        sqrt(u1) * sin(2.0 * pi * u3),
        sqrt(u1) * cos(2.0 * pi * u3)
    ])
    vec = np.random.normal(0.0, 1.0, 3)
    w = (vec / np.linalg.norm(vec)) * np.random.uniform(2.0, MAX_ANG_SPEED)
    return q, w
