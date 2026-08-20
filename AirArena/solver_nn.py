"""
solver_neural.py - Evaluates the trained ZeroBiasPolicy.
"""
import torch
import numpy as np
from simulator import (
    BASIS_QUAT, qmul, qrot, Car, CarControls, SIM_FPS, SIM_DT
)
from policy_training import ZeroBiasPolicy

# Load model onto CPU
_policy = ZeroBiasPolicy()
_policy.load_state_dict(torch.load("satellite_policy.pth", map_location="cpu"))
_policy.eval()


def run_neural_controller(car: Car, target_rot):
    # 1. Transform world state into relative control frame
    dir_curr = qmul(car.rot, BASIS_QUAT)
    dir_tgt = qmul(target_rot, BASIS_QUAT)
    inv_curr = np.array([-dir_curr[0], -dir_curr[1], -dir_curr[2], dir_curr[3]])

    # Shortest path relative quaternion
    q_rel = qmul(inv_curr, dir_tgt)
    if q_rel[3] < 0:
        q_rel = -q_rel

    w_local = qrot(inv_curr, car.ang_vel)

    # 2. 6D Zero-Centered Vector [qx, qy, qz, wx, wy, wz]
    state_6d = torch.tensor(
        [q_rel[0], q_rel[1], q_rel[2], w_local[0], w_local[1], w_local[2]],
        dtype=torch.float32
    )

    # 3. Direct evaluation
    with torch.no_grad():
        u = _policy(state_6d).numpy()

    return CarControls(pitch=float(u[0]), yaw=float(u[1]), roll=float(u[2]))


def solve_neural(q0, w0, target_rot, tol_angle=0.05, tol_speed=0.07, max_seconds=6.0):
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

        ctrl = run_neural_controller(car, target_rot) if not settled else CarControls(0.0, 0.0, 0.0)
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
