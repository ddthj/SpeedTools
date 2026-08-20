"""
solver_rlutilities.py - Solver interface for RLUtilities ReorientML neural controller.
"""
import numpy as np
from rlutilities.simulation import Car as RLCar
from rlutilities.mechanics import ReorientML
from rlutilities.linear_algebra import mat3, vec3


from simulator import (
    Car, CarControls, SIM_FPS, SIM_DT
)


def _quat_to_rlu_mat3(q):
    """Converts a [qx, qy, qz, qw] quaternion into an RLUtilities mat3 rotation matrix."""
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]

    # 3x3 rotation matrix from unit quaternion
    r00 = 1.0 - 2.0 * (qy ** 2 + qz ** 2)
    r01 = 2.0 * (qx * qy - qz * qw)
    r02 = 2.0 * (qx * qz + qy * qw)

    r10 = 2.0 * (qx * qy + qz * qw)
    r11 = 1.0 - 2.0 * (qx ** 2 + qz ** 2)
    r12 = 2.0 * (qy * qz - qx * qw)

    r20 = 2.0 * (qx * qz - qy * qw)
    r21 = 2.0 * (qy * qz + qx * qw)
    r22 = 1.0 - 2.0 * (qx ** 2 + qy ** 2)

    return mat3(
        r00, r01, r02,
        r10, r11, r12,
        r20, r21, r22
    )


def solve_rlutilities(q0, w0, target_rot, tol_angle=0.05, tol_speed=0.07, max_seconds=6.0):
    """
    Evaluates RLUtilities' C++ ReorientML policy on the simulator.
    """
    # 1. Initialize Simulator Car
    sim_car = Car(rot=q0.copy(), ang_vel=w0.copy())

    # 2. Initialize RLUtilities Car and ReorientML controller
    rl_car = RLCar()
    rl_car.o = _quat_to_rlu_mat3(sim_car.rot)
    rl_car.w = vec3(float(sim_car.ang_vel[0]), float(sim_car.ang_vel[1]), float(sim_car.ang_vel[2]))

    reorient = ReorientML(rl_car)
    reorient.target_orientation = _quat_to_rlu_mat3(target_rot)
    reorient.eps_phi = float(tol_angle)

    hist = []
    max_ticks = int(max_seconds * SIM_FPS)
    settled = False
    t_finish = max_seconds

    for tick in range(max_ticks):
        sim_time = tick * SIM_DT

        # Compute metric tolerances against ground truth
        inner = min(1.0, max(-1.0, abs(float(np.dot(sim_car.rot, target_rot)))))
        angle_rad = 2.0 * np.arccos(inner)
        spd = float(np.linalg.norm(sim_car.ang_vel))

        if not settled and angle_rad < tol_angle and spd < tol_speed:
            settled = True
            t_finish = sim_time

        # Update RLUtilities state from simulator state
        rl_car.o = _quat_to_rlu_mat3(sim_car.rot)
        rl_car.w = vec3(float(sim_car.ang_vel[0]), float(sim_car.ang_vel[1]), float(sim_car.ang_vel[2]))

        if not settled:
            # Step the RLUtilities neural policy
            reorient.step(SIM_DT)
            ctrl = CarControls(
                pitch=float(reorient.controls.pitch),
                yaw=float(reorient.controls.yaw),
                roll=float(reorient.controls.roll)
            )
        else:
            ctrl = CarControls(0.0, 0.0, 0.0)

        status = "SETTLED" if settled else "ACTIVE"
        hist.append({
            "rot": sim_car.rot.copy(),
            "u": np.array([ctrl.pitch, ctrl.yaw, ctrl.roll]),
            "t": sim_time,
            "angle_rad": angle_rad,
            "speed": spd,
            "status": status
        })

        if not settled:
            sim_car.step_turn(ctrl, SIM_DT)
        elif tick > int(t_finish / SIM_DT) + 25:
            break

    return t_finish, hist