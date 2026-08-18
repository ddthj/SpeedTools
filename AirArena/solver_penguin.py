"""
solver_align_to.py - Port of Align_Car_To vector-alignment controller.
"""
import numpy as np
from math import atan2
from simulator import qrot, Car, CarControls, SIM_FPS, SIM_DT


def run_align_car_to(car: Car, target_rot):
    target_fwd = qrot(target_rot, np.array([1.0, 0.0, 0.0]))
    target_up = qrot(target_rot, np.array([0.0, 0.0, 1.0]))

    inv_rot = np.array([-car.rot[0], -car.rot[1], -car.rot[2], car.rot[3]])
    align_local = qrot(inv_rot, target_fwd)
    local_up = qrot(inv_rot, target_up)
    local_euler = qrot(inv_rot, car.ang_vel)

    rot_ang_const = 0.25
    stick_correct = 6.0

    a1 = atan2(align_local[1], align_local[0])
    a2 = atan2(align_local[2], align_local[0])
    a3 = 0.0 if (abs(local_up[1]) < 1e-6 and abs(local_up[2]) < 1e-6) else atan2(local_up[1], local_up[2])

    yaw = (0.0 - (-a1 + local_euler[2] * rot_ang_const)) * stick_correct
    pitch = (0.0 - (-a2 - local_euler[1] * rot_ang_const)) * stick_correct
    roll = (0.0 - (-a3 - local_euler[0] * rot_ang_const)) * stick_correct

    return CarControls(
        pitch=float(np.clip(pitch, -1.0, 1.0)),
        yaw=float(np.clip(yaw, -1.0, 1.0)),
        roll=float(np.clip(roll, -1.0, 1.0))
    )


def solve_penguin(q0, w0, target_rot, tol_angle=0.05, tol_speed=0.07, max_seconds=6.0):
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

        ctrl = run_align_car_to(car, target_rot) if not settled else CarControls(0.0, 0.0, 0.0)
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
