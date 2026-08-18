import numpy as np
from dataclasses import dataclass

# Aerial Control simulation adapted from Samuel Mish's excellent article:
# https://www.smish.dev/rocket_league/aerial_control/

# --- Constants ---
OMEGA_MAX = 5.5

# Torque coefficients
T_r = -36.07956616966136  # for roll
T_p = -12.14599781908070  # for pitch
T_y = 8.91962804287785  # for yaw

# Drag coefficients
D_r = -4.47166302201591  # for roll
D_p = -2.798194258050845  # for pitch
D_y = -1.886491900437232  # for yaw


# --- State Representation ---
@dataclass
class State:
    """
    Represents the physical state of the airborne object.

    Attributes:
        omega (np.ndarray): Angular velocity vector in world coordinates (shape: 3,).
        theta (np.ndarray): Orientation matrix (3x3 rotation matrix).
    """
    omega: np.ndarray
    theta: np.ndarray


def aerial_control(current: State, roll: float, pitch: float, yaw: float, dt: float) -> State:
    """
    Simulates one time step of the object's rotational dynamics.

    Args:
        current (State): The current state (omega, theta) of the object.
        roll (float): Control input for roll, typically in [-1, 1].
        pitch (float): Control input for pitch, typically in [-1, 1].
        yaw (float): Control input for yaw, typically in [-1, 1].
        dt (float): The time step for the simulation (e.g., 1/60.0).

    Returns:
        State: The new state of the object after the time step.
    """

    # Torque and Drag matrices in the object's local (body) coordinates
    T = np.diag([T_r, T_p, T_y])

    D = np.diag([
        D_r,
        D_p * (1.0 - abs(pitch)),
        D_y * (1.0 - abs(yaw))
    ])

    # --- Compute Net Torque ---
    # 1. Convert world angular velocity to body-local angular velocity
    omega_body = current.theta.T @ current.omega

    # 2. Calculate drag torque in body coordinates (opposes local angular velocity)
    drag_torque_body = D @ omega_body

    # 3. Calculate control torque from inputs in body coordinates
    control_inputs = np.array([roll, pitch, yaw])
    control_torque_body = T @ control_inputs

    # 4. Sum torques in body coordinates and convert back to world coordinates
    net_torque_body = drag_torque_body + control_torque_body
    tau = current.theta @ net_torque_body

    # --- Update Angular Velocity ---
    # Use the net torque (in world coords) to get the next angular velocity
    omega_next = current.omega + tau * dt

    # Prevent the angular velocity from exceeding a threshold
    current_norm = np.linalg.norm(current.omega)
    if current_norm > 0:
        # This clamping logic uses the *current* omega's norm as the basis for scaling,
        # matching the original code's `norm(omega)`.
        scale = min(1.0, OMEGA_MAX / current_norm)
        omega_next *= scale

    # --- Update Orientation Matrix ---
    # Compute the average angular velocity for this step for a more stable integration
    omega_avg = 0.5 * (current.omega + omega_next)

    # Total angle of rotation for this step
    phi = np.linalg.norm(omega_avg) * dt

    # Avoid division by zero if there's no rotation
    if phi < 1e-9:  # A small epsilon to handle floating point inaccuracies
        # No rotation, the rotation matrix is the identity matrix
        R = np.identity(3)
    else:
        # This is Rodrigues' rotation formula, derived from the matrix exponential
        # It creates a rotation matrix R that rotates by angle `phi` around the `omega_avg` axis.

        # 1. Get the axis of rotation
        axis = omega_avg / np.linalg.norm(omega_avg)

        # 2. Create the skew-symmetric cross-product matrix for `axis * phi`
        w = axis * phi
        Omega_dt = np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0]
        ])

        # 3. Apply Rodrigues' formula
        R = np.identity(3) \
            + (np.sin(phi) / phi) * Omega_dt \
            + ((1.0 - np.cos(phi)) / (phi * phi)) * (Omega_dt @ Omega_dt)

    # Update the orientation by applying the rotation R
    theta_next = R @ current.theta

    return State(omega=omega_next, theta=theta_next)


def detumble(current: State, desired: State) -> (float, float, float):
    pass
