"""
train_policy.py - End-to-end dataset generation, symmetry augmentation, and MLP training.
"""
import os
import multiprocessing as mp
from math import pi, sqrt, cos, sin
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import casadi as ca

from simulator import (
    MAX_ANG_SPEED, slerp, quat_from_axis_angle, qmul, qrot, BASIS_QUAT
)

MODEL_FILE = "casadi_solver.casadi"
N_STEPS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SatellitePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [qx, qy, qz, qw, wx, wy, wz] in relative control frame
        # Output: [u_pitch, u_yaw, u_roll] in [-1, 1]
        self.net = nn.Sequential(
            nn.Linear(7, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------
# 2. Parallel CasADi Worker
# ----------------------------------------------------------------------
_worker_solver = None

def _init_worker():
    global _worker_solver
    _worker_solver = ca.Function.load(MODEL_FILE)

def _solve_single_scenario(args):
    q0, w0, target_rot = args
    x0 = np.concatenate([q0, w0])

    # Direct branch SLERP guess
    q_tgt = target_rot if np.dot(q0, target_rot) >= 0 else -target_rot
    w_spd = float(np.linalg.norm(w0))
    inner = min(1.0, max(-1.0, float(abs(np.dot(q0, q_tgt)))))
    delta_angle = 2.0 * np.arccos(inner)
    Tf_init = float(np.clip((w_spd / 22.0) + 2.0 * sqrt(max(0.01, delta_angle) / 22.0), 0.2, 4.5))

    X_init = np.zeros((7, N_STEPS + 1))
    for k in range(N_STEPS + 1):
        ratio = k / N_STEPS
        X_init[0:4, k] = slerp(q0, q_tgt, ratio)
        X_init[4:7, k] = (1.0 - ratio) * w0

    try:
        Tf_val, U_val = _worker_solver(x0, q_tgt, Tf_init, X_init)
        tf = float(Tf_val)
        u_mat = np.array(U_val)

        if 0.05 < tf < 7.0 and not np.isnan(tf):
            # Reconstruct intermediate state trajectory X via forward simulation
            dt = tf / N_STEPS
            curr_x = x0.copy()
            states = [curr_x.copy()]

            # Extract world-frame state nodes
            from simulator import vehicle_dynamics
            for k in range(N_STEPS):
                u_k = u_mat[:, k]
                dq, dw = vehicle_dynamics(curr_x[0:4], curr_x[4:7], u_k)
                # RK4 forward step for accurate intermediate states
                k1_q, k1_w = dq, dw
                k2_q, k2_w = vehicle_dynamics(curr_x[0:4] + 0.5*dt*k1_q, curr_x[4:7] + 0.5*dt*k1_w, u_k)
                k3_q, k3_w = vehicle_dynamics(curr_x[0:4] + 0.5*dt*k2_q, curr_x[4:7] + 0.5*dt*k2_w, u_k)
                k4_q, k4_w = vehicle_dynamics(curr_x[0:4] + dt*k3_q, curr_x[4:7] + dt*k3_w, u_k)

                next_q = curr_x[0:4] + (dt/6.0) * (k1_q + 2*k2_q + 2*k3_q + k4_q)
                next_w = curr_x[4:7] + (dt/6.0) * (k1_w + 2*k2_w + 2*k3_w + k4_w)
                next_q /= np.linalg.norm(next_q)

                curr_x = np.concatenate([next_q, next_w])
                states.append(curr_x.copy())

            return {
                "states": np.array(states),   # (N+1, 7)
                "controls": u_mat.T,           # (N, 3)
                "target": q_tgt
            }
    except Exception:
        pass
    return None


# ----------------------------------------------------------------------
# 3. Stratified Scenario Generator
# ----------------------------------------------------------------------
def generate_stratified_scenarios(num_scenarios):
    scenarios = []

    # Tier 1 (40%): High angular velocity & opposing large angles (Wrap-around / momentum)
    n_tier1 = int(num_scenarios * 0.40)
    for _ in range(n_tier1):
        axis = np.random.normal(0, 1, 3)
        axis /= np.linalg.norm(axis)
        angle = np.random.uniform(pi * 0.6, pi)
        q0 = quat_from_axis_angle(axis, angle)
        # Opposing angular velocity
        w0 = -axis * np.random.uniform(2.5, 5.0) + np.random.normal(0, 0.5, 3)
        w0 = np.clip(w0, -MAX_ANG_SPEED, MAX_ANG_SPEED)
        scenarios.append((q0, w0, np.array([0., 0., 0., 1.])))

    # Tier 2 (35%): Large Yaw error (Teaches Roll-to-Pitch whip)
    n_tier2 = int(num_scenarios * 0.35)
    for _ in range(n_tier2):
        yaw_angle = np.random.uniform(pi * 0.4, pi) * (1 if np.random.rand() > 0.5 else -1)
        q0 = np.array([0.0, sin(yaw_angle / 2.0), 0.0, cos(yaw_angle / 2.0)])
        w0 = np.random.uniform(-2.0, 2.0, 3)
        scenarios.append((q0, w0, np.array([0., 0., 0., 1.])))

    # Tier 3 (25%): Fine settling zone (Teaches precise terminal braking)
    n_tier3 = num_scenarios - n_tier1 - n_tier2
    for _ in range(n_tier3):
        axis = np.random.normal(0, 1, 3)
        axis /= np.linalg.norm(axis)
        angle = np.random.uniform(0.02, 0.25)
        q0 = quat_from_axis_angle(axis, angle)
        w0 = np.random.uniform(-1.0, 1.0, 3)
        scenarios.append((q0, w0, np.array([0., 0., 0., 1.])))

    return scenarios


# ----------------------------------------------------------------------
# 4. Trajectory Unrolling & 8x Parity Symmetry Augmentation
# ----------------------------------------------------------------------
def process_and_augment_trajectories(results):
    X_train, Y_train = [], []

    # 4 Parity sign matrices for [pitch, yaw, roll]
    parity_flips = [
        ( 1,  1,  1),
        (-1,  1, -1),
        ( 1, -1, -1),
        (-1, -1,  1)
    ]

    for traj in results:
        states = traj["states"]      # (N+1, 7)
        controls = traj["controls"]  # (N, 3)
        target_rot = traj["target"]

        for k in range(len(controls)):
            q_world = states[k, 0:4]
            w_world = states[k, 4:7]
            u_ctrl = controls[k]     # [u_pitch, u_yaw, u_roll]

            # Transform into control frame relative state
            dir_curr = qmul(q_world, BASIS_QUAT)
            dir_tgt = qmul(target_rot, BASIS_QUAT)
            inv_curr = np.array([-dir_curr[0], -dir_curr[1], -dir_curr[2], dir_curr[3]])

            q_rel = qmul(inv_curr, dir_tgt)
            if q_rel[3] < 0:
                q_rel = -q_rel
            w_local = qrot(inv_curr, w_world)

            qx, qy, qz, qw = q_rel[0], q_rel[1], q_rel[2], q_rel[3]
            wx, wy, wz = w_local[0], w_local[1], w_local[2]
            up, uy, ur = u_ctrl[0], u_ctrl[1], u_ctrl[2]

            # Apply 8x Symmetries (4 parities x 2 antipodal signs)
            for sp, sy, sr in parity_flips:
                # 1. Positive quaternion representation
                x_aug1 = [qx*sp, qy*sy, qz*sr, qw, wx*sp, wy*sy, wz*sr]
                u_aug1 = [up*sp, uy*sy, ur*sr]
                X_train.append(x_aug1)
                Y_train.append(u_aug1)

                # 2. Antipodal quaternion (-q represents the same physical rotation)
                x_aug2 = [-qx*sp, -qy*sy, -qz*sr, -qw, wx*sp, wy*sy, wz*sr]
                X_train.append(x_aug2)
                Y_train.append(u_aug1)

    return np.array(X_train, dtype=np.float32), np.array(Y_train, dtype=np.float32)


# ----------------------------------------------------------------------
# 5. Model Training Loop
# ----------------------------------------------------------------------
def train_policy(X_data, Y_data, epochs=60, batch_size=256, lr=1e-3):
    print(f"\n--- Training Policy on {len(X_data):,} Augmented Samples ({DEVICE}) ---")

    # Shuffle & 90/10 Train-Val Split
    indices = np.random.permutation(len(X_data))
    split = int(0.9 * len(X_data))
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = TensorDataset(torch.tensor(X_data[train_idx]), torch.tensor(Y_data[train_idx]))
    val_ds = TensorDataset(torch.tensor(X_data[val_idx]), torch.tensor(Y_data[val_idx]))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SatellitePolicy().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)

        train_loss /= len(train_idx)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                val_loss += criterion(model(bx), by).item() * len(bx)
        val_loss /= len(val_idx)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "satellite_policy.pth")

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

    print(f"\nBest Model Saved to 'satellite_policy.pth' (Val Loss: {best_val_loss:.5f})")


# ----------------------------------------------------------------------
# 6. Main Execution Pipeline
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(MODEL_FILE):
        print(f"Error: {MODEL_FILE} not found. Please run generate_solver.py first.")
        exit(1)

    # x initial scenarios * 25 steps * 8 symmetries =
    NUM_SCENARIOS = 500
    print(f"1. Sampling {NUM_SCENARIOS} Stratified Scenarios...")
    scenarios = generate_stratified_scenarios(NUM_SCENARIOS)

    print(f"2. Solving via CasADi across {mp.cpu_count()} CPU cores...")
    with mp.Pool(processes=mp.cpu_count(), initializer=_init_worker) as pool:
        raw_results = pool.map(_solve_single_scenario, scenarios, chunksize=25)

    valid_trajs = [r for r in raw_results if r is not None]
    print(f"   Successfully solved {len(valid_trajs)} / {NUM_SCENARIOS} trajectories.")

    print("3. Unrolling trajectories & applying 8x physical symmetries...")
    X_train, Y_train = process_and_augment_trajectories(valid_trajs)
    print(f"   Generated {len(X_train):,} total training state-action pairs.")

    print("4. Training Neural Policy...")
    train_policy(X_train, Y_train, epochs=600, batch_size=256)
