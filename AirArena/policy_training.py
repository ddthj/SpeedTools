"""
train_model.py - Trains a zero-bias neural controller on the saved .npz dataset.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

DATASET_FILE = "reorientation_dataset.npz"
MODEL_SAVE_PATH = "satellite_policy.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ZeroBiasPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # bias=False ensures that when input = [0,0,0,0,0,0], output is EXACTLY [0,0,0]
        self.fc1 = nn.Linear(6, 64, bias=False)
        self.fc2 = nn.Linear(64, 64, bias=False)
        self.fc3 = nn.Linear(64, 32, bias=False)
        self.fc4 = nn.Linear(32, 3, bias=False)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return self.act(self.fc4(x))


def train():
    data = np.load(DATASET_FILE)
    X = data["X"]
    Y = data["Y"]
    print(f"Loaded {len(X):,} samples from {DATASET_FILE}")

    # Shuffle & Split
    indices = np.random.permutation(len(X))
    split = int(0.9 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = TensorDataset(torch.tensor(X[train_idx]), torch.tensor(Y[train_idx]))
    val_ds = TensorDataset(torch.tensor(X[val_idx]), torch.tensor(Y[val_idx]))

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = ZeroBiasPolicy().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    # Loss function with sample weighting for the terminal region
    def weighted_loss(pred, target, x_state):
        # Base MSE loss
        err = (pred - target) ** 2

        # State magnitude: small when near target
        state_mag = torch.norm(x_state, dim=-1, keepdim=True)
        # Give 3x weight to states close to settling
        terminal_weights = 1.0 + 2.0 * torch.exp(-state_mag / 0.2)

        return torch.mean(err * terminal_weights)

    best_val_loss = float("inf")
    epochs = 500

    print(f"\n--- Training Zero-Bias Architecture on {DEVICE} ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            pred = model(bx)
            loss = weighted_loss(pred, by, bx)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)

        train_loss /= len(train_idx)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                val_loss += weighted_loss(model(bx), by, bx).item() * len(bx)
        val_loss /= len(val_idx)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

    print(f"\nModel saved to '{MODEL_SAVE_PATH}' (Val Loss: {best_val_loss:.5f})")


if __name__ == "__main__":
    train()
