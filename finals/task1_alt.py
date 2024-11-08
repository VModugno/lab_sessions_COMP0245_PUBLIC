import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

TASKS = [1.1, 1.2, 1.3, 1.4]  # choose one or multiple from 1.1, 1.2, 1.3, 1.4
TASKS = [1.2]
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
FIG_BASE_DIR = os.path.join(CUR_DIR, "figures")
EXT = "pdf"  # figure extension


# Constants
m = 1.0  # Mass (kg)
b = 10  # Friction coefficient
k_p = 50  # Proportional gain
k_d = 10  # Derivative gain
dt = 0.01  # Time step
num_samples = 1000  # Number of samples in dataset

# Generate synthetic data for trajectory tracking
t = np.linspace(0, 10, num_samples)
q_target = np.sin(t)
dot_q_target = np.cos(t)

# Initial conditions for training data generation
q = 0
dot_q = 0
X = []
Y = []

for i in range(num_samples):
    # PD control output
    tau = k_p * (q_target[i] - q) + k_d * (dot_q_target[i] - dot_q)
    # Ideal motor dynamics (variable mass for realism)
    # m_real = m * (1 + 0.1 * np.random.randn())  # Mass varies by +/-10%
    ddot_q_real = (tau - b * dot_q) / m

    # Calculate error
    ddot_q_ideal = (tau) / m
    ddot_q_error = ddot_q_ideal - ddot_q_real

    # Store data
    X.append([q, dot_q, q_target[i], dot_q_target[i]])
    Y.append(ddot_q_error)

    # Update state
    dot_q += ddot_q_real * dt
    q += dot_q * dt

# Convert data for PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)


# MLP Model Definition
class ShallowCorrectorMLP(nn.Module):
    def __init__(self, num_hidden_units=32):
        super(ShallowCorrectorMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, num_hidden_units),
            nn.ReLU(),
            nn.Linear(num_hidden_units, 1),
        )

    def forward(self, x):
        return self.layers(x)


# MLP Model Definition
class DeepCorrectorMLP(nn.Module):
    def __init__(self, num_hidden_units=32):
        super(DeepCorrectorMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, num_hidden_units),
            nn.ReLU(),
            nn.Linear(num_hidden_units, num_hidden_units),
            nn.ReLU(),
            nn.Linear(num_hidden_units, 1),
        )

    def forward(self, x):
        return self.layers(x)


def train_and_evaluate_model(
    model_class, num_hidden_units=32, lr=1e-1, batch_size=32
):
    model = model_class(num_hidden_units)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)

    train_dataset = TensorDataset(X_tensor, Y_tensor)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    # Training Loop
    epochs = 100
    train_losses = []

    model.train()

    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))
        print(
            f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.6f}, Time: {time.time() - start_time:.2f}s"
        )
    train_time = time.time() - start_time

    # Testing Phase: Simulate trajectory tracking
    q_test = 0
    dot_q_test = 0
    q_real = []
    q_real_corrected = []

    # integration with only PD Control
    for i in range(len(t)):
        tau = k_p * (q_target[i] - q_test) + k_d * (
            dot_q_target[i] - dot_q_test
        )
        ddot_q_real = (tau - b * dot_q_test) / m
        dot_q_test += ddot_q_real * dt
        q_test += dot_q_test * dt
        q_real.append(q_test)

    model.eval()

    q_test = 0
    dot_q_test = 0
    for i in range(len(t)):
        # Apply MLP correction
        tau = k_p * (q_target[i] - q_test) + k_d * (
            dot_q_target[i] - dot_q_test
        )
        inputs = torch.tensor(
            [q_test, dot_q_test, q_target[i], dot_q_target[i]],
            dtype=torch.float32,
        )
        with torch.no_grad():
            correction = model(inputs.unsqueeze(0)).item()
        ddot_q_corrected = (tau - b * dot_q_test) / m + correction
        dot_q_test += ddot_q_corrected * dt
        q_test += dot_q_test * dt
        q_real_corrected.append(q_test)
    return train_losses, train_time, q_real, q_real_corrected


hidden_units_list = range(32, 129, 32)
learning_rates = [1.0, 1e-1, 1e-2, 1e-3, 1e-4]
batch_sizes = [64, 128, 256, 1000]
model_classes = [ShallowCorrectorMLP, DeepCorrectorMLP]


if 1.1 in TASKS:
    print("=" * 20, "Task 1.1", "=" * 20)
    for num_hidden_units in hidden_units_list:
        train_losses, train_time, q_real, q_real_corrected = (
            train_and_evaluate_model(ShallowCorrectorMLP, num_hidden_units)
        )
        FIG_DIR = os.path.join(
            FIG_BASE_DIR, "task1_1_alt", f"hidden_units_{num_hidden_units}"
        )
        os.makedirs(FIG_DIR, exist_ok=True)
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(t, q_target, "r-", label="Target")
        plt.plot(t, q_real, "b--", label="PD Only")
        plt.plot(t, q_real_corrected, "g:", label="PD + MLP Correction")
        plt.title("Trajectory Tracking with and without MLP Correction")
        plt.xlabel("Time [s]")
        plt.ylabel("Position")
        plt.legend()
        plt.savefig(f"{FIG_DIR}/trajectory.{EXT}")

        # Plot the errors
        plt.figure(figsize=(12, 6))
        plt.plot(t, q_target - q_real, "b--", label="PD Only")
        plt.plot(
            t, q_target - q_real_corrected, "g:", label="PD + MLP Correction"
        )
        plt.title("Trajectory Tracking Error with and without MLP Correction")
        plt.xlabel("Time [s]")
        plt.ylabel("Position")
        plt.legend()
        plt.savefig(f"{FIG_DIR}/error.{EXT}")
        plt.close("all")

if 1.2 in TASKS:
    print("=" * 20, "Task 1.2", "=" * 20)
    results = {}
    for num_hidden_units in hidden_units_list:
        results[num_hidden_units] = {}
        for model_class in model_classes:
            class_name = model_class.__name__
            results[num_hidden_units][class_name] = train_and_evaluate_model(
                model_class, num_hidden_units
            )
        FIG_DIR = os.path.join(
            FIG_BASE_DIR, "task1_2_alt", f"hidden_units_{num_hidden_units}"
        )
        os.makedirs(FIG_DIR, exist_ok=True)
        plt.figure(figsize=(12, 6))
        plt.plot(t, q_target, "r-", label="Target")
        for model_name, (_, _, _, q_real_corrected) in results[
            num_hidden_units
        ].items():
            plt.plot(t, q_real_corrected, label=model_name)
        plt.title("Trajectory Tracking with MLP Correction")
        plt.xlabel("Time [s]")
        plt.ylabel("Position")
        plt.legend()
        plt.savefig(f"{FIG_DIR}/trajectory.{EXT}")

        plt.figure(figsize=(12, 6))
        for model_name, (_, _, _, q_real_corrected) in results[
            num_hidden_units
        ].items():
            plt.plot(t, q_target - q_real_corrected, label=model_name)
        plt.title("Trajectory Tracking Error with MLP Correction")
        plt.xlabel("Time [s]")
        plt.ylabel("Position")
        plt.legend()
        plt.savefig(f"{FIG_DIR}/error.{EXT}")
        plt.close("all")
    final_loss = []
    for i, model_class in enumerate(model_classes):
        final_loss.append([])
        for num_hidden_units in hidden_units_list:
            final_loss[i].append(
                results[num_hidden_units][model_class.__name__][0][-1]
            )
    fix, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(final_loss)
    ax.set_xticks(np.arange(len(hidden_units_list)), labels=hidden_units_list)
    ax.set_yticks(
        np.arange(len(model_classes)),
        labels=[m.__name__ for m in model_classes],
    )
    ax.set_xlabel("Hidden Units")
    ax.set_ylabel("Model")
    ax.set_title("Final Loss for Different Models and Hidden Units")
    plt.colorbar(im)
    plt.savefig(f"{FIG_BASE_DIR}/task1_2_alt/final_loss.{EXT}")
    plt.close("all")

if 1.3 in TASKS:
    pass

if 1.4 in TASKS:
    pass
