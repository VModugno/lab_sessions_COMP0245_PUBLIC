import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
Y_tensor = torch.tensor(Y, dtype=torch.float32).view(
    -1, 1
)  # one column and as many rows as needed

# Dataset and DataLoader
dataset = TensorDataset(X_tensor, Y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# ShallowCorrectorMLP Model Definition
class ShallowCorrectorMLP(nn.Module):
    def __init__(self, hidden_nodes):
        super(ShallowCorrectorMLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(4, hidden_nodes), nn.ReLU(), nn.Linear(hidden_nodes, 1)
        )

    def forward(self, x):
        return self.layers(x)


"""
Task 1.2
"""


# Define the DeepCorrectorMLP model with two hidden layers
class DeepCorrectorMLP(nn.Module):
    def __init__(self, num_hidden_nodes):
        super(DeepCorrectorMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, num_hidden_nodes),
            nn.ReLU(),
            nn.Linear(num_hidden_nodes, num_hidden_nodes),
            nn.ReLU(),
            nn.Linear(num_hidden_nodes, 1),
        )

    def forward(self, x):
        return self.layers(x)


def train_and_evaluate(model_class, hidden_nodes, learning_rate):
    # Model, Loss, Optimizer
    model = model_class(hidden_nodes)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    epochs = 1000
    train_losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()  # Gradient of the loss
            optimizer.step()  # Update the model parameters

        epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))
        print(
            f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.6f}"
        )  # the average loss of the most recent epoch
    return model, train_losses


def train_and_test_with_batch_size(
    model_class, batch_size, hidden_nodes, learning_rate
):
    # hidden_nodes=32
    model = model_class(hidden_nodes)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Training Loop
    epochs = 1000
    train_losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for data, target in train_loader:
            start_time = time.time()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()  # Gradient of the loss
            optimizer.step()  # Update the model parameters
            end_time = time.time()
            training_time = end_time - start_time

        epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))
        print(
            f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.6f}"
        )  # the average loss of the most recent epoch

    # Evaluate model on test set
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
    test_losses = test_loss / len(test_loader)

    return training_time, train_losses, test_losses


def PD_control_only(q_target, dot_q_target, t):
    # Testing Phase: Simulate trajectory tracking
    q_test = 0
    dot_q_test = 0
    q_real = []
    # integration with only PD Control
    for i in range(len(t)):
        tau = k_p * (q_target[i] - q_test) + k_d * (
            dot_q_target[i] - dot_q_test
        )
        ddot_q_real = (tau - b * dot_q_test) / m
        dot_q_test += ddot_q_real * dt
        q_test += dot_q_test * dt
        q_real.append(q_test)
    return q_real


def PD_and_MLP_correction(model, q_target, dot_q_target, t):
    q_test = 0
    dot_q_test = 0
    q_real_corrected = []
    for i in range(len(t)):
        # Apply MLP correction
        tau = k_p * (q_target[i] - q_test) + k_d * (
            dot_q_target[i] - dot_q_test
        )
        inputs = torch.tensor(
            [q_test, dot_q_test, q_target[i], dot_q_target[i]],
            dtype=torch.float32,
        )
        correction = model(inputs.unsqueeze(0)).item()
        ddot_q_corrected = (tau - b * dot_q_test + correction) / m
        dot_q_test += ddot_q_corrected * dt
        q_test += dot_q_test * dt
        q_real_corrected.append(q_test)
    return q_real_corrected


"""
Task 1.1
"""
# Increase hidden nodes between 32 and 128 in steps of 32 units
hidden_nodes_range = range(32, 129, 32)
for hidden_nodes in hidden_nodes_range:
    model, train_losses = train_and_evaluate(
        ShallowCorrectorMLP, hidden_nodes, learning_rate=0.00001
    )
    q_real_corrected = PD_and_MLP_correction(model, q_target, dot_q_target, t)
    q_real_baseline = PD_control_only(q_target, dot_q_target, t)
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(t, q_target, "r-", label="Target")
    plt.plot(t, q_real_baseline, "b--", label="PD Only")
    plt.plot(t, q_real_corrected, "g:", label="PD + MLP Correction")
    plt.title(
        f"Trajectory Tracking Comparison with {hidden_nodes} Hidden Nodes"
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Position")
    plt.legend()
    file_path = f"/Users/huangyuting/Machine Learning for Robotics/week_5/figures/task1.1/{hidden_nodes}_hidden_nodes.pdf"
    try:
        plt.savefig(file_path)
        print(
            f"Figure for trajectory with {hidden_nodes} hidden nodes saved successfully"
        )
    except Exception as e:
        print(f"Error saving figure for: {e}")
    plt.show()

"""
Task 1.3
"""
learning_rates = [1.0, 1e-1, 1e-2, 1e-3, 1e-4]
for lr in learning_rates:
    shallow_model, shallow_train_losses = train_and_evaluate(
        ShallowCorrectorMLP, hidden_nodes, lr
    )
    deep_model, deep_train_losses = train_and_evaluate(
        DeepCorrectorMLP, hidden_nodes, lr
    )
    q_real_shallow = PD_and_MLP_correction(
        shallow_model, q_target, dot_q_target, t
    )
    q_real_deep = PD_and_MLP_correction(deep_model, q_target, dot_q_target, t)

    plt.figure(figsize=(12, 6))
    plt.plot(t, q_target, "r-", label="Target")
    plt.plot(t, q_real_shallow, "b--", label=f"Shallow MLP (LR={lr})")
    plt.plot(t, q_real_deep, "g:", label=f"Deep MLP (LR={lr})")
    plt.title(f"Trajectory Tracking Comparison at Learning Rate {lr}")
    plt.xlabel("Time [s]")
    plt.ylabel("Position")
    plt.legend()
    file_path = f"/Users/huangyuting/Machine Learning for Robotics/week_5/figures/task1.3/{lr}_learning_rate.pdf"
    try:
        plt.savefig(file_path)
        print(f"Comparison figure for learning rate {lr} saved successfully")
    except Exception as e:
        print(f"Error saving figure for learning rate {lr}: {e}")

    plt.show()

"""
Task 1.4
"""
batch_sizes = [64, 128, 256, 1000]
results = []
for batch_size in batch_sizes:
    print(f"\nEvaluating ShallowCorrectorMLP with batch size {batch_size}")
    shallow_train_losses, shallow_test_loss, shallow_training_time = (
        train_and_test_with_batch_size(
            ShallowCorrectorMLP,
            batch_size,
            hidden_nodes=32,
            learning_rate=0.00001,
        )
    )

    print(f"\nEvaluating DeepCorrectorMLP with batch size {batch_size}")
    deep_train_losses, deep_test_loss, deep_training_time = (
        train_and_test_with_batch_size(
            DeepCorrectorMLP, batch_size, hidden_nodes=32, learning_rate=0.00001
        )
    )

    # Store results for analysis
    results.append(
        {
            "batch_size": batch_size,
            "shallow_train_losses": shallow_train_losses,
            "shallow_test_loss": shallow_test_loss,
            "shallow_training_time": shallow_training_time,
            "deep_train_losses": deep_train_losses,
            "deep_test_loss": deep_test_loss,
            "deep_training_time": deep_training_time,
        }
    )

# Plot Training Times
plt.figure(figsize=(10, 6))
plt.plot(
    [result["batch_size"] for result in results],
    [result["shallow_training_time"] for result in results],
    label="Shallow MLP Training Time",
)
plt.plot(
    [result["batch_size"] for result in results],
    [result["deep_training_time"] for result in results],
    label="Deep MLP Training Time",
)
plt.xlabel("Batch Size")
plt.ylabel("Training Time (s)")
plt.title("Comparison of Training Time for each Batch Size")
plt.legend()
file_path = f"/Users/huangyuting/Machine Learning for Robotics/week_5/figures/task1.4/Comparison of Training Time for each Batch Size.pdf"
try:
    plt.savefig(file_path)
    print(f"figure for Training Time saved successfully")
except Exception as e:
    print(f"Error saving figure for Training Time: {e}")

plt.show()

# Plot Training Loss for each Batch Size
for result in results:
    # Plot Training Losses over epchos for each Batch Size
    plt.figure(figsize=(10, 6))
    plt.plot(
        [result["shallow_train_losses"] for result in results],
        label="Shallow MLP Train Losses",
    )
    plt.plot(
        [result["deep_train_losses"] for result in results],
        label="Deep MLP Train Losses",
    )
    plt.xlabel("Epchos")
    plt.ylabel("Training Loss")
    plt.title(
        f'Comparison of Training Loss for {result["batch_size"]} Batch Size'
    )
    plt.legend()
    file_path = f'/Users/huangyuting/Machine Learning for Robotics/week_5/figures/task1.4/Comparison of Training Losses over epchos for {result["batch_size"]} Batch Size.pdf'
    try:
        plt.savefig(file_path)
        print(
            f'figure for Training Losses for {result["batch_size"]} Batch Size saved successfully'
        )
    except Exception as e:
        print(
            f'Error saving figure for Training Losses for {result["batch_size"]} Batch Size: {e}'
        )

    plt.show()

    # Plot Test Loss
    plt.figure(figsize=(10, 6))
    plt.plot([result["shallow_test_loss"] for result in results])
    plt.plot([result["deep_test_loss"] for result in results])
    plt.xlabel("Echos")
    plt.ylabel("Test Loss")
    plt.title(f'Comparison of Test Loss for {result["batch_size"]} Batch Size')
    plt.legend()
    file_path = f'/Users/huangyuting/Machine Learning for Robotics/week_5/figures/task1.4/Comparison of Test Losses over epchos for {result["batch_size"]} Batch Size.pdf'
    try:
        plt.savefig(file_path)
        print(
            f'figure for Test Losses for {result["batch_size"]} Batch Size saved successfully'
        )
    except Exception as e:
        print(
            f'Error saving figure for Test Losses for {result["batch_size"]} Batch Size: {e}'
        )

    plt.show()
