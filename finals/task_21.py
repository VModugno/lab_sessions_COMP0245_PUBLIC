import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Dataset

TASK = 2.1
CUR_DIR = os.path.dirname(os.path.realpath(__file__))  # current directory
DIR = os.path.join(CUR_DIR, "figures", f"task{TASK}")  # figure directory
EXT = "pdf"  # figure extension
os.makedirs(DIR, exist_ok=True)  # create figure directory if not exist
print(f"Performing Task {TASK}...")


# Set the visualization flag
visualize = True  # Set to True to enable visualization, False to disable
training_flag = True  # Set to True to train the models, False to skip training
test_cartesian_accuracy_flag = True  # Set to True to test the model with a new goal position, False to skip testing


# MLP Model Definition
class JointAngleRegressor(nn.Module):
    def __init__(self, hidden_units=128):
        super(JointAngleRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(
                4, hidden_units
            ),  # Input layer to hidden layer (4 inputs: time + goal positions)
            nn.ReLU(),
            nn.Linear(hidden_units, 1),  # Hidden layer to output layer
        )

    def forward(self, x):
        return self.model(x)

start_time = time.time()

if training_flag:
    # Load the saved data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(
        script_dir, "data.pkl"
    )  # Replace with your actual filename

    # Check if the file exists
    if not os.path.isfile(filename):
        print(f"Error: File {filename} not found in {script_dir}")
    else:
        with open(filename, "rb") as f:
            data = pickle.load(f)

        # Extract data
        time_array = np.array(data["time"])  # Shape: (N,)
        q_mes_all = np.array(data["q_mes_all"])  # Shape: (N, 7)
        goal_positions = np.array(data["goal_positions"])  # Shape: (N, 3)

        # Optional: Normalize time data for better performance
        # time_array = (time_array - time_array.min()) / (time_array.max() - time_array.min())

        # Custom Dataset Class
        class JointDataset(Dataset):
            def __init__(self, time_data, goal_data, joint_data):
                # Combine time and goal data to form the input features
                x = np.hstack(
                    (time_data.reshape(-1, 1), goal_data)
                )  # Shape: (N, 4)
                self.x_data = torch.from_numpy(x).float()
                self.y_data = (
                    torch.from_numpy(joint_data).float().unsqueeze(1)
                )  # Shape: (N, 1)

            def __len__(self):
                return len(self.x_data)

            def __getitem__(self, idx):
                return self.x_data[idx], self.y_data[idx]

        # Split ratio
        split_ratio = 0.8

        # Initialize lists to hold datasets and data loaders for all joints
        train_loaders = []
        test_loaders = []
        x_train_list = []
        x_test_list = []
        y_train_list = []
        y_test_list = []
        goal_train_list = []
        goal_test_list = []

        for joint_idx in range(7):
            # Extract joint data
            joint_positions = q_mes_all[:, joint_idx]  # Shape: (N,)

            # Split data
            (
                x_train_time,
                x_test_time,
                y_train,
                y_test,
                goal_train,
                goal_test,
            ) = train_test_split(
                time_array,
                joint_positions,
                goal_positions,
                train_size=split_ratio,
                shuffle=True,
            )

            # Store split data for visualization
            x_train_list.append(x_train_time)
            x_test_list.append(x_test_time)
            y_train_list.append(y_train)
            y_test_list.append(y_test)
            goal_train_list.append(goal_train)
            goal_test_list.append(goal_test)

            # Create datasets
            train_dataset = JointDataset(x_train_time, goal_train, y_train)
            test_dataset = JointDataset(x_test_time, goal_test, y_test)

            # Create data loaders
            train_loader = DataLoader(
                dataset=train_dataset, batch_size=32, shuffle=True
            )
            test_loader = DataLoader(
                dataset=test_dataset, batch_size=32, shuffle=False
            )

            # Store loaders
            train_loaders.append(train_loader)
            test_loaders.append(test_loader)

        # Training parameters
        epochs = 500
        learning_rate = 0.01

        for joint_idx in range(7):

            # The name of the saved model
            model_filename = os.path.join(
                script_dir, f"neuralq{joint_idx+1}.pt"
            )

            # If the save model file exists, assume it's been trained already and skip training it
            if os.path.isfile(model_filename):
                print(f"File {model_filename} exists; assume trained already")
                continue

            print(f"\nTraining model for Joint {joint_idx+1}")

            # Initialize the model, criterion, and optimizer
            model = JointAngleRegressor()
            criterion = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)

            train_loader = train_loaders[joint_idx]
            test_loader = test_loaders[joint_idx]

            train_losses = []
            test_losses = []

            # Training loop
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0

                for data, target in train_loader:
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                train_loss = epoch_loss / len(train_loader)
                train_losses.append(train_loss)

                # Evaluate on test set for this epoch
                model.eval()
                test_loss = 0
                with torch.no_grad():
                    for data, target in test_loader:
                        output = model(data)
                        loss = criterion(output, target)
                        test_loss += loss.item()
                test_loss /= len(test_loader)
                test_losses.append(test_loss)

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(
                        f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}"
                    )

            # Final evaluation on test set
            print(
                f"Final Test Loss for Joint {joint_idx+1}: {test_losses[-1]:.6f}"
            )

            # Save the trained model
            model_filename = os.path.join(
                script_dir, f"neuralq{joint_idx+1}.pt"
            )
            torch.save(model.state_dict(), model_filename)
            print(f"Model for Joint {joint_idx+1} saved as {model_filename}")

            # Visualization (if enabled)
            if visualize:
                print(f"Visualizing results for Joint {joint_idx+1}...")

                # Plot training and test loss over epochs
                plt.figure(figsize=(10, 5))
                plt.plot(
                    range(1, epochs + 1), train_losses, label="Training Loss"
                )
                plt.plot(range(1, epochs + 1), test_losses, label="Test Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"Loss Curve for Joint {joint_idx+1}")
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{DIR}/losses_joint_{joint_idx+1}.{EXT}")             
                plt.show()

                # Plot true vs predicted positions on the test set
                model.eval()
                with torch.no_grad():
                    x_test_time = x_test_list[joint_idx]
                    y_test = y_test_list[joint_idx]
                    goal_test = goal_test_list[joint_idx]
                    x_test = np.hstack((x_test_time.reshape(-1, 1), goal_test))
                    x_test_tensor = torch.from_numpy(x_test).float()
                    predictions = model(x_test_tensor).numpy().flatten()

                # Sort the test data for better visualization
                sorted_indices = np.argsort(x_test_time)
                x_test_time_sorted = x_test_time[sorted_indices]
                y_test_sorted = y_test[sorted_indices]
                predictions_sorted = predictions[sorted_indices]

                plt.figure(figsize=(10, 5))
                plt.plot(
                    x_test_time_sorted,
                    y_test_sorted,
                    label="True Joint Positions",
                )
                plt.plot(
                    x_test_time_sorted,
                    predictions_sorted,
                    label="Predicted Joint Positions",
                    linestyle="--",
                )
                plt.xlabel("Time (s)")
                plt.ylabel("Joint Position (rad)")
                plt.title(
                    f"Joint {joint_idx+1} Position Prediction on Test Set"
                )
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{DIR}/test_pos_prediction_joint_{joint_idx+1}.{EXT}") 
                plt.show()

        print("Training and visualization completed.")

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

if test_cartesian_accuracy_flag:

    if not training_flag:
        # Load the saved data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(
            script_dir, "data.pkl"
        )  # Replace with your actual filename
        if not os.path.isfile(filename):
            print(f"Error: File {filename} not found in {script_dir}")
        else:
            with open(filename, "rb") as f:
                data = pickle.load(f)

            # Extract data
            time_array = np.array(data["time"])  # Shape: (N,)
            # q_mes_all = np.array(data['q_mes_all'])        # Shape: (N, 7)
            # goal_positions = np.array(data['goal_positions'])  # Shape: (N, 3)

            # Optional: Normalize time data for better performance
            # time_array = (time_array - time_array.min()) / (time_array.max() - time_array.min())

    # Testing with a new goal position
    print("\nTesting the model with a new goal position...")

    # load all the model in a list of models
    models = []
    for joint_idx in range(7):
        # Instantiate the model
        model = JointAngleRegressor()

        # Load the saved model
        model_filename = os.path.join(script_dir, f"neuralq{joint_idx+1}.pt")
        try:
            model.load_state_dict(
                torch.load(model_filename, weights_only=False)
            )

        except FileNotFoundError:
            print(f"Cannot find file {model_filename}")
            print(
                "task_21_goal_pos needs to be run at least once with training_flag=True"
            )
            quit()

        model.eval()
        models.append(model)

    # Generate a new goal position
    goal_position_bounds = {
        "x": (0.6, 0.8),
        "y": (-0.1, 0.1),
        "z": (0.12, 0.12),
    }
    # create a set of goal positions
    number_of_goal_positions_to_test = 10
    # goal_positions = []
    # for i in range(number_of_goal_positions_to_test):
    #     goal_positions.append(
    #         [
    #             np.random.uniform(*goal_position_bounds["x"]),
    #             np.random.uniform(*goal_position_bounds["y"]),
    #             np.random.uniform(*goal_position_bounds["z"]),
    #         ]
    #     )

    goal_positions = [[0.691397925624174, 0.002102621718463246, 0.12], [0.7098937836805903, -0.08694438235835333, 0.12], [0.7672410862743346, 0.035875201286842656, 0.12], [0.6627737729819734, 0.046329614423186144, 0.12], [0.6539269657617497, 0.00494133901407659, 0.12], [0.6132856638707109, -0.0991282973197782, 0.12], [0.7156702852353941, -0.003447701778067927, 0.12], [0.6054355103671692, 0.005484734697656202, 0.12], [0.7837080889282269, -0.030574751431943464, 0.12], [0.6534992783237846, -0.04376744160873707, 0.12]]
    pos_errs = []

    # Generate test time array
    test_time_array = np.linspace(
        time_array.min(), time_array.max(), 100
    )  # For example, 100 time steps

    # Initialize the dynamic model
    from simulation_and_control import (
        CartesianDiffKin,
        MotorCommands,
        PinWrapper,
        feedback_lin_ctrl,
        pb,
    )

    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    root_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjust root directory if necessary
    name_current_directory = "tests"
    root_dir = root_dir.replace(name_current_directory, "")
    # Initialize simulation interface
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=root_dir, use_gui=False)

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(
        np.array(ext_names), axis=0
    )  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(
        conf_file_name, "pybullet", ext_names, source_names, False, 0, root_dir
    )
    num_joints = dyn_model.getNumberofActuatedJoints()

    controlled_frame_name = "panda_link8"
    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos, init_R = dyn_model.ComputeFK(
        init_joint_angles, controlled_frame_name
    )
    print(f"Initial joint angles: {init_joint_angles}")

    for ii, goal_position in enumerate(goal_positions):
        print(f"Testing {ii+1} goal position------------------------------------")

        # Create test input features
        test_goal_positions = np.tile(
            goal_position, (len(test_time_array), 1)
        )  # Shape: (100, 3)
        test_input = np.hstack(
            (test_time_array.reshape(-1, 1), test_goal_positions)
        )  # Shape: (100, 4)

        # Predict joint positions for the new goal position
        predicted_joint_positions_over_time = np.zeros(
            (len(test_time_array), 7)
        )  # Shape: (num_points, 7)

        for joint_idx in range(7):
            # Instantiate the model
            # model = MLP()
            # Load the saved model
            # model_filename = os.path.join(script_dir, f'neuralq{joint_idx+1}.pt')
            # model.load_state_dict(torch.load(model_filename))
            # model.eval()

            # Prepare the test input
            test_input_tensor = torch.from_numpy(
                test_input
            ).float()  # Shape: (num_points, 4)

            # Predict joint positions
            with torch.no_grad():
                predictions = (
                    models[joint_idx](test_input_tensor).numpy().flatten()
                )  # Shape: (num_points,)

            # Store the predicted joint positions
            predicted_joint_positions_over_time[:, joint_idx] = predictions

        # Get the final predicted joint positions (at the last time step)
        final_predicted_joint_positions = predicted_joint_positions_over_time[
            -1, :
        ]  # Shape: (7,)

        # Compute forward kinematics
        final_cartesian_pos, final_R = dyn_model.ComputeFK(
            final_predicted_joint_positions, controlled_frame_name
        )

        print(f"Goal position: {goal_position}")
        print(f"Computed cartesian position: {final_cartesian_pos}")
        print(
            f"Predicted joint positions at final time step: {final_predicted_joint_positions}"
        )

        # Compute position error
        position_error = np.linalg.norm(final_cartesian_pos - goal_position)
        print(
            f"Position error between computed position and goal: {position_error}"
        )
        mse = mean_squared_error(final_cartesian_pos, goal_position)
        print(
            f"MSE between computed position and goal: {mse}"
        )

        pos_errs.append(position_error)

        # Optional: Visualize the cartesian trajectory over time
        if visualize:
            cartesian_positions_over_time = []
            for i in range(len(test_time_array)):
                joint_positions = predicted_joint_positions_over_time[i, :]
                cartesian_pos, _ = dyn_model.ComputeFK(
                    joint_positions, controlled_frame_name
                )
                cartesian_positions_over_time.append(cartesian_pos.copy())

            cartesian_positions_over_time = np.array(
                cartesian_positions_over_time
            )  # Shape: (num_points, 3)

            # Plot x, y, z positions over time
            plt.figure(figsize=(10, 5))
            plt.plot(
                test_time_array,
                cartesian_positions_over_time[:, 0],
                label="X Position",
            )
            plt.plot(
                test_time_array,
                cartesian_positions_over_time[:, 1],
                label="Y Position",
            )
            plt.plot(
                test_time_array,
                cartesian_positions_over_time[:, 2],
                label="Z Position",
            )
            plt.scatter([test_time_array[-1]] * 3, goal_position, s=50, c='red', label="Goal position")
            plt.xlabel("Time (s)")
            plt.ylabel("Cartesian Position (m)")
            plt.title("Predicted Cartesian Positions Over Time")
            plt.legend()
            plt.grid(True)
            os.makedirs(f"{DIR}/testing", exist_ok=True)
            plt.savefig(f"{DIR}/testing/predicted_pos_goal_{ii+1}.{EXT}")
            #plt.show()

            # Plot the trajectory in 3D space
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(
                cartesian_positions_over_time[:, 0],
                cartesian_positions_over_time[:, 1],
                cartesian_positions_over_time[:, 2],
                label="Predicted Trajectory",
            )
            ax.scatter(
                goal_position[0],
                goal_position[1],
                goal_position[2],
                color="red",
                label="Goal Position",
            )
            ax.set_xlabel("X Position (m)")
            ax.set_ylabel("Y Position (m)")
            ax.set_zlabel("Z Position (m)")
            ax.set_title("Predicted Cartesian Trajectory")
            plt.legend()
            plt.savefig(f"{DIR}/testing/predicted_trajectory_{ii+1}.{EXT}")
            #plt.show()

    formatted_vector = [float(f"{x:.3f}") for x in pos_errs]
    print('Pos Errors: ')
    print(*formatted_vector, sep=' & ')
